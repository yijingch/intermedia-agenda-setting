import pandas as pd
import numpy as np
from collections import Counter
from datetime import date
import traceback
import os, re, glob
import multiprocess as mp

import matplotlib.pyplot as plt
import matplotlib

import nltk
from src.utils.data_loader import Headlines, Surveys 
from src.utils.dict_loader import TopicDictionary


def adjust_pos(pos1, pos2):
    p1_remove = []
    p2_remove = []
    for p1 in pos1:
        len1 = p1[1] - p1[0] 
        for p2 in pos2:
            len2 = p2[1] - p2[0] 
            if bool_overlap(p1, p2):
                if len1 < len2:
                    p1_remove.append(p1)
                else:
                    p2_remove.append(p2)
    for p1r in p1_remove:
        pos1.remove(p1r)
    for p2r in p2_remove:
        pos2.remove(p2r)
    return pos1, pos2

def bool_overlap(intv1, intv2):
    max_lower = max(intv1[0], intv2[0])
    min_upper = min(intv1[1], intv2[1])
    if max_lower < min_upper:
        return True
    else:
        return False

def search_all_pos(w, text):
    p = re.compile(r"\b{}\b".format(w))
    m = p.search(text)
    pos = []
    while m:
        start,end = m.span()
        pos.append((start, end))
        m = p.search(text, start+1)
    return pos

class DictBasedTopicModel():
    def __init__(self, dictionary:TopicDictionary, text_input, text_type) -> None:
        self.dictionary = dictionary
        self.text_input = text_input # class object: Headlines or Surveys
        self.text_type = text_type # headline or survey

    def build_wordvec(self, text):
        wordvec = np.zeros(self.dictionary.n_words)
        occur_pos = {}
        occur_words_idx = []
        for w in self.dictionary.words:
            pos1 = search_all_pos(w, text)
            idx_w = self.dictionary.word2index[w]
            occur_pos[idx_w] = pos1
            if len(pos1) > 0:
                occur_words_idx.append(idx_w)

        if len(occur_words_idx) > 0:
            for idx1 in occur_words_idx:
                overlap_vec = self.dictionary.overlap_mat[idx1]
                overlap_words_idx = np.nonzero(overlap_vec)[0]
                if len(overlap_words_idx) > 0:
                    for idx2 in overlap_words_idx:
                        if len(occur_pos[idx2]) > 0:
                            pos1 = occur_pos[idx1]
                            pos2 = occur_pos[idx2]
                            pos1, pos2 = adjust_pos(pos1, pos2)
        for i,pos in occur_pos.items():
            wordvec[i] = len(pos)
        return wordvec

    def build_wordvec_df(self, drop_no_topic:bool=False, normalize_vec:bool=False, save_output:bool=False, output_cache_fpath="") -> None:
        with mp.Pool(mp.cpu_count()-2) as pool:
            self.text_input.df_cand1["wordvec"] = pool.map(self.build_wordvec, self.text_input.df_cand1["cleaned_textbody"])
            print("Finished counting topic keywords:", self.text_input.df_cand1_label)
            self.text_input.df_cand2["wordvec"] = pool.map(self.build_wordvec, self.text_input.df_cand2["cleaned_textbody"])
            print("Finished counting topic keywords:", self.text_input.df_cand2_label)

        self.text_input.df_cand1["sum"] = self.text_input.df_cand1["wordvec"].map(lambda x: sum(x))
        original_len1 = len(self.text_input.df_cand1)
        if drop_no_topic:
            self.text_input.df_cand1 = self.text_input.df_cand1[self.text_input.df_cand1["sum"]>0].drop(columns="sum")
            filtered_len1 = len(self.text_input.df_cand1)
            print(f"Rate of coverage for {self.text_input.df_cand1_label}:", filtered_len1/original_len1)

        self.text_input.df_cand2["sum"] = self.text_input.df_cand2["wordvec"].map(lambda x: sum(x))
        original_len2 = len(self.text_input.df_cand2)
        if drop_no_topic:
            self.text_input.df_cand2 = self.text_input.df_cand2[self.text_input.df_cand2["sum"]>0].drop(columns="sum")
            filtered_len2 = len(self.text_input.df_cand2)
            print(f"Rate of coverage for {self.text_input.df_cand2_label}:", filtered_len2/original_len2)

        if save_output:
            date_string = date.today().strftime("%m%d%y")
            if not os.path.exists(f"{output_cache_fpath}/{self.text_type}/"):
                os.mkdir(f"{output_cache_fpath}/{self.text_type}/")
            else:
                pass
            try:
                print("Now we can save wordvecs as well! [new!]...")
                self.text_input.df_cand1[["date","domain","path","textbody","wordvec"]].to_pickle(f"{output_cache_fpath}/{self.text_type}/{date_string}_{self.text_input.df_cand1_label}_wordvec_cache.pkl")
                self.text_input.df_cand2[["date","domain","path","textbody","wordvec"]].to_pickle(f"{output_cache_fpath}/{self.text_type}/{date_string}_{self.text_input.df_cand2_label}_wordvec_cache.pkl")
            except:                 
                self.text_input.df_cand1[["date","textbody","wordvec"]].to_pickle(f"{output_cache_fpath}/{self.text_type}/{date_string}_{self.text_input.df_cand1_label}_wordvec_cache.pkl")
                self.text_input.df_cand2[["date","textbody","wordvec"]].to_pickle(f"{output_cache_fpath}/{self.text_type}/{date_string}_{self.text_input.df_cand2_label}_wordvec_cache.pkl")

    def build_topvec(self, wordvec):
        topvec = np.dot(self.dictionary.topword_matrix, np.array(wordvec))
        return topvec 

    def build_topvec_df(self, normalize_vec:bool=False, save_output:bool=False, output_cache_fpath="") -> None:
        with mp.Pool(mp.cpu_count()-2) as pool:
            self.text_input.df_cand1["topvec"] = pool.map(self.build_topvec, self.text_input.df_cand1["wordvec"])
            print("Finished computing topic vector:", self.text_input.df_cand1_label)
            self.text_input.df_cand2["topvec"] = pool.map(self.build_topvec, self.text_input.df_cand2["wordvec"])
            print("Finished computing topic vector:", self.text_input.df_cand2_label)
        if normalize_vec:
            self.text_input.df_cand1["topvec"] = self.text_input.df_cand1["topvec"].map(lambda x: x/np.sum(x))
            self.text_input.df_cand2["topvec"] = self.text_input.df_cand2["topvec"].map(lambda x: x/np.sum(x))

        if save_output:
            date_string = date.today().strftime("%m%d%y")
            if not os.path.exists(f"{output_cache_fpath}/{self.text_type}/"):
                os.mkdir(f"{output_cache_fpath}/{self.text_type}/")
            else:
                pass
            try:
                # print("Now saving domain names as well [new!]...")
                self.text_input.df_cand1[["date","domain","path","textbody","topvec"]].to_pickle(f"{output_cache_fpath}/{self.text_type}/{date_string}_{self.text_input.df_cand1_label}_topvec_cache.pkl")
                self.text_input.df_cand2[["date","domain","path","textbody","topvec"]].to_pickle(f"{output_cache_fpath}/{self.text_type}/{date_string}_{self.text_input.df_cand2_label}_topvec_cache.pkl")
            except:                 
                self.text_input.df_cand1[["date","textbody","topvec"]].to_pickle(f"{output_cache_fpath}/{self.text_type}/{date_string}_{self.text_input.df_cand1_label}_topvec_cache.pkl")
                self.text_input.df_cand2[["date","textbody","topvec"]].to_pickle(f"{output_cache_fpath}/{self.text_type}/{date_string}_{self.text_input.df_cand2_label}_topvec_cache.pkl")