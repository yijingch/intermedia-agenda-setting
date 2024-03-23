import pandas as pd
import numpy as np
# from nltk.stem import WordNetLemmatizer
from typing import List
import json
from src.utils.preprocessor import get_lemmas, get_stems
from src.utils.preprocessor import TOKENIZER
from src.utils.text import contractions
# import os, re 

class TopicDictionary():
    def __init__(
        self, 
        dictpath: str,
        relevance_col: str = "if_reasonable",
        min_relevance: int = 1,
        lemmatize: bool = False,
        stemming: bool = False,
        weight_col: str = "",
        post_mturk_change: dict = {"remove":[], "add":[]},
        topic_idx_ext: List = [],
        # word_idx_ext: List = [], 
    ) -> None:
        """_summary_

        Args:
            dictpath (str): the filepath to the dictionary; expected format: csv; expected [required] columns: topic, word, if_reasonable 

            relevance_col (str, optional): the column name indicating the strength of relevance, default set to "if_reasonable"

            min_relevance (int, optional): the minimum value of relevance we need in the dictioanry (to filter the dataframe), defalt set to 1

            lemmatize: a bool switch to indicate whether we need to lemmatize the dictionary, default set to False
        """

        df = pd.read_csv(dictpath, sep="\t")
        df = df[df[relevance_col]>=min_relevance].reset_index().drop(columns=["index"])

        if len(post_mturk_change["remove"]) > 0:
            for t,w in post_mturk_change["remove"]:
                df = df.drop(df[(df["topic"]==t)&(df["word"]==w)].index)
        if len(post_mturk_change["add"]) > 0:
            for t,w,rel in post_mturk_change["add"]:
                new_row = {"topic":t, "word":w, relevance_col:rel}
                df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

        df["word"] = df["word"].map(lambda x: str(x).lower().strip())
        for char in ["_", "-", " & ", "/"]:
            df["word"] = df["word"].map(lambda x: str(x).replace(char, " "))
        df["word"] = df["word"].map(lambda x: contractions.expand(x, drop_ownership=True))
        df["tokens"] =  df["word"].map(lambda x: x.split(" "))
        df["n_tokens"] = df["tokens"].map(lambda x: len(x))
        
        if lemmatize:
            df["lemmas"] = df["tokens"].map(lambda x: get_lemmas(x))
            df["word"] = df["lemmas"].map(lambda x: " ".join(x))
        if stemming:
            df["stems"] = df["tokens"].map(lambda x: get_stems(x))
            df["word"] = df["stems"].map(lambda x: " ".join(x))


        df = df.drop_duplicates(subset=["topic","word"])
        df = df.sort_values(by="n_tokens", ascending=False) 
        # sort df by the number of tokens so that later
        # when counting keywords, we'll start from the longest keyword (trigram first, bigram, unigram, ...)

        self.topic2index = {}
        self.index2topic = {}
        self.word2index = {}
        self.index2word = {}
        self.nonspace_word2index = {}
        self.nonspace_index2word = {}

        self.df = df

        if len(topic_idx_ext) == 0:
            self.topics = list(df.topic.unique())
            self.topics.append("no_topic")
            self.n_topics = df.topic.nunique() + 1
            for i,t in enumerate(self.topics):
                self.index2topic[i] = t
                self.topic2index[t] = i
        else:
            with open(topic_idx_ext[0], "r") as jsonf:
                self.index2topic = json.load(jsonf)
            with open(topic_idx_ext[1], "r") as jsonf:
                self.topic2index = json.load(jsonf)
            self.topics = self.topic2index.keys()
            self.n_topics = len(self.topics)

        # TODO: implement external index source for words too
        self.words = df.word.unique()
        self.n_words = df.word.nunique()
        for i,w in enumerate(self.words):
            self.index2word[i] = w
            self.word2index[w] = i
        for i,w in enumerate(self.words):
            w = w.replace(" ", "")
            self.nonspace_index2word[i] = w
            self.nonspace_word2index[w] = i

        print("Successfully loaded dictionary!")
        print("\t# of unique topics:", len(self.topics))
        print("\t# of unique words:", len(self.words))

        # construct topic word matrix
        self.topword_matrix = np.zeros((self.n_topics, self.n_words))
        if len(weight_col) > 0:  # apply different weights on topic keywords
            for i,row in df.iterrows():
                indx_t = self.topic2index[row.topic]
                indx_w = self.word2index[row.word]
                self.topword_matrix[indx_t, indx_w] += row[weight_col]
        else:
            for i,row in df.iterrows():
                indx_t = self.topic2index[row.topic]
                indx_w = self.word2index[row.word]
                self.topword_matrix[indx_t, indx_w] += 1

    def construct_overlap_matrix(self) -> None:
        self.overlap_mat = np.zeros((self.n_words, self.n_words))
        for i in self.words:
            for j in self.words:
                if (" " + i in j) or (i + " " in j) or (" " + i + " " in j):
                    idx_i = self.word2index[i]
                    idx_j = self.word2index[j]
                    self.overlap_mat[idx_i, idx_j] += 1
        # self.overlap_mat -= np.identity(self.n_words)
        # construct overlap matrix --> for later deduction to prevent over-counting

    def convert_to_json(self, output_fpath) -> None:
        aggr_func = {"word": lambda x: list(x)}
        output_df = self.df[["word", "topic"]].groupby("topic").aggregate(aggr_func).reset_index()
        output_df = output_df.rename(columns={"word": "words"})
        output_df.to_json(output_fpath, orient="records", indent=1)

        

    