"""Code for summing topic or keyword vectors (during the entire period) without applying any weekly/daily aggregation"""

import pandas as pd 
import numpy as np 
from typing import List, Dict, Any
from datetime import datetime

from src.utils.dict_loader import TopicDictionary
from src.utils.downstream_process import merge_topics_from_arr, collapse_general_controversies, get_majority
from src.utils.downstream_process import assign_popularity_weight


def sum_headline_topvec(
        output_df:pd.DataFrame, 
        raw_df:pd.DataFrame,
        cand:str,
        dictionary:TopicDictionary,
        select_domains:List = [],
        weight_by_popularity:bool = False,
        popularity_dict:Dict = {},
        print_info:bool = False, 
        # normalize_by_day:bool = True,
        normalize_by_snapshot:bool = True) -> np.array:
    if len(select_domains) > 0:
        process_df = output_df[output_df["domain"].isin(select_domains)].copy()
        raw_df_select = raw_df[raw_df["domain"].isin(select_domains)].copy()
    else:
        process_df = output_df.copy()
        raw_df_select = raw_df.copy()

    # manipulate topic output: merge and collapse topics
    process_df["topvec"] = process_df["topvec"].map(lambda x:merge_topics_from_arr(x, merge_to="government_ops", to_merge="election_campaign", dictionary=dictionary))
    process_df["topvec"] = process_df["topvec"].map(lambda x: collapse_general_controversies(x, cand, dictionary))
    process_df["majority_topvec"] = process_df["topvec"].map(lambda x: get_majority(x))

    if weight_by_popularity:
        if len(popularity_dict) == 0:
            print("If hoping to use popularity weight, please feed in a popularity dictionary (currently empty)!")
        else:
            process_df["pop_weight"] = process_df["domain"].map(lambda x: assign_popularity_weight(x, popularity_dict=popularity_dict))
            process_df["majority_topvec"] = process_df.apply(lambda x: x["topvec"]*x["pop_weight"], axis=1)
    
    # merge topic output with the full headlines parsed from data (now we do not drop anything)
    aggr_func = {"majority_topvec": lambda x: np.sum(np.array(list(x)), axis=0), "path": lambda x: len(set(x))}
    full_process_df = raw_df_select.merge(process_df[["textbody","majority_topvec"]], how="left", on="textbody").dropna(subset="majority_topvec")

    # normalize topic vectors by domain by day 
    full_aggr_df = full_process_df.groupby(["domain","date"]).agg(aggr_func).reset_index()
    if normalize_by_snapshot:
        # full_aggr_df["majority_topvec"] = full_aggr_df["majority_topvec"].map(lambda x: normalize(x))
        full_aggr_df["majority_topvec"] = full_aggr_df["majority_topvec"]/full_aggr_df["path"]
    if print_info:
        print("\t# of unique domains:", full_aggr_df["domain"].nunique())

    sum_arr = np.sum(full_aggr_df["majority_topvec"].tolist(), axis=0)
    assert len(sum_arr) == dictionary.n_topics, "Wrong output shape! Please check if there's a bug."
    return sum_arr
    

def sum_survey_topvec(
        output_df:pd.DataFrame,
        cand:str, 
        dictionary:TopicDictionary,
        select_leaning:str="", 
        apply_weights:bool = True) -> np.array:
    
    if len(select_leaning) > 0:
        process_df = output_df[output_df["partyln"]==select_leaning].copy()
    else:
        process_df = output_df.copy()
    
    process_df["topvec"] = process_df["topvec"].map(lambda x:merge_topics_from_arr(x, merge_to="government_ops", to_merge="election_campaign", dictionary=dictionary))
    process_df["topvec"] = process_df["topvec"].map(lambda x: collapse_general_controversies(x, cand, dictionary))

    # get the majority vote 
    process_df["majority_topvec"] = process_df["topvec"].map(lambda x: get_majority(x))
    if apply_weights:
        process_df["majority_topvec"] = process_df.apply(lambda x: x["majority_topvec"]*x["weights"], axis=1)

    sum_arr = np.sum(process_df["majority_topvec"].tolist(), axis=0)
    assert len(sum_arr) == dictionary.n_topics, "Wrong output shape! Please check if there's a bug."
    return sum_arr


def sum_tweet_topvec(
        output_df:pd.DataFrame,
        cand:str,
        dictionary:TopicDictionary) -> np.array:
    
    process_df = output_df.copy()
    process_df["topvec"] = process_df["topvec"].map(lambda x:merge_topics_from_arr(x, merge_to="government_ops", to_merge="election_campaign", dictionary=dictionary))
    process_df["topvec"] = process_df["topvec"].map(lambda x: collapse_general_controversies(x, cand, dictionary))
    process_df["majority_topvec"] = process_df["topvec"].map(lambda x: get_majority(x))

    sum_arr = np.sum(process_df["majority_topvec"].tolist(), axis=0)
    assert len(sum_arr) == dictionary.n_topics, "Wrong output shape! Please check if there's a bug."
    return sum_arr


def bootstrap_sum_topvec(
        data_source:str,
        output_df:pd.DataFrame, 
        cand:str,
        dictionary:TopicDictionary,
        raw_df:pd.DataFrame = None,
        select_domains:List = [],
        select_leaning:str = "",
        weight_by_popularity:bool = False,
        popularity_dict:Dict = {},
        # normalize_by_day:bool = True,
        normalize_by_snapshot:bool = True, 
        apply_survey_weights:bool = True,
        bootstrap_runs:int = 200,
        sample_frac:float = .8,) -> np.array:
    
    bstr_arr = []
    if data_source == "headline":
        for i in range(bootstrap_runs):
            if i%20==0: print("progress:", i/bootstrap_runs)    
            bstr_raw_df = raw_df.sample(frac=sample_frac)
            bstr_sum_arr = sum_headline_topvec(
                output_df=output_df, 
                raw_df=bstr_raw_df, 
                cand=cand, 
                dictionary=dictionary,
                select_domains=select_domains,
                weight_by_popularity=weight_by_popularity,
                popularity_dict=popularity_dict,
                normalize_by_snapshot=normalize_by_snapshot)
            bstr_arr.append(bstr_sum_arr)
    elif data_source == "survey":
        for i in range(bootstrap_runs):
            if i%20==0: print("progress:", i/bootstrap_runs)
            bstr_output_df = output_df.sample(frac=sample_frac)
            bstr_sum_arr = sum_survey_topvec(
                output_df=bstr_output_df,
                cand=cand,
                dictionary=dictionary,
                select_leaning=select_leaning, 
                apply_weights=apply_survey_weights)
            bstr_arr.append(bstr_sum_arr)
    elif data_source == "tweet":
        for i in range(bootstrap_runs):
            if i%20==0: print("progress:", i/bootstrap_runs)
            bstr_output_df = output_df.sample(frac=sample_frac)
            bstr_sum_arr = sum_tweet_topvec(
                output_df=bstr_output_df,
                cand=cand,
                dictionary=dictionary)
            bstr_arr.append(bstr_sum_arr)
    else:
        print("Please enter a valid data source! (headline, survey, tweet)")

    bstr_arr = np.array(bstr_arr)
    assert bstr_arr.shape == (bootstrap_runs, dictionary.n_topics), "Wrong output array shape!"
    print(bstr_arr.shape)
    return bstr_arr

def sum_headline_wordvec(
        output_df:pd.DataFrame,
        raw_df:pd.DataFrame,
        dictionary:TopicDictionary,
        select_domains:List = [],
        weight_by_popularity:bool = False,
        popularity_dict:Dict={},
        print_info:bool = False, 
        # normalize_by_day:bool = True
        normalize_by_snapshot:bool = True,
        ) -> np.array:
    
    if len(select_domains) > 0:
        process_df = output_df[output_df["domain"].isin(select_domains)].copy()
        raw_df_select = raw_df[raw_df["domain"].isin(select_domains)].copy()
    else:
        process_df = output_df.copy()
        raw_df_select = raw_df.copy()

    if weight_by_popularity:
        if len(popularity_dict) == 0:
            print("If hoping to use popularity weight, please feed in a popularity dictionary (currently empty)!")
        else:
            process_df["pop_weight"] = process_df["domain"].map(lambda x:assign_popularity_weight(x, popularity_dict=popularity_dict))
            process_df["wordvec"] = process_df.apply(lambda x: x["wordvec"]*x["pop_weight"], axis=1)
    
    # merge word count vectors with the full headlines parsed from data (we do not drop anything)
    aggr_func = {"wordvec": lambda x: np.sum(np.array(list(x)), axis=0), "path": lambda x: len(set(x))}
    full_process_df = raw_df_select.merge(process_df[["textbody", "wordvec"]], how="left", on="textbody").dropna(subset="wordvec")

    # normalize word vectors by domain by day 
    full_aggr_df = full_process_df.groupby(["domain", "date"]).agg(aggr_func).reset_index()
    if normalize_by_snapshot:
        # full_aggr_df["wordvec"] = full_aggr_df["wordvec"].map(lambda x: normalize(x))
        full_aggr_df["wordvec"] = full_aggr_df["wordvec"]/full_aggr_df["path"]
    full_aggr_df["date"] = pd.to_datetime(full_aggr_df["date"])

    if print_info:
        print("\t# of unique domains:", full_aggr_df["domain"].nunique())

    sum_arr = np.sum(full_aggr_df["wordvec"].tolist(), axis=0)
    # print(sum_arr.shape)
    assert len(sum_arr) == dictionary.n_words, "Wrong output shape! Please check if there's a bug."
    return sum_arr


def sum_survey_wordvec(
        output_df:pd.DataFrame,
        dictionary:TopicDictionary,
        select_leaning:str = "", 
        apply_weights:bool = True) -> np.array:
    
    if len(select_leaning) > 0:
        process_df = output_df[output_df["partyln"]==select_leaning].copy()
    else:
        process_df = output_df.copy()

    if apply_weights:
        process_df["wordvec"] = process_df.apply(lambda x: x["wordvec"]*x["weights"], axis=1)
    sum_arr = np.sum(process_df["wordvec"].tolist(), axis=0)

    assert len(sum_arr) == dictionary.n_words, "Wrong output shape! Please check if there's a bug."
    return sum_arr


def sum_tweet_wordvec(
        output_df:pd.DataFrame,
        dictionary:TopicDictionary) -> np.array:
    sum_arr = np.sum(output_df["wordvec"].tolist(), axis=0)
    assert len(sum_arr) == dictionary.n_words, "Wrong output shape! Please check if there's a bug."
    return sum_arr


def bootstrap_sum_wordvec(
        data_source:str,
        output_df:pd.DataFrame,
        dictionary:TopicDictionary,
        raw_df:pd.DataFrame = None,
        select_domains:List = [],
        select_leaning:str = "",
        weight_by_popularity:bool = False,
        popularity_dict:Dict = {},
        # normalize_by_day:bool = True,
        normalize_by_snapshot:bool = True,
        apply_survey_weights:bool = True,
        bootstrap_runs:int = 200,
        sample_frac:float = .8) -> np.array:
    bstr_arrs = []
    if data_source == "headline":
        for i in range(bootstrap_runs):
            if i%20 == 0: print("progress:", i/bootstrap_runs)
            bstr_raw_df = raw_df.sample(frac=sample_frac)
            bstr_sum_arr = sum_headline_wordvec(
                output_df=output_df, 
                raw_df=bstr_raw_df,
                dictionary=dictionary,
                select_domains=select_domains,
                weight_by_popularity=weight_by_popularity,
                popularity_dict=popularity_dict, 
                normalize_by_snapshot=normalize_by_snapshot)
            bstr_arrs.append(bstr_sum_arr)
    elif data_source == "survey":
        for i in range(bootstrap_runs):
            if i%20 == 0: print("progress:", i/bootstrap_runs)
            bstr_output_df = output_df.sample(frac=sample_frac)
            bstr_sum_arr = sum_survey_wordvec(
                output_df=bstr_output_df,
                dictionary=dictionary,
                select_leaning=select_leaning, 
                apply_weights=apply_survey_weights)
            bstr_arrs.append(bstr_sum_arr)
    elif data_source == "tweet":
        for i in range(bootstrap_runs):
            if i%20 == 0: print("progress:", i/bootstrap_runs)
            bstr_output_df = output_df.sample(frac=sample_frac)
            bstr_sum_arr = sum_tweet_wordvec(
                output_df=bstr_output_df,
                dictionary=dictionary)
            bstr_arrs.append(bstr_sum_arr)
    else:
        print("Please enter a valid data source! (headline, survey, tweet)")   
    return bstr_arrs
