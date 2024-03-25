"""Code for aggregating topic or keyword vectors (with the bootstrapping option)"""

import pandas as pd 
import numpy as np 
from typing import List, Dict, Any
from datetime import datetime

from src.utils.dict_loader import TopicDictionary
from src.utils.downstream_process import merge_topics_from_arr, collapse_general_controversies, get_majority
from src.utils.downstream_process import trim_period, assign_popularity_weight, normalize


def load_model_output(fpath:str, start, end, trim:bool=True, strip_time=True) -> pd.DataFrame:
    df = pd.read_pickle(fpath)
    if strip_time:
        df["date"] = df["date"].map(lambda x: str(x)[:10])
    if not isinstance(df["date"].tolist()[0], datetime):
        df["date"] = pd.to_datetime(df["date"])
    if trim:
        df = trim_period(df, start=start, end=end)
    return df 

def aggregate_headline_topvec(
        output_df:pd.DataFrame, 
        raw_df:pd.DataFrame,
        aggr_unit:str,
        cand:str,
        dictionary:TopicDictionary,
        select_domains:List = [],
        force_time_window:List = [],
        weight_by_popularity:bool = False,
        popularity_dict:Dict = {},
        print_info:bool = False, 
        # normalize_by_day:bool = True,
        normalize_by_snapshot:bool = True) -> pd.DataFrame:
    """Aggregate headline topic vectors by a given time unit 

    Args:
        output_df (pd.DataFrame): the topic vector output (for all unique headlines)
        raw_df (pd.DataFrame): the complete headline dataframe parsed from Wayback.
        aggr_unit (str): the time unit for aggregation ("D" for day and "W" for week)
        cand (str): the candidate for this dataframe (for collapsing general_controversies).
        dictionary (TopicDictionary): the topic dictionary to use.
        select_domains (List, optional): select a certain set of domains. Defaults to [].
        force_time_window(List, optional): force the output into a specified time window (input should be a list of dates). Defaults to [].
        weight_by_popularity (bool, optional): whether to weight the topic count by the domain popularity. Defaults to False.
        popularity_dict (Dict, optional): if weight_by_popularity is True, provide a dictionary of popularity that will be used as weights to multiply with topic counts. Defaults to {}.

    Returns:
        pd.DataFrame: return an aggregated dataframe 
    """
    # if select a certain number of domains
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
    full_aggr_df["date"] = pd.to_datetime(full_aggr_df["date"])
    if print_info:
        print("\t# of unique domains:", full_aggr_df["domain"].nunique())

    # aggregate output by a given time unit
    aggr_func_by_unit = {"majority_topvec": lambda x: np.sum(list(x), axis=0)}
    full_aggr_df_by_unit = full_aggr_df.set_index("date").resample(aggr_unit).agg(aggr_func_by_unit).reset_index()
    full_aggr_df_by_unit["majority_topvec"] = full_aggr_df_by_unit["majority_topvec"].map(lambda x: x if np.sum(x) > 0 else np.zeros(dictionary.n_topics))
    if print_info:
        print("\tstart:", full_aggr_df_by_unit["date"].min())
        print("\tend:", full_aggr_df_by_unit["date"].max())

    if len(force_time_window) > 0:
        full_aggr_df_by_unit_fix = pd.DataFrame()
        full_aggr_df_by_unit_fix["date"] = force_time_window
        full_aggr_df_by_unit_fix = full_aggr_df_by_unit_fix.merge(full_aggr_df_by_unit, how="left", on="date").fillna(0)
        full_aggr_df_by_unit_fix["majority_topvec"] = full_aggr_df_by_unit_fix["majority_topvec"].map(lambda x: x if np.sum(x) > 0 else np.zeros(dictionary.n_topics))
        return full_aggr_df_by_unit_fix
    else:
        return full_aggr_df_by_unit


def aggregate_survey_topvec(
        output_df:pd.DataFrame,
        aggr_unit:str,
        cand:str, 
        dictionary:TopicDictionary,
        force_time_window:List = [],
        select_leaning:str="", 
        apply_weights:bool = True) -> pd.DataFrame:
    """Aggregate survey topic vectors by a given time unit.

    Args:
        output_df (pd.DataFrame): the topic vector output (for all unique survey response)
        aggr_unit (str): the time unit for aggregation ("D" for day and "W" for week)
        cand (str): the candidate for this dataframe (for collapsing general_controversies)
        dictionary (TopicDictionary): the topic dictionary to use.
        select_leaning (str, optional): select a certain group of respondents. Defaults to "".
        force_time_window(List, optional): force the output into a specified time window (input should be a list of dates). Defaults to [].

    Returns:
        pd.DataFrame: return an aggregated dataframe 
    """
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

    aggr_func = {"majority_topvec": lambda x: np.sum(list(x), axis=0)}
    process_df["date"] = pd.to_datetime(process_df["date"])
    process_df.set_index("date", inplace=True)
    aggr_df = process_df.resample(aggr_unit).agg(aggr_func).reset_index()
    aggr_df["majority_topvec"] = aggr_df["majority_topvec"].map(lambda x: x if np.sum(x) > 0 else np.zeros(dictionary.n_topics))

    if len(force_time_window) > 0:
        aggr_df_fix = pd.DataFrame()
        aggr_df_fix["date"] = force_time_window
        aggr_df_fix = aggr_df_fix.merge(aggr_df, how="left", on="date").fillna(0)
        aggr_df_fix["majority_topvec"] = aggr_df_fix["majority_topvec"].map(lambda x: x if np.sum(x) > 0 else np.zeros(dictionary.n_topics))
        return aggr_df_fix
    else:
        return aggr_df 


def aggregate_tweet_topvec(
        output_df:pd.DataFrame,
        aggr_unit:str,
        cand:str,
        dictionary:TopicDictionary,
        force_time_window:List = []) -> pd.DataFrame:
    """Aggregate tweet topic vectors by a given time unit 

    Args:
        output_df (pd.DataFrame): the topic vector output (for all unique survey response)
        aggr_unit (str): the time unit for aggregation ("D" for day and "W" for week)
        cand (str): the candidate for this dataframe (for collapsing general_controversies)
        dictionary (TopicDictionary): the topic dictionary to use.
        select_leaning (str, optional): select a certain group of respondents. Defaults to "".
        force_time_window(List, optional): force the output into a specified time window (input should be a list of dates). Defaults to [].

    Returns:
        pd.DataFrame: return an aggregated dataframe 
    """
    process_df = output_df.copy()
    process_df["topvec"] = process_df["topvec"].map(lambda x:merge_topics_from_arr(x, merge_to="government_ops", to_merge="election_campaign", dictionary=dictionary))
    process_df["topvec"] = process_df["topvec"].map(lambda x: collapse_general_controversies(x, cand, dictionary))
    process_df["majority_topvec"] = process_df["topvec"].map(lambda x: get_majority(x))

    aggr_func = {"majority_topvec": lambda x: np.sum(list(x), axis=0)}
    process_df["date"] = process_df["date"].map(lambda x: pd.to_datetime(str(x)[:10]))
    process_df.set_index("date", inplace=True)
    aggr_df = process_df.resample(aggr_unit).agg(aggr_func).reset_index()
    aggr_df["majority_topvec"] = aggr_df["majority_topvec"].map(lambda x: x if np.sum(x) > 0 else np.zeros(dictionary.n_topics))

    if len(force_time_window) > 0:
        aggr_df_fix = pd.DataFrame()
        aggr_df_fix["date"] = force_time_window
        aggr_df_fix = aggr_df_fix.merge(aggr_df, how="left", on="date").fillna(0)
        aggr_df_fix["majority_topvec"] = aggr_df_fix["majority_topvec"].map(lambda x: x if np.sum(x) > 0 else np.zeros(dictionary.n_topics))
        return aggr_df_fix
    else:
        return aggr_df 
    

def bootstrap_aggregate_topvec(
        data_source:str,
        output_df:pd.DataFrame, 
        aggr_unit:str,
        cand:str,
        dictionary:TopicDictionary,
        raw_df:pd.DataFrame = None,
        select_domains:List = [],
        select_leaning:str = "",
        force_time_window:List = [],
        weight_by_popularity:bool = False,
        popularity_dict:Dict = {},
        # normalize_by_day:bool = True,
        normalize_by_snapshot:bool = True, 
        apply_survey_weights:bool = True,
        bootstrap_runs:int = 200,
        sample_frac:float = .8,):
    """Perform bootstrapping in by-unit aggregation

    Args:
        data_source (str): the type of data to aggregate (headline, survey, or tweet)
        output_df (pd.DataFrame): the topic output file (for unique texts)
        aggr_unit (str): the time unit for aggregation ("D" for day and "W" for week)
        cand (str): the candidate for this dataframe (for collapsing general_controversies)
        dictionary (TopicDictionary): the topic dictionary to use.
        raw_df (pd.DataFrame): the complete headline dataframe parsed from Wayback. Defaults to None.
        select_domains (List, optional): select a certain set of domains. Defaults to [].
        select_leaning (str, optional): select a certain group of respondents. Defaults to "".
        force_time_window (List, optional): force the output into a specified time window (input should be a list of dates). Defaults to [].
        bootstrap_runs (int, optional): the number of rounds for bootstrapping. Defaults to 200.
        sample_frac (float, optional): the fraction of dataframe for sampling. Defaults to .8.

    Returns:
        _type_: _description_
    """
    bstr_arr = []
    if data_source == "headline":
        for i in range(bootstrap_runs):
            if i%20==0: print("progress:", i/bootstrap_runs)    
            bstr_raw_df = raw_df.sample(frac=sample_frac)
            bstr_aggr_df = aggregate_headline_topvec(
                output_df=output_df, 
                raw_df=bstr_raw_df, 
                aggr_unit=aggr_unit, 
                cand=cand, 
                dictionary=dictionary,
                select_domains=select_domains,
                force_time_window=force_time_window,
                weight_by_popularity=weight_by_popularity,
                popularity_dict=popularity_dict,
                normalize_by_snapshot=normalize_by_snapshot)
            bstr_arr.append(np.array(bstr_aggr_df["majority_topvec"].tolist()))
    elif data_source == "survey":
        for i in range(bootstrap_runs):
            if i%20==0: print("progress:", i/bootstrap_runs)
            bstr_output_df = output_df.sample(frac=sample_frac)
            bstr_aggr_df = aggregate_survey_topvec(
                output_df=bstr_output_df,
                aggr_unit=aggr_unit,
                cand=cand,
                dictionary=dictionary,
                force_time_window=force_time_window,
                select_leaning=select_leaning, 
                apply_weights=apply_survey_weights)
            bstr_arr.append(np.array(bstr_aggr_df["majority_topvec"].tolist()))
    elif data_source == "tweet":
        for i in range(bootstrap_runs):
            if i%20==0: print("progress:", i/bootstrap_runs)
            bstr_output_df = output_df.sample(frac=sample_frac)
            bstr_aggr_df = aggregate_tweet_topvec(
                output_df=bstr_output_df,
                aggr_unit=aggr_unit,
                cand=cand,
                dictionary=dictionary,
                force_time_window=force_time_window)
            bstr_arr.append(np.array(bstr_aggr_df["majority_topvec"].tolist()))
    else:
        print("Please enter a valid data source! (headline, survey, tweet)")

    bstr_arr = np.array(bstr_arr)
    if len(force_time_window) > 0:
        assert bstr_arr.shape == (bootstrap_runs, len(force_time_window), dictionary.n_topics), "Wrong output array shape!"
        print(bstr_arr.shape)
    return bstr_arr


def aggregate_headline_wordvec(
        output_df:pd.DataFrame,
        raw_df:pd.DataFrame,
        aggr_unit:str,
        dictionary:TopicDictionary,
        select_domains:List = [],
        force_time_window:List = [],
        weight_by_popularity:bool = False,
        popularity_dict:Dict={},
        print_info:bool = False,
        # normalize_by_day:bool = True,
        normalize_by_snapshot:bool = True) -> pd.DataFrame:
    
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

    # aggregate output by a given time unit 
    aggr_func_by_unit = {"wordvec": lambda x: np.sum(list(x), axis=0)}
    full_aggr_df_by_unit = full_aggr_df.set_index("date").resample(aggr_unit).agg(aggr_func_by_unit).reset_index()
    full_aggr_df_by_unit["wordvec"] = full_aggr_df_by_unit["wordvec"].map(lambda x: x if np.sum(x) > 0 else np.zeros(dictionary.n_words))
    if print_info:
        print("\tstart:", full_aggr_df_by_unit["date"].min())
        print("\tend:", full_aggr_df_by_unit["date"].max())
    
    if len(force_time_window) > 0:
        full_aggr_df_by_unit_fix = pd.DataFrame()
        full_aggr_df_by_unit_fix["date"] = force_time_window 
        full_aggr_df_by_unit_fix = full_aggr_df_by_unit_fix.merge(full_aggr_df_by_unit, how="left", on="date").fillna(0)
        full_aggr_df_by_unit_fix["wordvec"] = full_aggr_df_by_unit_fix["wordvec"].map(lambda x: x if np.sum(x) > 0 else np.zeros(dictionary.n_words))
        return full_aggr_df_by_unit_fix 
    else:
        return full_aggr_df_by_unit 
    

def aggregate_survey_wordvec(
        output_df:pd.DataFrame,
        aggr_unit:str, 
        dictionary:TopicDictionary,
        force_time_window:List = [],
        select_leaning:str = "",
        apply_weights:bool = True) -> pd.DataFrame:
    
    if len(select_leaning) > 0:
        process_df = output_df[output_df["partyln"]==select_leaning].copy()
    else:
        process_df = output_df.copy()

    if apply_weights:
        process_df["wordvec"] = process_df.apply(lambda x: x["wordvec"]*x["weights"], axis=1)

    aggr_func = {"wordvec": lambda x: np.sum(list(x), axis=0)}
    process_df["date"] = pd.to_datetime(process_df["date"])
    process_df.set_index("date", inplace=True)
    aggr_df = process_df.resample(aggr_unit).agg(aggr_func).reset_index()
    aggr_df["wordvec"] = aggr_df["wordvec"].map(lambda x: x if np.sum(x) > 0 else np.zeros(dictionary.n_words))

    if len(force_time_window) > 0:
        aggr_df_fix = pd.DataFrame()
        aggr_df_fix["date"] = force_time_window
        aggr_df_fix = aggr_df_fix.merge(aggr_df, how="left", on="date").fillna(0)
        aggr_df_fix["wordvec"] = aggr_df_fix["wordvec"].map(lambda x: x if np.sum(x) > 0 else np.zeros(dictionary.n_words))
        return aggr_df_fix
    else:
        return aggr_df     

def aggregate_tweet_wordvec(
        output_df:pd.DataFrame,
        aggr_unit:str, 
        dictionary:TopicDictionary,
        force_time_window:List = []) -> pd.DataFrame:

    process_df = output_df.copy()
    
    aggr_func = {"wordvec": lambda x: np.sum(list(x), axis=0)}
    process_df["date"] = process_df["date"].map(lambda x: pd.to_datetime(str(x)[:10]))
    process_df.set_index("date", inplace=True)
    aggr_df = process_df.resample(aggr_unit).agg(aggr_func).reset_index()
    aggr_df["wordvec"] = aggr_df["wordvec"].map(lambda x: x if np.sum(x) > 0 else np.zeros(dictionary.n_words))

    if len(force_time_window) > 0:
        aggr_df_fix = pd.DataFrame()
        aggr_df_fix["date"] = force_time_window
        aggr_df_fix = aggr_df_fix.merge(aggr_df, how="left", on="date").fillna(0)
        aggr_df_fix["wordvec"] = aggr_df_fix["wordvec"].map(lambda x: x if np.sum(x) else np.zeros(dictionary.n_words))
        return aggr_df_fix 
    else:
        return aggr_df 

def bootstrap_aggregate_wordvec(
        data_source:str,
        output_df:pd.DataFrame,
        aggr_unit:str,
        dictionary:TopicDictionary,
        raw_df:pd.DataFrame = None, 
        select_domains:List = [],
        select_leaning:str = "",
        force_time_window:List = [],
        weight_by_popularity:bool = False,
        popularity_dict:Dict = {},
        # normalize_by_day:bool = True,
        normalize_by_snapshot:bool = True, 
        apply_survey_weights:bool = True,
        bootstrap_runs:int = 200,
        sample_frac:float = .8,):
    bstr_arr = []
    if data_source == "headline":
        for i in range(bootstrap_runs):
            if i%20 == 0: print("progress:", i/bootstrap_runs)
            bstr_raw_df = raw_df.sample(frac=sample_frac)
            bstr_aggr_df = aggregate_headline_wordvec(
                output_df=output_df, 
                raw_df=bstr_raw_df,
                aggr_unit=aggr_unit,
                dictionary=dictionary,
                select_domains=select_domains,
                force_time_window=force_time_window,
                weight_by_popularity=weight_by_popularity,
                popularity_dict=popularity_dict,
                normalize_by_snapshot=normalize_by_snapshot)
            bstr_arr.append(np.array(bstr_aggr_df["wordvec"].tolist()))
    elif data_source == "survey":
        for i in range(bootstrap_runs):
            if i%20 == 0: print("progress:", i/bootstrap_runs)
            bstr_output_df = output_df.sample(frac=sample_frac)
            bstr_aggr_df = aggregate_survey_wordvec(
                output_df=bstr_output_df,
                aggr_unit=aggr_unit,
                dictionary=dictionary,
                force_time_window=force_time_window,
                select_leaning=select_leaning,
                apply_weights=apply_survey_weights)
            bstr_arr.append(np.array(bstr_aggr_df["wordvec"].tolist()))
    elif data_source == "tweet":
        for i in range(bootstrap_runs):
            if i%20 == 0: print("progress:", i/bootstrap_runs)
            bstr_output_df = output_df.sample(frac=sample_frac)
            bstr_aggr_df = aggregate_tweet_wordvec(
                output_df=bstr_output_df,
                aggr_unit=aggr_unit,
                dictionary=dictionary,
                force_time_window=force_time_window)
            bstr_arr.append(np.array(bstr_aggr_df["wordvec"].tolist()))
    else:
        print("Please enter a valid data source! (headline, survey, tweet)")
    
    bstr_arr = np.array(bstr_arr)
    if len(force_time_window) > 0:
        assert bstr_arr.shape == (bootstrap_runs, len(force_time_window), dictionary.n_words), "Wrong output array shape!"
        print(bstr_arr.shape)
    return bstr_arr