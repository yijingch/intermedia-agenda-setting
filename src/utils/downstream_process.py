"""Functions for downstream analysis"""

import numpy as np 
import pandas as pd
import re
from typing import List, Dict, Any
from src.utils.dict_loader import TopicDictionary


def merge_topics_from_arr(old_arr:np.array, merge_to:str, to_merge:str, dictionary:TopicDictionary):
    """Merge two topics in the output.
    Example:
    merge_topics_from_arr(a, merge_to="government_ops", to_merge="election_campaign", dictionary=dictionary)
    this collapse election_campign into government_ops
    """
    merge_to_idx = dictionary.topic2index[merge_to]
    to_merge_idx = dictionary.topic2index[to_merge]
    new_arr = old_arr.copy()
    new_arr[to_merge_idx] = 0
    new_arr[merge_to_idx] += old_arr[to_merge_idx]
    return new_arr


def collapse_general_controversies(old_arr:np.array, cand:str, dictionary:TopicDictionary):
    """Collapse general_controversies into a specific candidate_controversies
    Example:
    collapse_general_controversies(a, "trump", dictionary)
    this collapse general_controversies into trump_controversies for a trump-related text
    """
    general_contro_idx = dictionary.topic2index["general_controversies"]
    new_arr = old_arr.copy()
    if old_arr[general_contro_idx] > 0:
        new_arr[general_contro_idx] = 0
        new_contro_idx = dictionary.topic2index[f"{cand}_controversies"]
        new_arr[new_contro_idx] += old_arr[general_contro_idx] 
    return new_arr 


def get_majority(arr):
    """Get the majority vote for a (topic) vector
    """
    out_arr = np.zeros(len(arr))
    non_zero_idx = np.argmax(arr)
    out_arr[non_zero_idx] = 1
    return out_arr


def clean_domain_url(domain_url):
    if str(domain_url) != "nan":
        if domain_url[:4] == "www.":
            domain_url = re.sub("www.", "", domain_url)
    else:
        return domain_url
    return domain_url


def trim_period(df, start, end):
    df["date"] = pd.to_datetime(df["date"])
    df = df[(df["date"]>=start)&(df["date"]<=end)]
    df = df.reset_index().drop(columns="index")
    return df


def normalize(arr, smooth=0):
    if np.sum(arr) > 0:
        return (np.array(arr)+smooth)/np.sum(np.array(arr)+smooth)
    else:
        return np.zeros(len(arr))
    

def assign_popularity_weight(domain:str, popularity_dict:Dict):
    label = 0
    if domain in popularity_dict.keys():
        label = popularity_dict[domain]
    return label