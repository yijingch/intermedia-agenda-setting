import pandas as pd 
import numpy as np
from datetime import datetime
from src.utils.downstream_process import trim_period
from typing import List, Dict, Any

# HEADLINE_FOLDER = "headline-filter0.5-nopopw-normsnap-wormn"
HEADLINE_FOLDER = "headline-filter0.5-nopopw-normsnap"
# HEADLINE_FOLDER = "headline-filter0.5-unweighted"
# SURVEY_FOLDER = "survey-weighted"

def normalize(arr, smooth=0):
    if np.sum(arr) == 0:
        return arr 
    else:
        return (np.array(arr)+smooth)/np.sum(np.array(arr)+smooth)

def load_topvecs(year, data_source, topvec_fpath, data_type="", normalize_by_unit=True, trim:List=[]):
    if year == 2016:
        cand1 = "trump"
        cand2 = "clinton"
    else:
        cand1 = "biden"
        cand2 = "trump"
    if len(data_type) > 0:
        topvec1 = pd.read_pickle(f"{topvec_fpath}{data_source}/{cand1}{year}_topvecs_{data_type}.pkl")
        topvec2 = pd.read_pickle(f"{topvec_fpath}{data_source}/{cand2}{year}_topvecs_{data_type}.pkl")
    else:
        topvec1 = pd.read_pickle(f"{topvec_fpath}{data_source}/{cand1}{year}_topvecs.pkl")
        topvec2 = pd.read_pickle(f"{topvec_fpath}{data_source}/{cand2}{year}_topvecs.pkl")
    if not isinstance(topvec1["date"].tolist()[0], datetime):
        topvec1["date"] = pd.to_datetime(topvec1["date"])
        topvec2["date"] = pd.to_datetime(topvec2["date"])
    if len(trim) > 0:
        topvec1 = trim_period(topvec1, start=trim[0], end=trim[1])
        topvec2 = trim_period(topvec2, start=trim[0], end=trim[1])
    if normalize_by_unit:
        topvec1["majority_topvec"] = topvec1["majority_topvec"].map(lambda x: normalize(x))
        topvec2["majority_topvec"] = topvec2["majority_topvec"].map(lambda x: normalize(x))
    return topvec1, topvec2 


def load_wordvecs(year, data_source, wordvec_fpath, data_type="", normalize_by_unit=False):
    if year == 2016:
        cand1 = "trump"
        cand2 = "clinton"
        # start = START2016
        # end = END2016
    else:
        cand1 = "biden"
        cand2 = "trump"
        # start = START2020
        # end = END2020
    if len(data_type) > 0:
        wordvec1 = pd.read_pickle(f"{wordvec_fpath}{data_source}/{cand1}{year}_wordvecs_{data_type}.pkl")
        wordvec2 = pd.read_pickle(f"{wordvec_fpath}{data_source}/{cand2}{year}_wordvecs_{data_type}.pkl")
    else:
        wordvec1 = pd.read_pickle(f"{wordvec_fpath}{data_source}/{cand1}{year}_wordvecs.pkl")
        wordvec2 = pd.read_pickle(f"{wordvec_fpath}{data_source}/{cand2}{year}_wordvecs.pkl")
    if not isinstance(wordvec1["date"].tolist()[0], datetime):
        wordvec1["date"] = pd.to_datetime(wordvec1["date"])
        wordvec2["date"] = pd.to_datetime(wordvec2["date"])
    # wordvec1 = trim_period(wordvec1, start=start, end=end)
    # wordvec2 = trim_period(wordvec2, start=start, end=end)
    if normalize_by_unit:
        wordvec1["wordvec"] = wordvec1["wordvec"].map(lambda x: normalize(x))
        wordvec2["wordvec"] = wordvec2["wordvec"].map(lambda x: normalize(x))
    return wordvec1, wordvec2 


def load_bstr_arrs(year, data_source, topvec_fpath, data_type="", normalize_by_unit=True, vec_type="topvecs"):
    if year == 2016:
        cand1 = "trump"
        cand2 = "clinton"
    else:
        cand1 = "biden"
        cand2 = "trump"
    if len(data_type) > 0:
        vec_arr1 = np.load(f"{topvec_fpath}{data_source}/bootstrap/{cand1}{year}_bstr_{vec_type}_{data_type}.npy")
        vec_arr2 = np.load(f"{topvec_fpath}{data_source}/bootstrap/{cand2}{year}_bstr_{vec_type}_{data_type}.npy")
    else:
        vec_arr1 = np.load(f"{topvec_fpath}{data_source}/bootstrap/{cand1}{year}_bstr_{vec_type}.npy")
        vec_arr2 = np.load(f"{topvec_fpath}{data_source}/bootstrap/{cand2}{year}_bstr_{vec_type}.npy")
    if normalize_by_unit:
        vec_arr1 = np.apply_along_axis(func1d=normalize, axis=2, arr=vec_arr1)
        vec_arr2 = np.apply_along_axis(func1d=normalize, axis=2, arr=vec_arr2)
    return vec_arr1, vec_arr2


def load_all_topvecs(year:int, topvec_fpath:str, normalize_by_unit:bool=False, trim:List=[]):

    headline_topvec1, headline_topvec2 = load_topvecs(year=year, topvec_fpath=topvec_fpath, data_source=HEADLINE_FOLDER, normalize_by_unit=normalize_by_unit, trim=trim)
    lowc_topvec1, lowc_topvec2 = load_topvecs(year=year, topvec_fpath=topvec_fpath, data_source=HEADLINE_FOLDER, normalize_by_unit=normalize_by_unit, data_type="lowc", trim=trim)
    trad_topvec1, trad_topvec2 = load_topvecs(year=year, topvec_fpath=topvec_fpath, data_source=HEADLINE_FOLDER, normalize_by_unit=normalize_by_unit, data_type="trad", trim=trim)
    left_topvec1, left_topvec2 = load_topvecs(year=year, topvec_fpath=topvec_fpath, data_source=HEADLINE_FOLDER, normalize_by_unit=normalize_by_unit, data_type="left", trim=trim)
    right_topvec1, right_topvec2 = load_topvecs(year=year, topvec_fpath=topvec_fpath, data_source=HEADLINE_FOLDER, normalize_by_unit=normalize_by_unit, data_type="right", trim=trim)
    center_topvec1, center_topvec2 = load_topvecs(year=year, topvec_fpath=topvec_fpath, data_source=HEADLINE_FOLDER, normalize_by_unit=normalize_by_unit, data_type="center", trim=trim)

    # survey_topvec1, survey_topvec2 = load_topvecs(year=year, topvec_fpath=topvec_fpath, data_source=SURVEY_FOLDER, normalize_by_unit=normalize_by_unit, trim=trim)

    # tweet_cand_topvec1, tweet_cand_topvec2 = load_topvecs(year=year, topvec_fpath=topvec_fpath, data_source="tweet-cand", normalize_by_unit=normalize_by_unit, trim=trim)
    # tweet_pub_topvec1, tweet_pub_topvec2 = load_topvecs(year=year, topvec_fpath=topvec_fpath, data_source="tweet-pub-sample", normalize_by_unit=normalize_by_unit, trim=trim)
    # tweet_pub_topvec1, tweet_pub_topvec2 = load_topvecs(year=year, topvec_fpath=topvec_fpath, data_source="tweet-pub") # uncomment when the gt output is ready

    # dem_survey_topvec1, dem_survey_topvec2 = load_topvecs(year=year, topvec_fpath=topvec_fpath, data_source=SURVEY_FOLDER, normalize_by_unit=normalize_by_unit, data_type="dem", trim=trim)
    # rep_survey_topvec1, rep_survey_topvec2 = load_topvecs(year=year, topvec_fpath=topvec_fpath, data_source=SURVEY_FOLDER, normalize_by_unit=normalize_by_unit, data_type="rep", trim=trim)
    # nei_survey_topvec1, nei_survey_topvec2 = load_topvecs(year=year, topvec_fpath=topvec_fpath, data_source=SURVEY_FOLDER, normalize_by_unit=normalize_by_unit, data_type="neither", trim=trim)

    topvec_dfs = {
        "headline":[
            [headline_topvec1, lowc_topvec1, trad_topvec1, left_topvec1, center_topvec1, right_topvec1],
            [headline_topvec2, lowc_topvec2, trad_topvec2, left_topvec2, center_topvec2, right_topvec2],
        ],
        # "tweet":[
        #     [tweet_pub_topvec1, tweet_cand_topvec1],
        #     [tweet_pub_topvec2, tweet_cand_topvec2],
        # ],
        # "survey":[
        #     [survey_topvec1, dem_survey_topvec1, nei_survey_topvec1, rep_survey_topvec1],
        #     [survey_topvec2, dem_survey_topvec2, nei_survey_topvec2, rep_survey_topvec2],
        # ]
    }
    return topvec_dfs


def load_all_wordvecs(year:int, wordvec_fpath:str):

    headline_wordvec1, headline_wordvec2 = load_wordvecs(year=year, wordvec_fpath=wordvec_fpath, data_source=HEADLINE_FOLDER)
    lowc_wordvec1, lowc_wordvec2 = load_wordvecs(year=year, wordvec_fpath=wordvec_fpath, data_source=HEADLINE_FOLDER, data_type="lowc")
    trad_wordvec1, trad_wordvec2 = load_wordvecs(year=year, wordvec_fpath=wordvec_fpath, data_source=HEADLINE_FOLDER, data_type="trad")
    left_wordvec1, left_wordvec2 = load_wordvecs(year=year, wordvec_fpath=wordvec_fpath, data_source=HEADLINE_FOLDER, data_type="left")
    right_wordvec1, right_wordvec2 = load_wordvecs(year=year,wordvec_fpath=wordvec_fpath, data_source=HEADLINE_FOLDER, data_type="right")
    center_wordvec1, center_wordvec2 = load_wordvecs(year=year, wordvec_fpath=wordvec_fpath, data_source=HEADLINE_FOLDER, data_type="center")

    # survey_wordvec1, survey_wordvec2 = load_wordvecs(year=year, wordvec_fpath=wordvec_fpath, data_source=SURVEY_FOLDER)

    # tweet_cand_wordvec1, tweet_cand_wordvec2 = load_wordvecs(year=year, wordvec_fpath=wordvec_fpath, data_source="tweet-cand")
    # tweet_pub_wordvec1, tweet_pub_wordvec2 = load_wordvecs(year=year, wordvec_fpath=wordvec_fpath, data_source="tweet-pub-sample")
    # tweet_pub_topvec1, tweet_pub_topvec2 = load_topvecs(year=year, wordvec_fpath=wordvec_fpath, data_source="tweet-pub") # uncomment when the gt output is ready
    # tweet_pub_wordvec1 = None 
    # tweet_pub_wordvec2 = None

    # dem_survey_wordvec1, dem_survey_wordvec2 = load_wordvecs(year=year, wordvec_fpath=wordvec_fpath, data_source=SURVEY_FOLDER, data_type="dem")
    # rep_survey_wordvec1, rep_survey_wordvec2 = load_wordvecs(year=year, wordvec_fpath=wordvec_fpath, data_source=SURVEY_FOLDER, data_type="rep")
    # nei_survey_wordvec1, nei_survey_wordvec2 = load_wordvecs(year=year, wordvec_fpath=wordvec_fpath, data_source=SURVEY_FOLDER, data_type="neither")

    wordvec_dfs = {
        "headline":[
            [headline_wordvec1, lowc_wordvec1, trad_wordvec1, left_wordvec1, center_wordvec1, right_wordvec1],
            [headline_wordvec2, lowc_wordvec2, trad_wordvec2, left_wordvec2, center_wordvec2, right_wordvec2],
        ],
        # "tweet":[
        #     [tweet_pub_wordvec1, tweet_cand_wordvec1],
        #     [tweet_pub_wordvec2, tweet_cand_wordvec2],
        # ],
        # "survey":[
        #     [survey_wordvec1, dem_survey_wordvec1, nei_survey_wordvec1, rep_survey_wordvec1],
        #     [survey_wordvec2, dem_survey_wordvec2, nei_survey_wordvec2, rep_survey_wordvec2],
        # ]
    }
    return wordvec_dfs


def load_all_bstr_arrs(year:int, vec_fpath:str, vec_type="topvecs", normalize_by_unit:bool=True):
    headline_bstr_arr1, headline_bstr_arr2 = load_bstr_arrs(year=year, topvec_fpath=vec_fpath, data_source=HEADLINE_FOLDER, vec_type=vec_type)
    lowc_bstr_arr1, lowc_bstr_arr2 = load_bstr_arrs(year=year, topvec_fpath=vec_fpath, data_source=HEADLINE_FOLDER, data_type="lowc", vec_type=vec_type, normalize_by_unit=normalize_by_unit)
    trad_bstr_arr1, trad_bstr_arr2 = load_bstr_arrs(year=year, topvec_fpath=vec_fpath, data_source=HEADLINE_FOLDER, data_type="trad", vec_type=vec_type, normalize_by_unit=normalize_by_unit)
    left_bstr_arr1, left_bstr_arr2 = load_bstr_arrs(year=year, topvec_fpath=vec_fpath, data_source=HEADLINE_FOLDER, data_type="left", vec_type=vec_type, normalize_by_unit=normalize_by_unit)
    right_bstr_arr1, right_bstr_arr2 = load_bstr_arrs(year=year, topvec_fpath=vec_fpath, data_source=HEADLINE_FOLDER, data_type="right", vec_type=vec_type, normalize_by_unit=normalize_by_unit)
    center_bstr_arr1, center_bstr_arr2 = load_bstr_arrs(year=year, topvec_fpath=vec_fpath, data_source=HEADLINE_FOLDER, data_type="center", vec_type=vec_type, normalize_by_unit=normalize_by_unit)

    # survey_bstr_arr1, survey_bstr_arr2 = load_bstr_arrs(year=year, topvec_fpath=vec_fpath, data_source=SURVEY_FOLDER, vec_type=vec_type, normalize_by_unit=normalize_by_unit)

    # tweet_cand_bstr_arr1, tweet_cand_bstr_arr2 = load_bstr_arrs(year=year, topvec_fpath=vec_fpath, data_source="tweet-cand", vec_type=vec_type, normalize_by_unit=normalize_by_unit)
    # tweet_pub_bstr_arr1, tweet_pub_bstr_arr2 = load_bstr_arrs(year=year, topvec_fpath=vec_fpath, data_source="tweet-pub-sample", vec_type=vec_type, normalize_by_unit=normalize_by_unit)
    # tweet_pub_bstr_arr1 = None 
    # tweet_pub_bstr_arr2 = None
    # dem_survey_bstr_arr1, dem_survey_bstr_arr2 = load_bstr_arrs(year=year, topvec_fpath=vec_fpath, data_source=SURVEY_FOLDER, data_type="dem", vec_type=vec_type, normalize_by_unit=normalize_by_unit)
    # rep_survey_bstr_arr1, rep_survey_bstr_arr2 = load_bstr_arrs(year=year, topvec_fpath=vec_fpath, data_source=SURVEY_FOLDER, data_type="rep", vec_type=vec_type, normalize_by_unit=normalize_by_unit)
    # nei_survey_bstr_arr1, nei_survey_bstr_arr2 = load_bstr_arrs(year=year, topvec_fpath=vec_fpath, data_source=SURVEY_FOLDER, data_type="neither", vec_type=vec_type, normalize_by_unit=normalize_by_unit)

    bstr_arrs = {
        "headline":[
            [headline_bstr_arr1, lowc_bstr_arr1, trad_bstr_arr1, left_bstr_arr1, center_bstr_arr1, right_bstr_arr1],
            [headline_bstr_arr2, lowc_bstr_arr2, trad_bstr_arr2, left_bstr_arr2, center_bstr_arr2, right_bstr_arr2],
        ],
        # "tweet":[
            # [tweet_pub_bstr_arr1, tweet_cand_bstr_arr1],
            # [tweet_pub_bstr_arr2, tweet_cand_bstr_arr2],
        # ],
        # "survey":[
            # [survey_bstr_arr1, dem_survey_bstr_arr1, nei_survey_bstr_arr1, rep_survey_bstr_arr1],
            # [survey_bstr_arr2, dem_survey_bstr_arr2, nei_survey_bstr_arr2, rep_survey_bstr_arr2],
        # ]
    }
    return bstr_arrs



# ******************************
# ****** LOAD SUM VECTORS ******
# ******************************

def load_sum_vectors(
        year, data_source, vector_fpath, vector_type:str, 
        data_type="", normalize_by_unit:bool=False, load_bstr:bool=False) -> Any:
    if year == 2016:
        cand1 = "trump"
        cand2 = "clinton"
    else:
        cand1 = "biden"
        cand2 = "trump"
    
    if len(data_type) > 0:
        data_type = "_" + data_type 

    if load_bstr:
        sumvec1 = np.load(f"{vector_fpath}{data_source}/bootstrap/{cand1}{year}_bstr_SUM_{vector_type}{data_type}.npy")
        sumvec2 = np.load(f"{vector_fpath}{data_source}/bootstrap/{cand2}{year}_bstr_SUM_{vector_type}{data_type}.npy")
        if normalize_by_unit:
            sumvec1 = np.apply_along_axis(func1d=normalize, axis=1, arr=sumvec1)
            sumvec2 = np.apply_along_axis(func1d=normalize, axis=1, arr=sumvec2)
    else:
        sumvec1 = np.load(f"{vector_fpath}{data_source}/{cand1}{year}_SUM_{vector_type}{data_type}.npy")
        sumvec2 = np.load(f"{vector_fpath}{data_source}/{cand2}{year}_SUM_{vector_type}{data_type}.npy")
        if normalize_by_unit:
            sumvec1 = normalize(sumvec1)
            sumvec2 = normalize(sumvec2)
    return sumvec1, sumvec2


def load_all_sum_vectors(
        year, sumvec_fpath:str, vector_type:str, 
        normalize_by_unit:bool=False, load_bstr:bool=False) -> Any:
    
    headline_sumvec1, headline_sumvec2 = load_sum_vectors(
        year=year, data_source=HEADLINE_FOLDER, vector_fpath=sumvec_fpath, vector_type=vector_type, normalize_by_unit=normalize_by_unit, load_bstr=load_bstr)
    lowc_sumvec1, lowc_sumvec2 = load_sum_vectors(
        year=year, data_source=HEADLINE_FOLDER, vector_fpath=sumvec_fpath, vector_type=vector_type, normalize_by_unit=normalize_by_unit, load_bstr=load_bstr, data_type="lowc")
    trad_sumvec1, trad_sumvec2 = load_sum_vectors(
        year=year, data_source=HEADLINE_FOLDER, vector_fpath=sumvec_fpath, vector_type=vector_type, normalize_by_unit=normalize_by_unit, load_bstr=load_bstr, data_type="trad")
    left_sumvec1, left_sumvec2 = load_sum_vectors(
        year=year, data_source=HEADLINE_FOLDER, vector_fpath=sumvec_fpath, vector_type=vector_type, normalize_by_unit=normalize_by_unit, load_bstr=load_bstr, data_type="left")
    right_sumvec1, right_sumvec2 = load_sum_vectors(
        year=year, data_source=HEADLINE_FOLDER, vector_fpath=sumvec_fpath, vector_type=vector_type, normalize_by_unit=normalize_by_unit, load_bstr=load_bstr, data_type="right")
    center_sumvec1, center_sumvec2 = load_sum_vectors(
        year=year, data_source=HEADLINE_FOLDER, vector_fpath=sumvec_fpath, vector_type=vector_type, normalize_by_unit=normalize_by_unit, load_bstr=load_bstr, data_type="center")
    
    # survey_sumvec1, survey_sumvec2 = load_sum_vectors(
    #     year=year, data_source=SURVEY_FOLDER, vector_fpath=sumvec_fpath, vector_type=vector_type,
    #     normalize_by_unit=normalize_by_unit, load_bstr=load_bstr)
    # dem_survey_sumvec1, dem_survey_sumvec2 = load_sum_vectors(
    #     year=year, data_source=SURVEY_FOLDER, vector_fpath=sumvec_fpath, vector_type=vector_type,
    #     normalize_by_unit=normalize_by_unit, load_bstr=load_bstr, data_type="dem")
    # rep_survey_sumvec1, rep_survey_sumvec2 = load_sum_vectors(
    #     year=year, data_source=SURVEY_FOLDER, vector_fpath=sumvec_fpath, vector_type=vector_type,
    #     normalize_by_unit=normalize_by_unit, load_bstr=load_bstr, data_type="rep")
    # nei_survey_sumvec1, nei_survey_sumvec2 = load_sum_vectors(
    #     year=year, data_source=SURVEY_FOLDER, vector_fpath=sumvec_fpath, vector_type=vector_type,
    #     normalize_by_unit=normalize_by_unit, load_bstr=load_bstr, data_type="neither")
    
    # when I have twitter data
    # tweet_cand_sumvec1, tweet_cand_sumvec2 = load_sum_vectors(
        # year=year, data_source="tweet-cand", vector_fpath=sumvec_fpath, vector_type=vector_type,
        # normalize_by_unit=normalize_by_unit, load_bstr=load_bstr)
    # tweet_pub_sumvec1, tweet_pub_sumvec2 = load_sum_vectors(
    #     year=year, data_source="tweet-pub-sample", vector_fpath=sumvec_fpath, vector_type=vector_type,
    #     normalize_by_unit=normalize_by_unit, load_bstr=load_bstr)
    tweet_pub_sumvec1 = None 
    tweet_pub_sumvec2 = None

    sumvecs_all = {
        "headline":[
            [headline_sumvec1, lowc_sumvec1, trad_sumvec1, left_sumvec1, center_sumvec1, right_sumvec1],
            [headline_sumvec2, lowc_sumvec2, trad_sumvec2, left_sumvec2, center_sumvec2, right_sumvec2],
        ],
        # "tweet":[
        #     [tweet_pub_sumvec1, tweet_cand_sumvec1],
        #     [tweet_pub_sumvec2, tweet_cand_sumvec2],
        # ],
        # "survey":[
        #     [survey_sumvec1, dem_survey_sumvec1, nei_survey_sumvec1, rep_survey_sumvec1],
        #     [survey_sumvec2, dem_survey_sumvec2, nei_survey_sumvec2, rep_survey_sumvec2],
        # ]
    }
    return sumvecs_all