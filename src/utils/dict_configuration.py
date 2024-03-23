from src.utils.dict_loader import TopicDictionary
import yaml

with open("../../src/configs.yml", "r") as conf:
    configs = yaml.safe_load(conf)

ROOTPATH = configs["ROOTPATH"]

dictpath2016 = ROOTPATH + "index/dictionary/gtm_round1_2016_merged_full.tsv"
dictionary2016 = TopicDictionary(
    dictpath=dictpath2016,
    relevance_col="if_reasonable_yijing", 
    lemmatize=False, 
    stemming=True, 
    min_relevance=2,
    post_mturk_change={"remove":[("election_campaign", "clinton")], "add":[]},
    topic_idx_ext = [
        "../../index/dictionary/index2topic_2016.json", 
        "../../index/dictionary/topic2index_2016.json"]
)

dictpath2020 = ROOTPATH + "index/dictionary/gtm_round1_2020_merged_full.tsv"
dictionary2020 = TopicDictionary(
    dictpath=dictpath2020,
    relevance_col="if_reasonable_yijing", 
    lemmatize=False, 
    stemming=True, 
    min_relevance=2,
    post_mturk_change={"remove":[("election_campaign", "clinton")], "add":[]},
    topic_idx_ext = [
        "../../index/dictionary/index2topic_2020.json", 
        "../../index/dictionary/topic2index_2020.json"]
)