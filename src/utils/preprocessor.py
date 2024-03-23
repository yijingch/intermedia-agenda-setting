from datetime import datetime, timedelta, timezone
from typing import Any, List, Set, Type, Tuple
from nltk.corpus import stopwords
import re
# import pickle
# import numpy as np

import nltk
nltk.download("omw-1.4")
nltk.download("wordnet", quiet=True)
nltk.download("punkt")

from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import wordnet
# from src.utils.reading_writing import *


TOKENIZER = RegexpTokenizer(r"\w+")
LEMMATIZER = WordNetLemmatizer()
EN_ST = SnowballStemmer(language="english")
STOPWORDS = list(set(stopwords.words("english")))

def clean_texts(text: str) -> str:
    text = re.sub(r"[^\x00-\x7f]","", str(text)) # clean non-english
    stopwords_pattern = re.compile(r"\b(" + r"|".join(STOPWORDS) + r")\b\s*")
    text = stopwords_pattern.sub("", str(text))
    text = text.translate(str.maketrans("","",string.punctuation))
    return text

def clean_at(text: str) -> str:
    return re.sub(r"r?t?\s?@\S+", "", text)

def clean_url(text: str) -> str:
    return re.sub(r"http\S+", "", text)

def utc2datetime(timestamp):
#     dt = datetime.utcfromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M")
    dt = datetime.utcfromtimestamp(timestamp)
    # https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    return dt

def datetime2utc(dt):
    timestamp = round(int(float(dt.timestamp())))
    return timestamp

def get_tokens(input_string: str) -> list:
    return TOKENIZER.tokenize(input_string)

def get_lemmas(tokens: list) -> str:
    words_tags = nltk.pos_tag(tokens) 
    lemmas = []
    for word, tag in words_tags:
        if tag.startswith("J"):
            lemmas.append(LEMMATIZER.lemmatize(word, wordnet.ADJ))
        elif tag.startswith("V"):
            lemmas.append(LEMMATIZER.lemmatize(word, wordnet.VERB))
        elif tag.startswith("N"):
            lemmas.append(LEMMATIZER.lemmatize(word, wordnet.NOUN))
        elif tag.startswith("R"):
            lemmas.append(LEMMATIZER.lemmatize(word, wordnet.ADV))
        else:
            lemmas.append(word)
    return lemmas
    
# def get_lemmas(tokens: list) -> list:
#     if len(tokens) > 0:
#         lemmas = inflections.lemmatize(tokens)
#     else:
#         lemmas = []
#     return lemmas

def get_stems(tokens: list) -> list:
    if len(tokens) > 0:
        stems = [EN_ST.stem(x) for x in tokens]
    else:
        stems = []
    return stems

def get_inflections_stem(tokens: list):
    if len(tokens) > 0:
        stems = [inflections.stem(x) for x in tokens]
    else:
        stems = []
    return stems

def remove_emojis(data):
    emoj = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
        u"\U00002500-\U00002BEF"  # chinese char
        u"\U00002702-\U000027B0"
        u"\U00002702-\U000027B0"
        u"\U000024C2-\U0001F251"
        u"\U0001f926-\U0001f937"
        u"\U00010000-\U0010ffff"
        u"\u2640-\u2642" 
        u"\u2600-\u2B55"
        u"\u200d"
        u"\u23cf"
        u"\u23e9"
        u"\u231a"
        u"\ufe0f"  # dingbats
        u"\u3030"
                      "]+", re.UNICODE)
    return re.sub(emoj, "", data)
    

def remove_urls(data):
    return re.sub(r"http\S+", "", data)

def remove_linebreaks(data):
    return re.sub("\n", "", data)
