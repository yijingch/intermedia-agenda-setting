import pandas as pd 
import numpy as np
import pickle
from collections import Counter
from typing import List, Dict, Any
from src.utils.preprocessor import clean_at, clean_url, clean_texts, get_tokens, get_lemmas, get_stems
from src.utils.downstream_process import trim_period
from src.utils.text import contractions


class Headlines():
    def __init__(
            self, 
            folderpath:str = "", 
            year:int = 2020,
            sample_size:int = -1, 
            sample_frac:float = -1.0, 
            drop_duplicates:bool = True, 
            texts_to_excl: List = []) -> None:
        
        if len(folderpath) > 0:
            df = pd.read_csv(folderpath + f"headline/headlines_{year}_all.tsv", sep="\t")
            self.year = year 

            if year == 2016:
                if drop_duplicates:
                    self.df_cand1 = df[df["candidate"]=="trump"].drop_duplicates(subset="textbody").reset_index().drop(columns="index")
                    self.df_cand2 = df[df["candidate"]=="clinton"].drop_duplicates(subset="textbody").reset_index().drop(columns="index")
                else:
                    self.df_cand1 = df[df["candidate"]=="trump"].reset_index().drop(columns="index")
                    self.df_cand2 = df[df["candidate"]=="clinton"].reset_index().drop(columns="index")
                self.df_cand1_label = "trump2016"
                self.df_cand2_label = "clinton2016"
            elif year == 2020:
                if drop_duplicates:
                    self.df_cand1 = df[df["candidate"]=="biden"].drop_duplicates(subset="textbody").reset_index().drop(columns="index")
                    self.df_cand2 = df[df["candidate"]=="trump"].drop_duplicates(subset="textbody").reset_index().drop(columns="index")
                else:
                    self.df_cand1 = df[df["candidate"]=="biden"].reset_index().drop(columns="index")
                    self.df_cand2 = df[df["candidate"]=="trump"].reset_index().drop(columns="index")                 
                self.df_cand1_label = "biden2020"
                self.df_cand2_label = "trump2020"
            else:
                print("Please enter a valid year integer!")

            # if loading samples
            if sample_size > 0:  # equal number of samples for both candidates
                if len(texts_to_excl) > 0:
                    print(f"Sampling {sample_size} texts from the unsampled headlines...")
                    self.df_cand1 = self.df_cand1[self.df_cand1["textbody"].notin(texts_to_excl)].sample(n=sample_size)
                    self.df_cand2 = self.df_cand2[self.df_cand2["textbody"].notin(texts_to_excl)].sample(n=sample_size)
                else:
                    print(f"Sampling {sample_size} texts from all headlines...")
                    self.df_cand1 = self.df_cand1.sample(n=sample_size)
                    self.df_cand2 = self.df_cand2.sample(n=sample_size)
            if sample_frac > 0: 
                if len(texts_to_excl) > 0:
                    print(f"Sampling {sample_frac*100}% of the unsampled headlines...")
                    self.df_cand1 = self.df_cand1[self.df_cand1["textbody"].notin(texts_to_excl)].sample(frac=sample_frac)
                    self.df_cand2 = self.df_cand2[self.df_cand2["textbody"].notin(texts_to_excl)].sample(frac=sample_frac)
                else:
                    print(f"Sampling {sample_frac*100}% of all headlines...")
                    self.df_cand1 = self.df_cand1.sample(frac=sample_frac)
                    self.df_cand2 = self.df_cand2.sample(frac=sample_frac)
        else:
            print("Empty file path, maybe load data from a saved pickle file...")
            pass

    def trim(self, start:str, end:str) -> None:
        self.df_cand1 = trim_period(self.df_cand1, start=start, end=end)
        self.df_cand2 = trim_period(self.df_cand2, start=start, end=end)
            
    def clean(self, lemmatize=True, stemming=False) -> None:
        print("cleaning...")
        self.df_cand1["cleaned_textbody"] = self.df_cand1["textbody"].map(lambda x: str(x).lower())
        self.df_cand2["cleaned_textbody"] = self.df_cand2["textbody"].map(lambda x: str(x).lower())
        self.df_cand1["cleaned_textbody"] = self.df_cand1["cleaned_textbody"].map(lambda x: contractions.expand(x, drop_ownership=True))
        self.df_cand1["cleaned_textbody"] = self.df_cand1["cleaned_textbody"].map(lambda x: contractions.expand(x, drop_ownership=True))
        self.df_cand1["cleaned_textbody"] = self.df_cand1["cleaned_textbody"].map(lambda x: clean_url(x))
        self.df_cand2["cleaned_textbody"] = self.df_cand2["cleaned_textbody"].map(lambda x: clean_url(x))
        self.df_cand1["tokens"] = self.df_cand1["cleaned_textbody"].map(lambda x: get_tokens(x))
        self.df_cand2["tokens"] = self.df_cand2["cleaned_textbody"].map(lambda x: get_tokens(x))
        if lemmatize:
            self.df_cand1["lemmas"] = self.df_cand1["tokens"].map(lambda x: get_lemmas(x))
            self.df_cand2["lemmas"] = self.df_cand2["tokens"].map(lambda x: get_lemmas(x))
            self.df_cand1["cleaned_textbody"] = self.df_cand1["lemmas"].map(lambda x: " ".join(x))
            self.df_cand2["cleaned_textbody"] = self.df_cand2["lemmas"].map(lambda x: " ".join(x))
            self.df_cand1["cleaned_textbody"] = self.df_cand1["cleaned_textbody"].map(lambda x: str(x).replace("president trump", "trump"))
            self.df_cand2["cleaned_textbody"] = self.df_cand2["cleaned_textbody"].map(lambda x: str(x).replace("president trump", "trump"))
        if stemming:
            self.df_cand1["stems"] = self.df_cand1["tokens"].map(lambda x: get_stems(x))
            self.df_cand2["stems"] = self.df_cand2["tokens"].map(lambda x: get_stems(x))
            self.df_cand1["cleaned_textbody"] = self.df_cand1["stems"].map(lambda x: " ".join(x))
            self.df_cand2["cleaned_textbody"] = self.df_cand2["stems"].map(lambda x: " ".join(x))
            self.df_cand1["cleaned_textbody"] = self.df_cand1["cleaned_textbody"].map(lambda x: str(x).replace("presid trump", "trump"))
            self.df_cand2["cleaned_textbody"] = self.df_cand2["cleaned_textbody"].map(lambda x: str(x).replace("presid trump", "trump"))
        if (lemmatize==False) and (stemming==False):
            self.df_cand1["cleaned_textbody"] = self.df_cand1["tokens"].map(lambda x: " ".join(x))
            self.df_cand2["cleaned_textbody"] = self.df_cand2["tokens"].map(lambda x: " ".join(x))
            self.df_cand1["cleaned_textbody"] = self.df_cand1["cleaned_textbody"].map(lambda x: str(x).replace("president trump", "trump"))
            self.df_cand2["cleaned_textbody"] = self.df_cand2["cleaned_textbody"].map(lambda x: str(x).replace("president trump", "trump"))

    def save(self, fpath) -> None:
        fhandle = open(fpath, "wb")
        pickle.dump(self, fhandle)
        fhandle.close()

    @classmethod
    def load(cls, fpath) -> None:
        fhandle = open(fpath, "rb")
        obj = pickle.load(fhandle)
        fhandle.close()
        return obj
    # thanks to https://stackoverflow.com/questions/52853754/self-unpickling-in-class-method-in-python


class Surveys():
    def __init__(self, fpath1:str = "", fpath2:str = "", sample_size:int = -1, sample_frac:float = -1.0, texts_to_excl:List = []) -> None:
        if len(fpath1) > 0 and len(fpath2) > 0:
            if "2016" in fpath1:
                assert ("trump" in fpath1) and ("clinton" in fpath2), "Please enter the winning candidate's file in fpath1, the other in fpath2."
                self.df_cand1_label = "trump2016"
                self.df_cand2_label = "clinton2016"
            elif "2020" in fpath1:
                assert ("biden" in fpath1) and ("trump" in fpath2), "Please enter the winning candidate's file in fpath1, the other in fpath2."
                self.df_cand1_label = "biden2020"
                self.df_cand2_label = "trump2020"
            else: 
                print("Please enter a valid year (2016 or 2020).")

            self.df_cand1 = pd.read_csv(fpath1)
            self.df_cand2 = pd.read_csv(fpath2)
            self.df_cand1.dropna(subset="textbody", inplace=True)
            self.df_cand2.dropna(subset="textbody", inplace=True)
            self.df_cand1["date"] = pd.to_datetime(self.df_cand1["date"])
            self.df_cand2["date"] = pd.to_datetime(self.df_cand2["date"])

            if sample_size > 0:  # equal number of samples for both candidates
                if len(texts_to_excl) > 0:
                    print(f"Sampling {sample_size} texts from the unsampled survey responses...")
                    self.df_cand1 = self.df_cand1[self.df_cand1["textbody"].notin(texts_to_excl)].sample(n=sample_size)
                    self.df_cand2 = self.df_cand2[self.df_cand2["textbody"].notin(texts_to_excl)].sample(n=sample_size)
                else:
                    print(f"Sampling {sample_size} texts from all survey responses...")
                    self.df_cand1 = self.df_cand1.sample(n=sample_size)
                    self.df_cand2 = self.df_cand2.sample(n=sample_size)
            if sample_frac > 0: 
                if len(texts_to_excl) > 0:
                    print(f"Sampling {sample_frac*100}% of the unsampled survey responses...")
                    self.df_cand1 = self.df_cand1[self.df_cand1["textbody"].notin(texts_to_excl)].sample(frac=sample_frac)
                    self.df_cand2 = self.df_cand2[self.df_cand2["textbody"].notin(texts_to_excl)].sample(frac=sample_frac)
                else:
                    print(f"Sampling {sample_frac*100}% of all survey responses...")
                    self.df_cand1 = self.df_cand1.sample(frac=sample_frac)
                    self.df_cand2 = self.df_cand2.sample(frac=sample_frac)

            self.df_cand1["textbody"] = self.df_cand1["textbody"].map(lambda x: str(x).strip())
            self.df_cand2["textbody"] = self.df_cand2["textbody"].map(lambda x: str(x).strip())

        else:
            print("Empty folder path, maybe load data from a saved pickle file...")
            pass

    def trim(self, start:str, end:str) -> None:
        self.df_cand1 = trim_period(self.df_cand1, start=start, end=end)
        self.df_cand2 = trim_period(self.df_cand2, start=start, end=end)

    def clean(self, lemmatize=True, stemming=False) -> None:
        print("cleaning...")
        self.df_cand1["cleaned_textbody"] = self.df_cand1["textbody"].map(lambda x: str(x).lower())
        self.df_cand2["cleaned_textbody"] = self.df_cand2["textbody"].map(lambda x: str(x).lower())
        self.df_cand1["cleaned_textbody"] = self.df_cand1["cleaned_textbody"].map(lambda x: contractions.expand(x, drop_ownership=True))
        self.df_cand1["cleaned_textbody"] = self.df_cand1["cleaned_textbody"].map(lambda x: contractions.expand(x, drop_ownership=True))
        self.df_cand1["tokens"] = self.df_cand1["cleaned_textbody"].map(lambda x: get_tokens(x))
        self.df_cand2["tokens"] = self.df_cand2["cleaned_textbody"].map(lambda x: get_tokens(x))
        if lemmatize:
            self.df_cand1["lemmas"] = self.df_cand1["tokens"].map(lambda x: get_lemmas(x))
            self.df_cand2["lemmas"] = self.df_cand2["tokens"].map(lambda x: get_lemmas(x))
            self.df_cand1["cleaned_textbody"] = self.df_cand1["lemmas"].map(lambda x: " ".join(x))
            self.df_cand2["cleaned_textbody"] = self.df_cand2["lemmas"].map(lambda x: " ".join(x))
            self.df_cand1["cleaned_textbody"] = self.df_cand1["cleaned_textbody"].map(lambda x: str(x).replace("president trump", "trump"))
            self.df_cand2["cleaned_textbody"] = self.df_cand2["cleaned_textbody"].map(lambda x: str(x).replace("president trump", "trump"))
        if stemming:
            self.df_cand1["stems"] = self.df_cand1["tokens"].map(lambda x: get_stems(x))
            self.df_cand2["stems"] = self.df_cand2["tokens"].map(lambda x: get_stems(x))
            self.df_cand1["cleaned_textbody"] = self.df_cand1["stems"].map(lambda x: " ".join(x))
            self.df_cand2["cleaned_textbody"] = self.df_cand2["stems"].map(lambda x: " ".join(x))
            self.df_cand1["cleaned_textbody"] = self.df_cand1["cleaned_textbody"].map(lambda x: str(x).replace("presid trump", "trump"))
            self.df_cand2["cleaned_textbody"] = self.df_cand2["cleaned_textbody"].map(lambda x: str(x).replace("presid trump", "trump"))
        if (lemmatize==False) and (stemming==False):
            self.df_cand1["cleaned_textbody"] = self.df_cand1["tokens"].map(lambda x: " ".join(x))
            self.df_cand2["cleaned_textbody"] = self.df_cand2["tokens"].map(lambda x: " ".join(x))
            self.df_cand1["cleaned_textbody"] = self.df_cand1["cleaned_textbody"].map(lambda x: str(x).replace("president trump", "trump"))
            self.df_cand2["cleaned_textbody"] = self.df_cand2["cleaned_textbody"].map(lambda x: str(x).replace("president trump", "trump"))

    def save(self, fpath) -> None:
        fhandle = open(fpath, "wb")
        pickle.dump(self, fhandle)
        fhandle.close()
        
    @classmethod
    def load(cls, fpath) -> None:
        fhandle = open(fpath, "rb")
        obj = pickle.load(fhandle)
        fhandle.close()
        return obj
        

class Tweets():
    def __init__(self, fpath1: str = "", fpath2: str = "", sample_size:int = -1, sample_frac: float = -1.0, texts_to_excl:List = []) -> None:
        if len(fpath1) > 0 and len(fpath2) > 0:
            if "2016" in fpath1:
                assert ("trump" in fpath1) and ("clinton" in fpath2), "Please enter the winning candidate's file in fpath1, the other in fpath2."
                self.df_cand1_label = "trump2016"
                self.df_cand2_label = "clinton2016"
            elif "2020" in fpath1:
                assert ("biden" in fpath1) and ("trump" in fpath2), "Please enter the winning candidate's file in fpath1, the other in fpath2."
                self.df_cand1_label = "biden2020"
                self.df_cand2_label = "trump2020"
            else: 
                print("Please enter a valid year (2016 or 2020).")
            
            self.df_cand1 = pd.read_csv(fpath1)
            self.df_cand2 = pd.read_csv(fpath2)

            if sample_size > 0:  # equal number of samples for both candidates
                if len(texts_to_excl) > 0:
                    print(f"Sampling {sample_size} texts from the unsampled tweets...")
                    self.df_cand1 = self.df_cand1[self.df_cand1["textbody"].notin(texts_to_excl)].sample(n=sample_size)
                    self.df_cand2 = self.df_cand2[self.df_cand2["textbody"].notin(texts_to_excl)].sample(n=sample_size)
                else:
                    print(f"Sampling {sample_size} texts from all tweets...")
                    self.df_cand1 = self.df_cand1.sample(n=sample_size)
                    self.df_cand2 = self.df_cand2.sample(n=sample_size)
            if sample_frac > 0: 
                if len(texts_to_excl) > 0:
                    print(f"Sampling {sample_frac*100}% of the unsampled tweets...")
                    self.df_cand1 = self.df_cand1[self.df_cand1["textbody"].notin(texts_to_excl)].sample(frac=sample_frac)
                    self.df_cand2 = self.df_cand2[self.df_cand2["textbody"].notin(texts_to_excl)].sample(frac=sample_frac)
                else:
                    print(f"Sampling {sample_frac*100}% of all tweets...")
                    self.df_cand1 = self.df_cand1.sample(frac=sample_frac)
                    self.df_cand2 = self.df_cand2.sample(frac=sample_frac)
            
            self.df_cand1.rename(columns={"full_text": "textbody"}, inplace=True)
            self.df_cand2.rename(columns={"full_text": "textbody"}, inplace=True)
            self.df_cand1["date"] = pd.to_datetime(self.df_cand1["date"])
            self.df_cand2["date"] = pd.to_datetime(self.df_cand2["date"])
            self.df_cand1["textbody"] = self.df_cand1["textbody"].map(lambda x: str(x).strip())
            self.df_cand2["textbody"] = self.df_cand2["textbody"].map(lambda x: str(x).strip())
            self.df_cand1.dropna(subset="textbody", inplace=True)
            self.df_cand2.dropna(subset="textbody", inplace=True)
        else:
            print("Empty folder path, maybe load data from a saved pickle file...")
            pass

    def trim(self, start:str, end:str) -> None:
        self.df_cand1 = trim_period(self.df_cand1, start=start, end=end)
        self.df_cand2 = trim_period(self.df_cand2, start=start, end=end)

    def clean(self, lemmatize=True, stemming=False) -> None:
        print("cleaning...")
        self.df_cand1["cleaned_textbody"] = self.df_cand1["textbody"].map(lambda x: str(x).lower())
        self.df_cand2["cleaned_textbody"] = self.df_cand2["textbody"].map(lambda x: str(x).lower())
        self.df_cand1["cleaned_textbody"] = self.df_cand1["cleaned_textbody"].map(lambda x: clean_at(x))
        self.df_cand2["cleaned_textbody"] = self.df_cand2["cleaned_textbody"].map(lambda x: clean_at(x))
        self.df_cand1["cleaned_textbody"] = self.df_cand1["cleaned_textbody"].map(lambda x: clean_url(x))
        self.df_cand2["cleaned_textbody"] = self.df_cand2["cleaned_textbody"].map(lambda x: clean_url(x))
        self.df_cand1["cleaned_textbody"] = self.df_cand1["cleaned_textbody"].map(lambda x: contractions.expand(x, drop_ownership=True))
        self.df_cand1["cleaned_textbody"] = self.df_cand1["cleaned_textbody"].map(lambda x: contractions.expand(x, drop_ownership=True))
        self.df_cand1["tokens"] = self.df_cand1["cleaned_textbody"].map(lambda x: get_tokens(x))
        self.df_cand2["tokens"] = self.df_cand2["cleaned_textbody"].map(lambda x: get_tokens(x))
        if lemmatize:
            self.df_cand1["lemmas"] = self.df_cand1["tokens"].map(lambda x: get_lemmas(x))
            self.df_cand2["lemmas"] = self.df_cand2["tokens"].map(lambda x: get_lemmas(x))
            self.df_cand1["cleaned_textbody"] = self.df_cand1["lemmas"].map(lambda x: " ".join(x))
            self.df_cand2["cleaned_textbody"] = self.df_cand2["lemmas"].map(lambda x: " ".join(x))
            self.df_cand1["cleaned_textbody"] = self.df_cand1["cleaned_textbody"].map(lambda x: str(x).replace("president trump", "trump"))
            self.df_cand2["cleaned_textbody"] = self.df_cand2["cleaned_textbody"].map(lambda x: str(x).replace("president trump", "trump"))
        if stemming:
            self.df_cand1["stems"] = self.df_cand1["tokens"].map(lambda x: get_stems(x))
            self.df_cand2["stems"] = self.df_cand2["tokens"].map(lambda x: get_stems(x))
            self.df_cand1["cleaned_textbody"] = self.df_cand1["stems"].map(lambda x: " ".join(x))
            self.df_cand2["cleaned_textbody"] = self.df_cand2["stems"].map(lambda x: " ".join(x))
            self.df_cand1["cleaned_textbody"] = self.df_cand1["cleaned_textbody"].map(lambda x: str(x).replace("presid trump", "trump"))
            self.df_cand2["cleaned_textbody"] = self.df_cand2["cleaned_textbody"].map(lambda x: str(x).replace("presid trump", "trump"))
        if (lemmatize==False) and (stemming==False):
            self.df_cand1["cleaned_textbody"] = self.df_cand1["tokens"].map(lambda x: " ".join(x))
            self.df_cand2["cleaned_textbody"] = self.df_cand2["tokens"].map(lambda x: " ".join(x))
            self.df_cand1["cleaned_textbody"] = self.df_cand1["cleaned_textbody"].map(lambda x: str(x).replace("president trump", "trump"))
            self.df_cand2["cleaned_textbody"] = self.df_cand2["cleaned_textbody"].map(lambda x: str(x).replace("president trump", "trump"))
    
    def save(self, fpath) -> None:
        fhandle = open(fpath, "wb")
        pickle.dump(self, fhandle)
        fhandle.close()

    @classmethod
    def load(cls, fpath) -> None:
        fhandle = open(fpath, "rb")
        obj = pickle.load(fhandle)
        fhandle.close()
        return obj