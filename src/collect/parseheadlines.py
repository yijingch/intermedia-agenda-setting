"""Parse headlines from snapshots and save as a tsv"""

import pandas as pd 
import numpy as np 
import glob 
import traceback
import os 

import yaml

with open("../src/configs.yml", "r") as configs:
    configs = yaml.safe_load(configs)

YEAR = 2020
INPUTPATH = configs["DATAPATH"] + f"headlines/filtered_links_ALL{YEAR}/"
OUTPUTPATH = configs["ROOTPATH"] + "data/"


def parse_headlines(fpath:str) -> pd.DataFrame:
    """Parse headlines from a given folder into a pd.DataFrame"""
    files = glob.glob(fpath + "*.tsv") 
    print("loading from:", fpath)
    print("\t# of domains:", len(files))

    df = pd.DataFrame()
    for i,fp in enumerate(files):
        if i%500 == 0: print("porgress:", i/len(files))
        try:
            this_df = pd.read_csv(fp, sep="\t")
        except Exception:
            print(traceback.format_exc())
            continue 
        if len(this_df) > 0:
            df = pd.concat([this_df, df])
    return df 


def basic_clean_and_split(df:pd.DataFrame, year:int) -> pd.DataFrame:
    """Convert timestamp, strip texts, and indicate candidate in a single column"""
    if year == 2016:
        cand1 = "trump"
        cand2 = "clinton"
    elif year == 2020:
        cand1 = "biden"
        cand2 = "trump"
    else:
        print("Please enter a valid year value! (int: 2016 or 2020)")

    df["date"] = df["timestamp"].map(lambda x: str(x)[:4]+"-"+str(x)[4:6]+"-"+str(x)[6:8])
    df["date"] = pd.to_datetime(df["date"])

    df = df.rename(columns={"headline":"textbody"})
    df["textbody"] = df["textbody"].map(lambda x: str(x).strip())

    cols = ["date", "textbody", "domain", "path"]
    df1 = df[df[f"if_{cand1}"]==1].copy()[cols]
    df2 = df[df[f"if_{cand2}"]==1].copy()[cols]
    df1["candidate"] = cand1 
    df2["candidate"] = cand2

    df_out = pd.concat([df1, df2]).reset_index().drop(columns="index")
    return df_out
    
def main():
    df = parse_headlines(INPUTPATH)
    df_out = basic_clean_and_split(df, year=YEAR)
    df_out.to_csv(OUTPUTPATH + f"headline/headlines_{YEAR}_all.tsv", sep="\t", index=False)

if __name__ == "__main__":
    main()

# cd src
# python collect/parseheadlines.py


