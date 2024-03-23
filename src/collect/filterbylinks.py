import glob
import sys
import os
import re
import pandas as pd
import numpy as np
import traceback


# ------------ #
# Read Domains #
# ------------ #

def read_domain_names(filename):
    if filename == "aggre_all_v3.csv":
        input_df = pd.read_csv('aggre_all_v3.csv', sep='\s+')
    else:
        input_df = pd.read_csv(filename)
    return input_df[['domain']]


# ------------ #
# Filter Links #
# ------------ #

def read_and_filter(fpath):
    # print("processing:", fpath)
    df = pd.read_csv(fpath, sep="\\t", quotechar="'", engine="python",
    names=["timestamp", "domain", "path", "headline", "link", "value"], 
    # error_bad_lines=False,  # deprecated in pandas
    on_bad_lines="skip",
    )
    if len(df) > 0:
            df["headline"] = df["headline"].map(lambda x: str(x).lower())
            df["if_trump"] = df["headline"].str.contains(r"\btrump\b|\bdonald\b").astype(int)
            df["if_biden"] = df["headline"].str.contains(r"\bjoe\b|\bbiden\b").astype(int)
            df["if_clinton"] = df["headline"].str.contains(r"\bhillary\b|\bclinton\b").astype(int)
            filtered_df = df[(df["if_trump"]==1)|(df["if_biden"]==1)|(df["if_clinton"]==1)]  # only save relevant links
            output_df = filtered_df[["timestamp", "path", "domain", "headline", "if_trump", "if_biden", "if_clinton"]]
    else:
        output_df = pd.DataFrame()
    return output_df


# ---- Main ---- #

if __name__ == "__main__":

    # all domains
    INPUT_FOLDER = sys.argv[1]
    OUTPUT_FOLDER = sys.argv[2]
    DOMAIN_SHEET = "/home/yijingch/index/domain_list_all.csv"
    domain_df = read_domain_names(DOMAIN_SHEET)
    print("input folder:", INPUT_FOLDER)
    print("output folder:", OUTPUT_FOLDER)

    # save output here:
    try: os.mkdir(OUTPUT_FOLDER)
    except: pass


    # keep links containing keywords
    for domain in domain_df["domain"].tolist():
        try:
            domain_new = "www." + domain
            links_fpath = os.path.join(INPUT_FOLDER, domain_new + "_All-Extracted-Links.tsv")
            if not os.path.exists(links_fpath):
                print("Didn't find:", domain_new)
                continue
            newfile = OUTPUT_FOLDER + domain_new + "_All-Filtered-Links.tsv"
            if os.path.exists(newfile):
                print("Skipping file", newfile)
                continue
            print("Processing:", domain)
            output_df = read_and_filter(links_fpath)
            if len(output_df)>0:
                print("\tSaving output:", domain)
                output_df.to_csv(newfile, sep="\t", index=False)
        except Exception:
            print(traceback.format_exc())
            continue


# python3 filterbylinks.py /home/yijingch/data/extracted_links2020 /home/yijingch/data/filtered_links_ALL2020/ 
        

# notes:
# 031624 started running 2020 8:22pm constantly got killed 