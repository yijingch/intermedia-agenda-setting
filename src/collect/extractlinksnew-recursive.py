import glob
import os
import re
import subprocess
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from multiprocessing import  Pool
import sys
import traceback

text_count_thresh = 3
link_count_thresh = 5

ROOTPATH = os.getcwd()

# ------------ #
# Read Domains #
# ------------ #

def read_domain_names(filename):
    input_df = pd.read_csv(filename)
    return input_df[["domain"]]

# -------------- #
#  Scraper Class #
# -------------- #

class WayBackMachineScraper:

 def extract_links_from_body(outf, domain, snapshot, snapshot_path, ts):
    with open(snapshot_path, "rb") as f: 
        snapshot_body = BeautifulSoup(f, 'html.parser')# 'lxml')
    for link in snapshot_body.find_all('a'):
        try:
            if link.get('href') != None:
                url = link.get('href').replace('\t', ' ').replace('\n', ' ').replace('\r', ' ')
                textinurl = str(link.findAll(text=True))
                outf.write(domain + "\t" + snapshot + "\t" + url + "\t" + textinurl + "\n")
        except Exception as e:
            raise e

# ------------------------------------------------------- Main ------------------------------------------------------- #

 if __name__ == '__main__':

    INPUT_FOLDER = sys.argv[1]  
    DOMAIN_SHEET = "/home/yijingch/index/domain_list_all.csv"
    domain_df = read_domain_names(DOMAIN_SHEET)
    OUTPUT_FOLDER = sys.argv[2] 
    
    print(f"INPUT_FOLDER=={INPUT_FOLDER}")
    print(f"DOMAIN_SHEET=={DOMAIN_SHEET}")
    print(f"OUTPUT_FOLDER=={INPUT_FOLDER}")

    try: os.mkdir(OUTPUT_FOLDER)
    except: pass

    for domain in domain_df["domain"].tolist():
    # for domain in ["aim4truth.org"]:  # for test run locally, e.g., "amherst.edu"
        try:                 
            # new: get the snapshot folder (now that the input folder has restructured a little)
            domain_new = "www." + domain
            domain_path = glob.glob(INPUT_FOLDER + "/*/" + domain_new)
            if len(domain_path) == 0: # no input snapshot
                continue
            else:
                domain_path = domain_path[0]
                
            newfile = OUTPUT_FOLDER + domain_new + '_All-Extracted-Links.tsv'
            if os.path.exists(newfile):
                print("Skipping file", newfile)
                continue
            print("Processing", domain)
            snapshot_paths = glob.glob(domain_path + "/*.snapshot") # modified
            outf = open(newfile, "w")
            for snapshot_path in snapshot_paths:
                timestamp = os.path.basename(snapshot_path[:-9])
                snapshot = snapshot_path.replace(INPUT_FOLDER + domain_new + "/","")
                snapshot = snapshot.replace(".snapshot", "")
                extract_links_from_body(outf, domain, snapshot, snapshot_path, timestamp)
            outf.close()
        except Exception:
            print(traceback.format_exc())
            continue


# cd src
# python3 extractlinksnew-recursive.py /home/cbudak/website20200615_20201130  /home/yijingch/data/extracted_links2020/ 
# notes:
# 031024: rummormillnews.com is taking extremely long (more than 2 days), skipped (temporarily)
# 031424: finished extracting 2020; will start 2016 once the data is ready 
        
        

