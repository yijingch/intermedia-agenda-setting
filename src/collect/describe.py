import glob
import os
import sys
import pandas as pd
import traceback

def read_domain_names(fname):
    if fname == "aggre_all_v3.csv":
        input_df = pd.read_csv(fname, sep="\s+")
    else:
        input_df = pd.read_csv(fname)
    return input_df["domain"].tolist()


if __name__ == "__main__":
    INPUT_FOLDER = sys.argv[1]
    DOMAIN_SHEET = "/home/yijingch/index/domain_list_all.csv"
    OUTPUT_FPATH = sys.argv[2]
    domains = read_domain_names(DOMAIN_SHEET)

    # timels = []
    domainls = []
    fullpathls = []

    for i,domain in enumerate(domains):
        if i%1000 == 0:
            print("progress:", i/len(domains))
        try:
            domain_new = "www." + domain.replace("/", "[slash]")
            domain_path = os.path.join(INPUT_FOLDER + "/" + domain_new)
            if not os.path.exists(domain_path):
                continue
            snapshots = glob.glob((INPUT_FOLDER + "/{}/*.snapshot").format(domain_new))
            for ss in snapshots:
                fullpathls.append(ss)
                # time = os.path.basename(ss[:-9])
                # timels.append(time)
                domainls.append(domain)
        except Exception:
            print(traceback.format_exc())
            continue

    df = pd.DataFrame()
    df["domain"] = domainls
    df["fullpath"] = fullpathls
    # df["timestamp"] = timels
    # df["date"] = df["timestamp"].map(lambda x: str(x)[:4]+"-"+str(x)[4:6]+"-"+str(x)[6:8])
    # df["date"] = pd.to_datetime(df["date"])
    # df = df.drop(columns=["timestamp"])
    if os.path.exists(OUTPUT_FPATH):
        df.to_csv(OUTPUT_FPATH, mode="a", header=False, index=False)
    else:
        df.to_csv(OUTPUT_FPATH, index=False)


# python3 describe.py /home/cbudak/website /home/yijingch/output/describe_snapshots_2016.csv -- finished!
# python3 describe.py /home/cbudak/website20160615_20161130 /home/yijingch/output/describe_snapshots_2016.csv -- finished!
# python3 describe.py /home/cbudak/website20200615_20201130 /home/yijingch/output/describe_snapshots_2020.csv -- finished!
# 092422 - finished all