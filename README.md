# intermedia-agenda-setting
Cleaned-up project repo for the Intermedia Agenda Setting paper 

Chen, Y., Liu, Y., Singh, L., Budak, C., (2024) Intermedia Agenda Setting during the 2016 and 2020 U.S. Presidential Elections. ICWSM 2024.


## Code Guide


### Collection and Preprocessing

**Remote pipeline on Rabbit server**
- Collect homepage snapshots via Wayback Machine API from a list of domains `/index/domain/domain_list_all.csv`. [script not here]
- Describe the number of snapshots per domain per day (for downstream weighting): `src/collect/desribe.py`
- Extract headlines from domain snapshots: `src/collect/extractlinksnew.py`
- Filter and keep only candidate-related headlines: `src/collect/filterbylinks.py` 

**Local pipeline for preprocessing**
- Parse headlines and save as tsv: `src/collect/parseheadlines.py`
- Filter domains with too few snapshots (date coverage < 50%): `src/preprocess/filter-domains.ipynb`

### Analysis 

**Dictionary-based Topic Modeling**

- Set up configrations (paths, start and end date) in `src/configs.yml`
- Specify the version of dictionary to use in `src/utils/dict_configurations.py`
- Generate vectors of topic and/or word for each text input: `src/analysis/01-generate-topic-output.ipynb`

**Aggregation**

- Aggregate topic or word vectors by a certain time unit:
    - `src/analysis/02a-aggregate-topvec.ipynb`
    - `src/analysis/02b-aggregate-wordvec.ipynb`
- Sum all topic or word vectors for aggregated analysis: 
    - `src/analysis/03a-sum-topvec.ipynb`
    - `src/analysis/03b-sum-wordvec.ipynb`

**Correlation**
- Correlate topic vectors at the aggregated level: `src/analysis/04a-intermedia-aggregated-topic-correlation.ipynb`
- Correlate topic vectors over time: `src/analysis/04b-intermedia-temporal-topic-correlation.ipynb`
- Correlate keyword vectors at the aggregated level: `src/analysis/04b-intermedia-aggregated-keywords-correlation.ipynb`
- Correlate keyword vectors by topic: `src/analysis/04c-intermedia-by-topic-keywords-correlation.ipynb`
- Correlate keyword vectors over time: `src/analysis/04c-intermedia-temporal-keywords-correlation.ipynb`


**Granger causality test**
- Run granger causality test using sliding windows: `src/analysis/06a-ias-granger-sliding-test.ipynb`
- Visualize sliding window results: `src/analysis/06b-ias-granger-sliding-vis.ipynb`
- Run granger causality test using the entire timeline: `src/analysis/06c-ias-granger-entire-timeline.ipynb`

**Other analysis**
- Visualize topic trends over time `src/analysis/05-visualize-topic-time-series.ipynb`




