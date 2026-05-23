# Data Handoff For Teammate

## Give This Raw Data Folder

Send this folder to the teammate:

```text
data/high-frequency/Optiver_additional data/Optiver_additional data/
```

Required files for reproducing the current cache:

```text
order_book_feature.csv
order_book_target.csv
train.csv
time_id_reference.csv
```

Optional files:

```text
trades.csv
trades.parquet
order_book_feature.parquet
order_book_target.parquet
stock_ids.csv
```

The current code reads the CSV files, not the parquet files, because the local
environment does not require `pyarrow` or `fastparquet`.

## Build The Cache

After placing the raw folder under the same project path, run:

```powershell
.\.conda-fincast\python.exe scripts\build_optiver_additional_second_feature_cache.py
```

Expected output:

```text
data/cache/position_optiver_additional_true_hour_second_feature_cache_10stocks_512h.npz
```

The local additional dataset has 10 usable dense stocks:

```text
stock_0 ... stock_9
```

Current default split:

```text
train stocks: 0-8
zero-shot:    9
```

## Current Experiment Protocol

```text
patch preset: balanced_60_45_24

second: context 60 seconds, target next 10-second cumulative return
minute: context 45 minutes, target next 1-minute return
hour:   context 24 true-hour time_ids, target next true-hour return
```

Current main runner:

```powershell
.\.conda-fincast\python.exe scripts\evaluate_gated_pre_asd_32stock_multiseed.py
```

Current main config:

```text
configs/recommended_patchtst_main.json
```

## Important

The raw data and generated cache are not tracked by Git because they are large.
GitHub contains only code, configs, and reports.
