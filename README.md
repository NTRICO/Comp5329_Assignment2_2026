# COMP5329 FinCast Position Trader

This project builds a position-aware daily trader on top of frozen FinCast forecasts.
FinCast is treated as an external dependency: it generates predictive distributions,
while this repository contains the data pipeline, cache builders, trader heads,
training notebooks, baselines, and evaluation scaffolding.

## Layout

```text
src/
  baselines/          Mean-variance and other comparison strategies
  datasets/           ETF data loading and cached sequence datasets
  fincast_io/         FinCast wrapper, forecast cache, encoder feature cache
  trader/             Trader heads: cnn_gru, encoder_transformer, encoder_policy
  training/           Mean-variance-turnover loss and train/eval loops
scripts/
  build_fincast_cache.py
  build_encoder_cache.py
  train_trader.py
notebooks/
  01_data_exploration.ipynb
  02_trader_training.ipynb
  03_fincast_smoke_test.ipynb
data/raw/
  etf_daily_close.csv
```

## External Files

These files are required locally but are intentionally not committed:

```text
FinCast-fts/                 Frozen upstream FinCast source tree
models/FinCast/v1.pth        FinCast checkpoint
data/cache/*.npz             Generated forecast/encoder caches
outputs/checkpoints/*.pt     Trained trader checkpoints
.conda-fincast/              Local Python environment
```

Clone FinCast separately at the project root:

```powershell
git clone https://github.com/vincent05r/FinCast-fts.git FinCast-fts
```

Place the FinCast checkpoint at:

```text
models/FinCast/v1.pth
```

See `FINCAST_SETUP.md` for the local CUDA/PyTorch environment notes.

## Current Pipeline

1. Daily ETF close data lives in `data/raw/etf_daily_close.csv`.
2. Frozen FinCast generates a daily cache:

   ```powershell
   & ".\.conda-fincast\python.exe" scripts\build_fincast_cache.py
   ```

   Default output:

   ```text
   data/cache/position_fincast_daily_cache.npz
   ```

3. Train the trader from the daily cache:

   ```powershell
   & ".\.conda-fincast\python.exe" scripts\train_trader.py
   ```

4. Main notebook workflow:

   ```text
   notebooks/02_trader_training.ipynb
   ```

## Trader Heads

Two main policy heads are available:

```text
model_kind = "cnn_gru"
```

Uses `src/trader/cnn_gru.py`:

```text
FinCast forecast patch -> Conv1D encoder -> GRU -> position
```

```text
model_kind = "encoder_transformer"
```

Uses `src/trader/encoder_transformer.py`:

```text
FinCast forecast patch -> vanilla TransformerEncoder -> GRU -> position
```

The direct hidden-feature branch is in `src/trader/encoder_policy.py` and
`scripts/build_encoder_cache.py`.

## Notes

- FinCast stays frozen; do not modify `FinCast-fts/` for trader experiments.
- Current daily label is next trading day return: `holding_horizon=1`.
- Current split is time-based per ETF: first 80% train, last 20% test.
- Cache and checkpoint artifacts are reproducible and excluded from Git.
