# COMP5329 PatchTST Optimization for Financial Time Series

This repository is now organized around optimizing and stress-testing PatchTST
for noisy financial time-series forecasting. The earlier FinCast position-trader
pipeline remains in the codebase as historical baseline infrastructure, but the
main assignment direction is PatchTST-centric model adaptation.

The current experiments focus on:

- multi-scale PatchTST baselines on second, minute, hour, day, and week data;
- Optiver high-frequency feature caches and FinCast-Paper-test benchmarks;
- lightweight denoising and adaptation modules around PatchTST;
- robustness checks that compare raw PatchTST against ASD, LoRA-MoE, and
  supplementary multi-channel or level-domain variants.

## Project Layout

```text
src/
  baselines/
    patchtst_lora.py              Self-contained PatchTST + LoRA baseline
    scale_aware_asd_patchtst.py   Scale-aware ASD / adapter modules
    position_rules.py             Simple policy baselines for legacy tests
  datasets/
    optiver_features.py           Optiver high-frequency feature utilities
    trader_dataset.py             Legacy and cache-backed sequence datasets
  eval/
    metrics.py                    Forecasting and backtest metrics
    backtest.py                   Legacy policy/backtest utilities
  fincast_io/
    simple_forecaster.py          Lightweight FinCast-style forecaster helpers
  trader/
    ...                           Historical position-trader components
scripts/
  evaluate_*patchtst*.py          PatchTST benchmark and ablation entrypoints
  build_optiver_*.py              Optiver feature/cache builders
  train_patchtst_lora.py          Daily-cache PatchTST LoRA baseline
  train_trader.py                 Historical FinCast trader trainer
report/
  README.md                       Current PatchTST result index
  *_patchtst*.md                  Experiment notes and robustness reports
tests/
  test_*patchtst*.py              PatchTST and adapter smoke tests
  test_optiver_features.py        Feature/cache tests
```

## Local Artifacts

Large datasets, checkpoints, and generated outputs are intentionally kept local.
They are required to reproduce the experiments, but they should not be committed:

```text
.conda-fincast/                         Local Python environment
FinCast-fts/                            Frozen upstream FinCast source tree
third_party/PatchTST/                   Local PatchTST reference checkout
models/                                 Local checkpoints
data/cache/                             Generated NPZ caches
data/high-frequency/                    Optiver high-frequency source data
data/fincast_inputs/                    Generated FinCast-style inputs
data/raw/FinCast-Paper-test/            Local copy of the paper-test data
outputs/                                Experiment outputs and checkpoints
```

Use the local project interpreter for experiments:

```powershell
& ".\.conda-fincast\python.exe" <script>
```

The system Python is not the reliable environment for this checkout.

## Main Experiment Entrypoints

Vanilla PatchTST and time-scale baselines:

```powershell
& ".\.conda-fincast\python.exe" scripts\evaluate_vanilla_timescales_additional_stock.py
& ".\.conda-fincast\python.exe" scripts\evaluate_abc_vanilla_timescales.py
& ".\.conda-fincast\python.exe" scripts\evaluate_hf_fincast_paper_test.py
```

Spectral denoising and scale-aware ASD:

```powershell
& ".\.conda-fincast\python.exe" scripts\evaluate_optiver_spectral_denoise_patchtst.py
& ".\.conda-fincast\python.exe" scripts\evaluate_scale_aware_asd_patchtst.py
```

Supplementary model variants:

```powershell
& ".\.conda-fincast\python.exe" scripts\train_patchtst_lora.py
& ".\.conda-fincast\python.exe" scripts\evaluate_level_asd_patchtst.py
& ".\.conda-fincast\python.exe" scripts\evaluate_multichannel_patchtst.py
```

Historical FinCast trader commands are still available, but they are no longer
the primary narrative for the assignment.

## Evidence Map

The concise report index is:

```text
report/README.md
```

It records the committed, human-readable evidence tables for:

- vanilla PatchTST and FinCast-Paper-test baselines;
- raw vs spectral-denoised PatchTST;
- scale-aware ASD robustness;
- ASD + LoRA-MoE guardrails;
- multi-channel and level-ASD supplementary runs.

Generated CSV/JSON/PNG artifacts remain under `outputs/` for local inspection.

## Testing

Run focused tests with the local interpreter:

```powershell
& ".\.conda-fincast\python.exe" -m pytest tests
```

If `pytest` is unavailable in the local environment, run the relevant test
modules directly or install the test dependency in `.conda-fincast`.

## Notes

- PatchTST is treated as a forecasting backbone, not a trading policy.
- Strong zero-return baselines are expected on several financial forecasting
  surfaces, so results should be interpreted against zero and last-return rows.
- Hour-scale results often have much smaller test counts than second-scale
  results; report them with the corresponding sample-size caveat.
- FinCast remains useful as a benchmark and data source, but the project story is
  PatchTST optimization.
