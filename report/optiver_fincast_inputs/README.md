# Optiver FinCast Input Preview

This artifact converts the first 8 Optiver stocks into FinCast-ready scalar context windows.

- input cache: `E:\Working Area\Comp5329_Assignment2_2026\data\fincast_inputs\optiver_8stocks_wap1_second_fincast_inputs.npz`
- sample CSV: `E:\Working Area\Comp5329_Assignment2_2026\data\fincast_inputs\optiver_8stocks_wap1_second_fincast_sample_contexts.csv`
- contexts shape: `(428960, 128)`
- future_values shape: `(428960, 32)`
- data_frequency: `S`
- price_source: `wap1`

The direct FinCast API contract is:

```python
model_api.forecast([row for row in contexts], freq=[freq_value] * len(contexts))
```

Each row is a single-variable WAP1 price history from one stock-time_id episode.
Windows never cross anonymous Optiver time_id boundaries.