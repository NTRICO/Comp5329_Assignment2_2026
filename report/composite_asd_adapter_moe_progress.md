# Routed Composite ASD-Adapter MoE Progress

This report tracks the current exploratory challenger:

```text
returns
-> position/scale router
-> composite experts: ASD + lightweight LoRA/MLP enhancer
-> gated residual to raw returns
-> shared PatchTST
-> scale-specific head
```

Raw output folders under `outputs/` remain uncommitted. The committed reports
preserve the decision trail.

## Current Setting

The composite branch now fixes the default expert pattern to `one_mlp`:

```text
[ASD+LoRA, ASD+LoRA, ASD+LoRA, ASD+MLP]
```

This choice came from a seed-42 small-data pattern sweep. `one_mlp` produced the
best validation mean among the tested expert layouts while keeping the strong
second/hour gains.

## Evidence So Far

Default alternating 3-seed quick test:

| split | second | minute | hour |
| --- | ---: | ---: | ---: |
| validation | +4.77% | -1.23% | +6.02% |
| test | +5.15% | -2.47% | +9.10% |
| zero-shot | +7.84% | -1.31% | +13.60% |

`one_mlp` seed-42 small-data test:

| split | second | minute | hour |
| --- | ---: | ---: | ---: |
| validation | +5.22% | -1.88% | +4.95% |
| test | +5.27% | -3.37% | +8.17% |
| zero-shot | +9.03% | -0.44% | +12.65% |

Percentages are relative NMSE improvement over the same-run `raw_joint`
baseline.

## Interpretation

The composite router is a promising challenger, not the final main model. It is
consistent on second/hour and especially useful on zero-shot hour, but minute
remains fragile. The likely issue is not router collapse: entropy stays near
0.5, and expert usage remains distributed. The weaker point is that minute has a
low signal-to-noise target where the extra ASD/adapter perturbation can hurt more
than it helps.

The current paper figure and final model claim should therefore stay on the
validated gated pre-ASD + LoRA-MoE PatchTST path. This branch can be reported as
an exploratory ablation or future-work direction unless a later full multi-seed
run fixes minute without sacrificing second/hour.

## Key Files

- `scripts/evaluate_composite_asd_adapter_moe.py`
- `scripts/summarize_composite_asd_adapter_moe_runs.py`
- `report/composite_asd_adapter_moe_3seed_quick_summary.md`
- `report/composite_asd_adapter_moe_one_mlp_default_small_seed42.md`
- `report/composite_asd_adapter_moe_pattern_one_mlp_seed42.md`
