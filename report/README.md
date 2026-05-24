# PatchTST Experiment Index

This folder is the committed evidence layer for the PatchTST optimization task.
Raw experiment folders under `outputs/` are intentionally not committed.

## Recommended Reading Order

1. `TEAMMATE_HANDOFF.md`
2. `gated_pre_asd_32stock_multiseed.md`
3. `multichannel_asd_lora_moe_32stock_multiseed.md`
4. `asd_moe_conflict_diagnosis.md`

## Current Model Decision

The current robust main model is:

```text
return input
-> scale-aware ASD
-> LoRA-MoE sequence adapter
-> gated residual to raw return
-> shared PatchTST
-> scale-specific linear head
```

It is selected because it is the most stable unified three-scale model in the
32-stock multi-seed setting.

The best open challenger is the multi-channel ASD + LoRA-MoE variant. It improves
second/hour more strongly but still has unstable minute-scale behavior.

The newest exploratory challenger is the routed composite ASD-adapter MoE branch.
It routes each position to composite experts that pair ASD with LoRA/MLP
enhancers. The current default expert layout is `one_mlp`
(`[ASD+LoRA, ASD+LoRA, ASD+LoRA, ASD+MLP]`). It is not the main paper claim yet
because minute-scale behavior is still fragile.

## Key Reports

- `gated_pre_asd_32stock_multiseed.md`: selected robust main model.
- `multichannel_asd_lora_moe_32stock_multiseed.md`: latest multi-channel
  confirmation.
- `direct_price_input_patchtst_32stock_multiseed.md`: price-input variant;
  useful mostly for hour-scale analysis.
- `frequency_router_shared_asd_mid_moe_32stock_balanced_router.md`: routed ASD
  plus intra-encoder MoE ablation.
- `frequency_router_scale_specific_patchtst_32stock_raw_compare.md`: check that
  the earlier small-data hour gain did not replicate on 32-stock data.
- `asd_moe_conflict_diagnosis.md`: why ASD and MoE/LoRA can conflict.
- `scale_aware_asd_lora_moe_final_decision.md`: earlier combined-model gate.
- `composite_asd_adapter_moe_progress.md`: current routed composite ASD-adapter
  MoE challenger and one-MLP decision trail.
- `composite_asd_adapter_moe_3seed_quick_summary.md`: three-seed quick check for
  the first composite router.
- `composite_asd_adapter_moe_one_mlp_default_small_seed42.md`: default one-MLP
  small-data smoke/quick result.

## Reporting Rule

For presentation, report improvements as percentages relative to the relevant raw
PatchTST baseline. Avoid leading with raw normalized-error values; they are easy
to over-interpret across scales.
