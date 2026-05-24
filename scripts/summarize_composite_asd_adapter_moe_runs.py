from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import pandas as pd


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE = "routed_composite_asd_adapter_moe_patchtst"
DEFAULT_BASELINE = "raw_joint"
SCALE_ORDER = ("second", "minute", "hour")
SPLIT_ORDER = ("validation", "test", "zero_shot")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate routed composite ASD-adapter MoE runs and report relative "
            "NMSE improvement over raw_joint."
        )
    )
    parser.add_argument(
        "--run-root",
        type=Path,
        default=WORKSPACE_ROOT / "outputs" / "composite_asd_adapter_moe_3seed_quick",
        help="Directory containing per-seed run directories.",
    )
    parser.add_argument(
        "--run-dir",
        type=Path,
        action="append",
        default=[],
        help="Explicit run directory containing summary.csv and diagnostics.csv. May be repeated.",
    )
    parser.add_argument(
        "--pattern",
        default="seed_*",
        help="Child directory pattern used under --run-root when --run-dir is omitted.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=WORKSPACE_ROOT / "outputs" / "composite_asd_adapter_moe_3seed_quick",
    )
    parser.add_argument(
        "--report-path",
        type=Path,
        default=WORKSPACE_ROOT / "report" / "composite_asd_adapter_moe_3seed_quick_summary.md",
    )
    parser.add_argument("--candidate", default=DEFAULT_CANDIDATE)
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    parser.add_argument("--label", default="composite_asd_adapter_moe_3seed_quick")
    return parser.parse_args()


def resolve_run_dirs(args: argparse.Namespace) -> list[Path]:
    if args.run_dir:
        candidates = args.run_dir
    else:
        candidates = sorted(path for path in args.run_root.glob(args.pattern) if path.is_dir())
    run_dirs = [path for path in candidates if (path / "summary.csv").exists()]
    if not run_dirs:
        raise FileNotFoundError("No run directories with summary.csv were found.")
    return run_dirs


def first_non_null(series: pd.Series) -> object:
    values = series.dropna()
    return values.iloc[0] if len(values) else pd.NA


def load_summary(run_dirs: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        frame = pd.read_csv(run_dir / "summary.csv")
        frame["run_dir"] = str(run_dir)
        frame["run_name"] = run_dir.name
        if "seed" not in frame.columns:
            frame["seed"] = pd.NA
        frames.append(frame)
    summary = pd.concat(frames, ignore_index=True)
    summary["seed"] = pd.to_numeric(summary["seed"], errors="coerce")
    return summary


def load_diagnostics(run_dirs: Iterable[Path]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run_dir in run_dirs:
        path = run_dir / "diagnostics.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        frame["run_dir"] = str(run_dir)
        frame["run_name"] = run_dir.name
        seed = pd.NA
        summary_path = run_dir / "summary.csv"
        if summary_path.exists():
            summary = pd.read_csv(summary_path, usecols=lambda column: column == "seed")
            if "seed" in summary.columns:
                seed = first_non_null(summary["seed"])
        frame["seed"] = seed
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    diagnostics = pd.concat(frames, ignore_index=True)
    diagnostics["seed"] = pd.to_numeric(diagnostics["seed"], errors="coerce")
    return diagnostics


def aggregate_metrics(summary: pd.DataFrame, baseline: str, candidate: str) -> pd.DataFrame:
    rows = summary[summary["model"].isin([baseline, candidate])].copy()
    metric_columns = [
        "n",
        "mse",
        "mae",
        "nmse",
        "direction_accuracy_nonzero",
        "corr",
        "best_epoch",
        "stopped_epoch",
        "best_validation_mean_nmse",
    ]
    available = [column for column in metric_columns if column in rows.columns]
    grouped = rows.groupby(["split", "scale", "model"], dropna=False)[available]
    aggregate = grouped.agg(["mean", "std"]).reset_index()
    aggregate.columns = [
        "_".join(str(part) for part in column if part) if isinstance(column, tuple) else column
        for column in aggregate.columns
    ]
    return sort_by_split_scale(aggregate)


def relative_improvement(summary: pd.DataFrame, baseline: str, candidate: str) -> pd.DataFrame:
    needed = summary[summary["model"].isin([baseline, candidate])].copy()
    keys = ["run_dir", "run_name", "seed", "split", "scale"]
    wide = needed.pivot_table(index=keys, columns="model", values="nmse", aggfunc="first").reset_index()
    if baseline not in wide.columns or candidate not in wide.columns:
        raise ValueError(f"Both {baseline!r} and {candidate!r} must be present in summary.csv.")
    wide["raw_joint_nmse"] = wide[baseline]
    wide["candidate_nmse"] = wide[candidate]
    wide["relative_nmse_improvement_pct"] = (
        (wide["raw_joint_nmse"] - wide["candidate_nmse"]) / wide["raw_joint_nmse"] * 100.0
    )
    cols = keys + ["raw_joint_nmse", "candidate_nmse", "relative_nmse_improvement_pct"]
    return sort_by_split_scale(wide[cols])


def aggregate_relative(relative: pd.DataFrame) -> pd.DataFrame:
    aggregate = (
        relative.groupby(["split", "scale"], dropna=False)
        .agg(
            seed_count=("seed", "nunique"),
            raw_joint_nmse_mean=("raw_joint_nmse", "mean"),
            raw_joint_nmse_std=("raw_joint_nmse", "std"),
            candidate_nmse_mean=("candidate_nmse", "mean"),
            candidate_nmse_std=("candidate_nmse", "std"),
            relative_nmse_improvement_pct_mean=("relative_nmse_improvement_pct", "mean"),
            relative_nmse_improvement_pct_std=("relative_nmse_improvement_pct", "std"),
            positive_seed_rate=("relative_nmse_improvement_pct", lambda values: float((values > 0).mean())),
        )
        .reset_index()
    )
    return sort_by_split_scale(aggregate)


def aggregate_diagnostics(diagnostics: pd.DataFrame, candidate: str) -> pd.DataFrame:
    if diagnostics.empty:
        return pd.DataFrame()
    candidate_rows = diagnostics[diagnostics["model"] == candidate].copy()
    if candidate_rows.empty:
        return pd.DataFrame()
    keep = [
        "router_entropy",
        "router_balance_loss",
        "final_gate_mean",
        "final_mean_abs_delta",
        "composite_mean_abs_delta",
    ]
    keep.extend(column for column in candidate_rows.columns if column.startswith("expert_prob_"))
    available = [column for column in keep if column in candidate_rows.columns]
    grouped = candidate_rows.groupby(["scale"], dropna=False)[available].agg(["mean", "std"]).reset_index()
    grouped.columns = [
        "_".join(str(part) for part in column if part) if isinstance(column, tuple) else column
        for column in grouped.columns
    ]
    return sort_by_split_scale(grouped)


def split_scale_rank(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = frame.copy()
    if "split" in ranked.columns:
        ranked["_split_rank"] = ranked["split"].map({value: idx for idx, value in enumerate(SPLIT_ORDER)}).fillna(99)
    else:
        ranked["_split_rank"] = 0
    if "scale" in ranked.columns:
        ranked["_scale_rank"] = ranked["scale"].map({value: idx for idx, value in enumerate(SCALE_ORDER)}).fillna(99)
    else:
        ranked["_scale_rank"] = 0
    return ranked


def sort_by_split_scale(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = split_scale_rank(frame)
    sort_cols = [column for column in ["_split_rank", "_scale_rank", "seed", "model"] if column in ranked.columns]
    ranked = ranked.sort_values(sort_cols).drop(columns=["_split_rank", "_scale_rank"], errors="ignore")
    return ranked.reset_index(drop=True)


def is_router_collapsed(diagnostics_aggregate: pd.DataFrame) -> bool:
    if diagnostics_aggregate.empty:
        return False
    entropy_col = "router_entropy_mean"
    entropy_bad = entropy_col in diagnostics_aggregate and (diagnostics_aggregate[entropy_col] < 0.20).any()
    expert_mean_cols = [
        column
        for column in diagnostics_aggregate.columns
        if column.startswith("expert_prob_") and column.endswith("_mean")
    ]
    expert_bad = False
    if expert_mean_cols:
        expert_bad = (diagnostics_aggregate[expert_mean_cols].max(axis=1) > 0.75).any()
    return bool(entropy_bad or expert_bad)


def verdict(relative_aggregate: pd.DataFrame, diagnostics_aggregate: pd.DataFrame) -> tuple[str, list[str]]:
    notes: list[str] = []
    test = relative_aggregate[relative_aggregate["split"] == "test"].set_index("scale")
    zero_shot = relative_aggregate[relative_aggregate["split"] == "zero_shot"].set_index("scale")
    if test.empty:
        return "insufficient evidence", ["No test split was found in the aggregated runs."]

    min_seed_count = int(test["seed_count"].min()) if "seed_count" in test else 0
    if min_seed_count < 3:
        notes.append(f"Only {min_seed_count} seed(s) are available on the test split.")

    router_collapsed = is_router_collapsed(diagnostics_aggregate)
    if router_collapsed:
        notes.append("Router collapse guard tripped: low entropy or one expert dominates above 75%.")

    test_improvements = {
        scale: float(test.loc[scale, "relative_nmse_improvement_pct_mean"])
        for scale in SCALE_ORDER
        if scale in test.index
    }
    zero_improvements = {
        scale: float(zero_shot.loc[scale, "relative_nmse_improvement_pct_mean"])
        for scale in SCALE_ORDER
        if scale in zero_shot.index
    }

    test_positive = sum(value > 0 for value in test_improvements.values())
    zero_positive = sum(value > 0 for value in zero_improvements.values()) if zero_improvements else 0
    minute_ok = test_improvements.get("minute", -999.0) >= -1.0
    if zero_improvements:
        minute_ok = minute_ok and zero_improvements.get("minute", -999.0) >= -1.0

    if min_seed_count >= 3 and test_positive >= 2 and minute_ok and not router_collapsed:
        if not zero_improvements or zero_positive >= 2:
            return "upgrade to full confirmation", notes

    second_hour_positive = (
        test_improvements.get("second", -999.0) > 0 and test_improvements.get("hour", -999.0) > 0
    )
    minute_bad = test_improvements.get("minute", 0.0) < -1.0
    if min_seed_count >= 3 and second_hour_positive and minute_bad:
        return "keep as promising challenger", notes

    return "exploratory only for now", notes


def format_number(value: object) -> str:
    if pd.isna(value):
        return "nan"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        if abs(value) < 0.001 and value != 0:
            return f"{value:.4e}"
        return f"{value:.4f}"
    return str(value)


def markdown_table(frame: pd.DataFrame) -> list[str]:
    if frame.empty:
        return ["_No rows._"]
    rows = frame.copy()
    columns = list(rows.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in rows.iterrows():
        lines.append("| " + " | ".join(format_number(row[column]) for column in columns) + " |")
    return lines


def compact_relative_table(relative_aggregate: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "split",
        "scale",
        "seed_count",
        "relative_nmse_improvement_pct_mean",
        "relative_nmse_improvement_pct_std",
        "positive_seed_rate",
        "raw_joint_nmse_mean",
        "candidate_nmse_mean",
    ]
    return relative_aggregate[[column for column in keep if column in relative_aggregate.columns]]


def compact_diagnostics_table(diagnostics_aggregate: pd.DataFrame) -> pd.DataFrame:
    if diagnostics_aggregate.empty:
        return diagnostics_aggregate
    keep = [
        "scale",
        "router_entropy_mean",
        "router_entropy_std",
        "router_balance_loss_mean",
        "final_gate_mean_mean",
        "final_gate_mean_std",
    ]
    keep.extend(column for column in diagnostics_aggregate.columns if column.startswith("expert_prob_") and column.endswith("_mean"))
    return diagnostics_aggregate[[column for column in keep if column in diagnostics_aggregate.columns]]


def write_report(
    *,
    path: Path,
    label: str,
    run_dirs: list[Path],
    metrics_aggregate: pd.DataFrame,
    relative: pd.DataFrame,
    relative_aggregate: pd.DataFrame,
    diagnostics_aggregate: pd.DataFrame,
    decision: str,
    decision_notes: list[str],
) -> None:
    lines: list[str] = []
    lines.append(f"# {label}")
    lines.append("")
    lines.append(
        "This report aggregates routed composite ASD-adapter MoE runs. The main metric is "
        "`relative_nmse_improvement_pct = (raw_joint_nmse - candidate_nmse) / raw_joint_nmse * 100`."
    )
    lines.append("")
    lines.append("## Decision")
    lines.append("")
    lines.append(f"- verdict: **{decision}**")
    lines.append(f"- run count: `{len(run_dirs)}`")
    if decision_notes:
        for note in decision_notes:
            lines.append(f"- note: {note}")
    lines.append("")
    lines.append("## Relative NMSE Improvement")
    lines.append("")
    lines.extend(markdown_table(compact_relative_table(relative_aggregate)))
    lines.append("")
    lines.append("## Diagnostics")
    lines.append("")
    lines.extend(markdown_table(compact_diagnostics_table(diagnostics_aggregate)))
    lines.append("")
    lines.append("## Model Metric Aggregate")
    lines.append("")
    metric_keep = [
        "split",
        "scale",
        "model",
        "n_mean",
        "nmse_mean",
        "nmse_std",
        "mae_mean",
        "mae_std",
        "direction_accuracy_nonzero_mean",
        "direction_accuracy_nonzero_std",
        "corr_mean",
        "corr_std",
    ]
    lines.extend(markdown_table(metrics_aggregate[[column for column in metric_keep if column in metrics_aggregate.columns]]))
    lines.append("")
    lines.append("## Run Directories")
    lines.append("")
    for run_dir in run_dirs:
        lines.append(f"- `{run_dir}`")
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    output_dir = path.parent.parent / "outputs"
    lines.append("- `aggregate_model_metrics.csv`")
    lines.append("- `aggregate_relative_improvement.csv`")
    lines.append("- `seed_relative_improvement.csv`")
    lines.append("- `aggregate_diagnostics.csv`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dirs = resolve_run_dirs(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    summary = load_summary(run_dirs)
    diagnostics = load_diagnostics(run_dirs)
    metrics_aggregate = aggregate_metrics(summary, args.baseline, args.candidate)
    relative = relative_improvement(summary, args.baseline, args.candidate)
    relative_aggregate = aggregate_relative(relative)
    diagnostics_aggregate = aggregate_diagnostics(diagnostics, args.candidate)
    decision, decision_notes = verdict(relative_aggregate, diagnostics_aggregate)

    metrics_aggregate.to_csv(args.output_dir / "aggregate_model_metrics.csv", index=False)
    relative.to_csv(args.output_dir / "seed_relative_improvement.csv", index=False)
    relative_aggregate.to_csv(args.output_dir / "aggregate_relative_improvement.csv", index=False)
    diagnostics_aggregate.to_csv(args.output_dir / "aggregate_diagnostics.csv", index=False)
    write_report(
        path=args.report_path,
        label=args.label,
        run_dirs=run_dirs,
        metrics_aggregate=metrics_aggregate,
        relative=relative,
        relative_aggregate=relative_aggregate,
        diagnostics_aggregate=diagnostics_aggregate,
        decision=decision,
        decision_notes=decision_notes,
    )
    print(f"saved_report={args.report_path}")
    print(f"verdict={decision}")


if __name__ == "__main__":
    main()
