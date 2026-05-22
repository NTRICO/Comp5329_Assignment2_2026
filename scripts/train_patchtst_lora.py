from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from src.baselines.patchtst_lora import (
    LoRAConfig,
    PatchTSTForecastConfig,
    PatchTSTLoRA,
    count_parameters,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a PatchTST financial forecaster, then LoRA-tune it."
    )
    parser.add_argument(
        "--cache",
        default=str(WORKSPACE_ROOT / "data" / "cache" / "position_fincast_daily_cache.npz"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(WORKSPACE_ROOT / "outputs" / "patchtst_lora"),
    )
    parser.add_argument("--context-length", type=int, default=64)
    parser.add_argument("--patch-length", type=int, default=16)
    parser.add_argument("--patch-stride", type=int, default=8)
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--lora-alpha", type=float, default=8.0)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--base-epochs", type=int, default=5)
    parser.add_argument("--lora-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--lora-learning-rate", type=float, default=3e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.1)
    parser.add_argument("--test-fraction", type=float, default=0.2)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-validation-samples", type=int, default=None)
    parser.add_argument("--max-test-samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny shape/training check instead of a full experiment.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.smoke:
        args.base_epochs = min(args.base_epochs, 1)
        args.lora_epochs = min(args.lora_epochs, 1)
        args.max_train_samples = args.max_train_samples or 256
        args.max_validation_samples = args.max_validation_samples or 128
        args.max_test_samples = args.max_test_samples or 128

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = _select_device(args.device)

    splits = build_return_prediction_splits(
        cache_path=Path(args.cache),
        context_length=args.context_length,
        validation_fraction=args.validation_fraction,
        test_fraction=args.test_fraction,
    )
    train_x, train_y = _cap_split(
        splits["train_x"],
        splits["train_y"],
        args.max_train_samples,
    )
    validation_x, validation_y = _cap_split(
        splits["validation_x"],
        splits["validation_y"],
        args.max_validation_samples,
    )
    test_x, test_y = _cap_split(
        splits["test_x"],
        splits["test_y"],
        args.max_test_samples,
    )

    normalizer = fit_normalizer(train_x, train_y)
    train_ds = TensorDataset(
        normalize_x(train_x, normalizer),
        normalize_y(train_y, normalizer),
    )
    validation_ds = TensorDataset(
        normalize_x(validation_x, normalizer),
        normalize_y(validation_y, normalizer),
    )
    test_ds = TensorDataset(
        normalize_x(test_x, normalizer),
        normalize_y(test_y, normalizer),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_ds, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    config = PatchTSTForecastConfig(
        context_length=args.context_length,
        input_channels=1,
        patch_length=args.patch_length,
        patch_stride=args.patch_stride,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        lora=LoRAConfig(
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            enabled=False,
        ),
    )
    model = PatchTSTLoRA(config).to(device)
    print(f"device={device}")
    print(f"samples train={len(train_ds)} validation={len(validation_ds)} test={len(test_ds)}")
    print(f"initial_params={count_parameters(model)}")

    model.set_lora_enabled(False)
    base_optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    print("stage=base_patchtst")
    base_history = train_epochs(
        model,
        train_loader,
        validation_loader,
        optimizer=base_optimizer,
        device=device,
        epochs=args.base_epochs,
        normalizer=normalizer,
    )
    base_validation = evaluate(model, validation_loader, device=device, normalizer=normalizer)
    base_test = evaluate(model, test_loader, device=device, normalizer=normalizer)

    model.freeze_base_for_lora(train_head=True)
    model.set_lora_enabled(True)
    print(f"lora_params={count_parameters(model)}")
    lora_optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lora_learning_rate,
        weight_decay=args.weight_decay,
    )
    print("stage=lora_adaptation")
    lora_history = train_epochs(
        model,
        train_loader,
        validation_loader,
        optimizer=lora_optimizer,
        device=device,
        epochs=args.lora_epochs,
        normalizer=normalizer,
    )
    lora_validation = evaluate(model, validation_loader, device=device, normalizer=normalizer)
    lora_test = evaluate(model, test_loader, device=device, normalizer=normalizer)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "config": asdict(config),
        "cache": str(Path(args.cache)),
        "normalizer": {key: float(value) for key, value in normalizer.items()},
        "samples": {
            "train": len(train_ds),
            "validation": len(validation_ds),
            "test": len(test_ds),
        },
        "base_history": base_history,
        "lora_history": lora_history,
        "base_validation": base_validation,
        "base_test": base_test,
        "lora_validation": lora_validation,
        "lora_test": lora_test,
    }
    metrics_path = output_dir / "patchtst_lora_metrics.json"
    checkpoint_path = output_dir / "patchtst_lora.pt"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(config),
            "normalizer": normalizer,
            "metrics": metrics,
        },
        checkpoint_path,
    )
    print(f"saved_metrics={metrics_path}")
    print(f"saved_checkpoint={checkpoint_path}")
    print(
        "summary "
        f"base_val_mse={base_validation['mse_raw']:.8f} "
        f"lora_val_mse={lora_validation['mse_raw']:.8f} "
        f"base_test_mse={base_test['mse_raw']:.8f} "
        f"lora_test_mse={lora_test['mse_raw']:.8f}"
    )


def build_return_prediction_splits(
    *,
    cache_path: Path,
    context_length: int,
    validation_fraction: float,
    test_fraction: float,
) -> dict[str, torch.Tensor]:
    cache = np.load(cache_path, allow_pickle=True)
    last_values = np.asarray(cache["last_values"], dtype=np.float64)
    targets = np.asarray(cache["realized_returns"], dtype=np.float64)
    asset_names = np.asarray(cache["asset_names"], dtype=object)

    split_arrays: dict[str, list[np.ndarray]] = {
        "train_x": [],
        "train_y": [],
        "validation_x": [],
        "validation_y": [],
        "test_x": [],
        "test_y": [],
    }
    for start, end in contiguous_runs(asset_names):
        prices = last_values[start:end]
        realized = targets[start:end]
        if len(prices) < context_length + 2:
            continue
        log_prices = np.log(np.maximum(prices, 1e-12))
        returns = np.zeros_like(log_prices)
        returns[1:] = np.diff(log_prices)

        windows = []
        y = []
        for t in range(context_length - 1, len(returns)):
            windows.append(returns[t - context_length + 1 : t + 1, None])
            y.append(realized[t])
        windows_arr = np.asarray(windows, dtype=np.float32)
        y_arr = np.asarray(y, dtype=np.float32).reshape(-1, 1)
        finite_mask = np.isfinite(windows_arr).all(axis=(1, 2)) & np.isfinite(y_arr).all(axis=1)
        windows_arr = windows_arr[finite_mask]
        y_arr = y_arr[finite_mask]
        n = len(windows_arr)
        if n < 5:
            continue
        n_test = max(1, int(n * test_fraction))
        n_validation = max(1, int(n * validation_fraction)) if validation_fraction > 0 else 0
        n_train = n - n_validation - n_test
        if n_train <= 0:
            continue
        split_arrays["train_x"].append(windows_arr[:n_train])
        split_arrays["train_y"].append(y_arr[:n_train])
        if n_validation:
            split_arrays["validation_x"].append(windows_arr[n_train : n_train + n_validation])
            split_arrays["validation_y"].append(y_arr[n_train : n_train + n_validation])
        split_arrays["test_x"].append(windows_arr[n_train + n_validation :])
        split_arrays["test_y"].append(y_arr[n_train + n_validation :])

    return {
        key: torch.as_tensor(np.concatenate(parts, axis=0), dtype=torch.float32)
        for key, parts in split_arrays.items()
        if parts
    }


def contiguous_runs(values: np.ndarray) -> list[tuple[int, int]]:
    runs: list[tuple[int, int]] = []
    if len(values) == 0:
        return runs
    start = 0
    current = values[0]
    for i in range(1, len(values)):
        if values[i] != current:
            runs.append((start, i))
            start = i
            current = values[i]
    runs.append((start, len(values)))
    return runs


def fit_normalizer(x: torch.Tensor, y: torch.Tensor) -> dict[str, float]:
    return {
        "x_mean": float(x.mean()),
        "x_std": float(x.std().clamp_min(1e-8)),
        "y_mean": float(y.mean()),
        "y_std": float(y.std().clamp_min(1e-8)),
    }


def normalize_x(x: torch.Tensor, normalizer: dict[str, float]) -> torch.Tensor:
    return (x - normalizer["x_mean"]) / normalizer["x_std"]


def normalize_y(y: torch.Tensor, normalizer: dict[str, float]) -> torch.Tensor:
    return (y - normalizer["y_mean"]) / normalizer["y_std"]


def denormalize_y(y: torch.Tensor, normalizer: dict[str, float]) -> torch.Tensor:
    return y * normalizer["y_std"] + normalizer["y_mean"]


def train_epochs(
    model: PatchTSTLoRA,
    train_loader: DataLoader,
    validation_loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    normalizer: dict[str, float],
) -> list[dict[str, float]]:
    history = []
    best_mse = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer=optimizer, device=device)
        validation = evaluate(model, validation_loader, device=device, normalizer=normalizer)
        row = {"epoch": epoch, "train_loss_scaled": train_loss, **validation}
        history.append(row)
        if validation["mse_raw"] < best_mse:
            best_mse = validation["mse_raw"]
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        print(
            f"epoch={epoch:03d} train_loss_scaled={train_loss:.6f} "
            f"val_mse_raw={validation['mse_raw']:.8f} "
            f"val_direction_acc={validation['direction_accuracy']:.4f}"
        )
    if best_state is not None:
        model.load_state_dict(best_state)
        print(f"restored_best_val_mse={best_mse:.8f}")
    return history


def train_one_epoch(
    model: PatchTSTLoRA,
    loader: DataLoader,
    *,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    total = 0.0
    count = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x).squeeze(1)
        loss = F.mse_loss(prediction, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += float(loss.detach().cpu()) * x.shape[0]
        count += x.shape[0]
    return total / max(count, 1)


@torch.no_grad()
def evaluate(
    model: PatchTSTLoRA,
    loader: DataLoader,
    *,
    device: torch.device,
    normalizer: dict[str, float],
) -> dict[str, float]:
    model.eval()
    preds = []
    actuals = []
    for x, y in loader:
        pred_scaled = model(x.to(device)).squeeze(1).cpu()
        preds.append(denormalize_y(pred_scaled, normalizer))
        actuals.append(denormalize_y(y, normalizer))
    pred = torch.cat(preds, dim=0)
    actual = torch.cat(actuals, dim=0)
    mse = torch.mean((pred - actual).square()).item()
    mae = torch.mean(torch.abs(pred - actual)).item()
    direction = (torch.sign(pred) == torch.sign(actual)).float().mean().item()
    return {
        "mse_raw": float(mse),
        "mae_raw": float(mae),
        "direction_accuracy": float(direction),
    }


def _cap_split(
    x: torch.Tensor,
    y: torch.Tensor,
    cap: int | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if cap is None or cap >= len(x):
        return x, y
    indices = torch.linspace(0, len(x) - 1, steps=cap).round().long()
    return x[indices], y[indices]


def _select_device(choice: str) -> torch.device:
    if choice == "cpu":
        return torch.device("cpu")
    if choice == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    main()
