# FinCast Local Setup

## Current layout

- Repo: `E:\Working Area\Comp5329_Assignment2_2026\FinCast-fts`
- Model: `E:\Working Area\Comp5329_Assignment2_2026\models\FinCast\v1.pth`
- Local env: `E:\Working Area\Comp5329_Assignment2_2026\.conda-fincast`
- Local conda cache: `E:\Working Area\Comp5329_Assignment2_2026\.conda-pkgs`
- Local conda state cache: `E:\Working Area\Comp5329_Assignment2_2026\.localappdata`

## Why this differs from upstream

The upstream repo suggests `torch 2.5.0 + cu124`.
That build does not support the `RTX 5080 Laptop GPU (sm_120)`.

This project was verified locally with:

- Python `3.11.11`
- PyTorch `2.10.0+cu128`
- CUDA runtime `12.8`

## Use the local environment

```powershell
Set-Location "E:\Working Area\Comp5329_Assignment2_2026\FinCast-fts"
& "..\.conda-fincast\python.exe" -c "import torch; print(torch.__version__)"
```

## Quick self-check

```powershell
Set-Location "E:\Working Area\Comp5329_Assignment2_2026"
.\check_fincast_setup.ps1
```

## Notes

The active project-local dependencies now live under this folder.
There is also an older external environment from the earlier setup attempt, but it is no longer required by the scripts in this project.

## Recommended starting inference settings for this laptop

- `backend = "gpu"`
- `context_len = 128` or `256`
- `horizon_len = 32`
- `batch_size = 16` or `32`

The notebook default `batch_size = 64` may still work, but `16` or `32` is a safer starting point on a 16 GB laptop GPU.
