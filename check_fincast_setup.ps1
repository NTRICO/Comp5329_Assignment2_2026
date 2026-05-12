$ErrorActionPreference = "Stop"

$envPython = "E:\Working Area\Comp5329_Assignment2_2026\.conda-fincast\python.exe"
$repoRoot = "E:\Working Area\Comp5329_Assignment2_2026\FinCast-fts"
$modelPath = "E:\Working Area\Comp5329_Assignment2_2026\models\FinCast\v1.pth"

if (-not (Test-Path $envPython)) {
    throw "Missing FinCast Python environment at $envPython"
}

if (-not (Test-Path $modelPath)) {
    throw "Missing FinCast checkpoint at $modelPath"
}

Push-Location $repoRoot
try {
    @"
from types import SimpleNamespace
import torch
from tools.inference_utils import get_model_api

print(f"torch={torch.__version__}")
print(f"cuda_available={torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"device_name={torch.cuda.get_device_name(0)}")
    print(f"device_capability={torch.cuda.get_device_capability(0)}")

cfg = SimpleNamespace(
    model_path=r"$modelPath",
    backend="gpu",
    horizon_len=32,
    context_len=128,
    num_experts=4,
    gating_top_n=2,
    load_from_compile=True,
    forecast_mode="mean",
)

get_model_api(cfg)
print("model_loaded_ok=True")
"@ | & $envPython -
}
finally {
    Pop-Location
}
