param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("validate", "checkout", "checkout-vision", "build", "build-vision", "install-wheel", "install-vision-wheel", "smoke", "smoke-vision", "probe")]
    [string]$Action
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$TheRockRoot = Join-Path $RepoRoot "external\TheRock"
$TheRockPyTorch = Join-Path $TheRockRoot "external-builds\pytorch"
$DefaultBuildRoot = Join-Path ([System.IO.Path]::GetPathRoot($RepoRoot)) "b\rx6900"
$BuildRoot = if ($env:RX6900_BUILD_ROOT) { $env:RX6900_BUILD_ROOT } else { $DefaultBuildRoot }
$PyTorchSource = Join-Path $BuildRoot "pytorch-src"
$VisionSource = Join-Path $BuildRoot "pytorch_vision"
$WheelOutput = Join-Path $BuildRoot "wheels"
$RocmIndexUrl = "https://rocm.nightlies.amd.com/v2/gfx103X-dgpu/"
$BuildJobs = if ($env:RX6900_BUILD_JOBS) { $env:RX6900_BUILD_JOBS } else { "1" }

function Enter-VsDevShell {
    $vcvars = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
    if (-not (Test-Path -LiteralPath $vcvars)) {
        throw "Visual Studio Build Tools vcvars64.bat not found: $vcvars"
    }

    $lines = cmd.exe /d /s /c "`"$vcvars`" >nul && set"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to load Visual Studio x64 build environment."
    }

    foreach ($line in $lines) {
        $index = $line.IndexOf("=")
        if ($index -gt 0) {
            $name = $line.Substring(0, $index)
            $value = $line.Substring($index + 1)
            Set-Item -Path "Env:$name" -Value $value
        }
    }
}

function Clear-ConflictingRocmEnvironment {
    foreach ($name in @("HIP_PATH", "ROCM_PATH")) {
        if (Test-Path "Env:$name") {
            Remove-Item "Env:$name"
        }
    }
}

function Join-EnvironmentPathList {
    param(
        [string[]]$Paths
    )

    ($Paths | Where-Object { $_ } | Select-Object -Unique) -join [System.IO.Path]::PathSeparator
}

function Set-TorchVisionCodecEnvironment {
    $pythonPrefix = (& python -c "import sys; print(sys.prefix)").Trim()
    if ($LASTEXITCODE -ne 0 -or -not $pythonPrefix) {
        throw "Failed to resolve the active Python prefix."
    }

    $libraryRoot = Join-Path $pythonPrefix "Library"
    $libraryBin = Join-Path $libraryRoot "bin"
    $libraryInclude = Join-Path $libraryRoot "include"
    $libraryLib = Join-Path $libraryRoot "lib"

    Test-RequiredPath -Path (Join-Path $libraryBin "pngfix.exe") -Message "pngfix.exe not found. Run: pixi install"
    Test-RequiredPath -Path (Join-Path $libraryInclude "jpeglib.h") -Message "jpeglib.h not found. Run: pixi install"
    Test-RequiredPath -Path (Join-Path $libraryInclude "webp\decode.h") -Message "webp/decode.h not found. Run: pixi install"
    Test-RequiredPath -Path (Join-Path $libraryLib "jpeg.lib") -Message "jpeg.lib not found. Run: pixi install"
    Test-RequiredPath -Path (Join-Path $libraryLib "libwebp.lib") -Message "libwebp.lib not found. Run: pixi install"

    $env:PATH = Join-EnvironmentPathList @($libraryBin, $env:PATH)
    $env:TORCHVISION_INCLUDE = Join-EnvironmentPathList @($libraryInclude, $env:TORCHVISION_INCLUDE)
    $env:TORCHVISION_LIBRARY = Join-EnvironmentPathList @($libraryLib, $env:TORCHVISION_LIBRARY)
    $env:TORCHVISION_USE_PNG = "1"
    $env:TORCHVISION_USE_JPEG = "1"
    $env:TORCHVISION_USE_WEBP = "1"
}

function Remove-VisionBuildDirectory {
    $visionBuildDir = Join-Path $VisionSource "build"
    if (-not (Test-Path -LiteralPath $visionBuildDir)) {
        return
    }

    $resolvedVision = (Resolve-Path -LiteralPath $VisionSource).Path
    $resolvedBuildDir = (Resolve-Path -LiteralPath $visionBuildDir).Path
    if (-not $resolvedBuildDir.StartsWith($resolvedVision + [System.IO.Path]::DirectorySeparatorChar, [System.StringComparison]::OrdinalIgnoreCase)) {
        throw "Refusing to remove unexpected build directory: $resolvedBuildDir"
    }

    Remove-Item -LiteralPath $resolvedBuildDir -Recurse -Force
}

function Invoke-RepoPython {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [string]$WorkingDirectory = $RepoRoot
    )

    Push-Location $WorkingDirectory
    try {
        & python @Arguments
    }
    finally {
        Pop-Location
    }
}

function Invoke-PythonOutsideSource {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Code
    )

    $probeCwd = Join-Path $BuildRoot "probe-cwd"
    New-Item -ItemType Directory -Force -Path $probeCwd | Out-Null

    $previousSourceTorch = $env:RX6900_SOURCE_TORCH
    $env:RX6900_SOURCE_TORCH = Join-Path $RepoRoot "torch"

    Push-Location $probeCwd
    try {
        $probeFile = Join-Path $probeCwd "rx6900_probe.py"
        Set-Content -LiteralPath $probeFile -Value $Code -Encoding UTF8

        try {
            & python $probeFile
            if ($LASTEXITCODE -ne 0) {
                throw "python probe failed with exit code $LASTEXITCODE"
            }
        }
        finally {
            Remove-Item -LiteralPath $probeFile -Force -ErrorAction SilentlyContinue
        }
    }
    finally {
        Pop-Location

        if ($null -eq $previousSourceTorch) {
            Remove-Item Env:RX6900_SOURCE_TORCH -ErrorAction SilentlyContinue
        }
        else {
            $env:RX6900_SOURCE_TORCH = $previousSourceTorch
        }
    }
}

function Invoke-Git {
    param(
        [Parameter(Mandatory = $true)]
        [string[]]$Arguments,
        [string]$WorkingDirectory = $RepoRoot,
        [ValidateSet("true", "false")]
        [string]$CoreSymlinks = "true"
    )

    Push-Location $WorkingDirectory
    try {
        & git -c core.longpaths=true -c "core.symlinks=$CoreSymlinks" @Arguments
        if ($LASTEXITCODE -ne 0) {
            throw "git $($Arguments -join ' ') failed with exit code $LASTEXITCODE"
        }
    }
    finally {
        Pop-Location
    }
}

function Test-SymbolicLinkCreation {
    $probeDir = Join-Path $BuildRoot ".symlink-probe"
    Remove-Item -LiteralPath $probeDir -Recurse -Force -ErrorAction SilentlyContinue

    try {
        New-Item -ItemType Directory -Force -Path $probeDir | Out-Null
        $target = Join-Path $probeDir "target.txt"
        $link = Join-Path $probeDir "link.txt"
        Set-Content -LiteralPath $target -Value "ok"
        New-Item -ItemType SymbolicLink -Path $link -Target $target -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
    finally {
        Remove-Item -LiteralPath $probeDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}

function Initialize-PyTorchSubmodules {
    $coreSymlinks = if (Test-SymbolicLinkCreation) { "true" } else { "false" }
    if ($coreSymlinks -eq "false") {
        Write-Host "Symlink creation is not available; checking out submodules with core.symlinks=false."
    }

    $moduleLines = & git -C $PyTorchSource config --file .gitmodules --get-regexp "path$"
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to read PyTorch .gitmodules under $PyTorchSource"
    }

    $submodulePaths = @()
    foreach ($line in $moduleLines) {
        $parts = $line -split "\s+", 2
        if ($parts.Count -ne 2) {
            continue
        }

        $path = $parts[1].Trim()
        if ($path -and $path -ne "external/TheRock") {
            $submodulePaths += $path
        }
    }

    if (-not $submodulePaths) {
        throw "No PyTorch submodules found in $PyTorchSource"
    }

    Invoke-Git -WorkingDirectory $PyTorchSource -Arguments (@(
        "submodule",
        "sync",
        "--recursive",
        "--"
    ) + $submodulePaths) -CoreSymlinks $coreSymlinks

    Invoke-Git -WorkingDirectory $PyTorchSource -Arguments (@(
        "submodule",
        "update",
        "--init",
        "--recursive",
        "--jobs",
        "10",
        "--"
    ) + $submodulePaths) -CoreSymlinks $coreSymlinks

    Invoke-Git -WorkingDirectory $PyTorchSource -Arguments @(
        "submodule",
        "sync",
        "--",
        "external/TheRock"
    ) -CoreSymlinks $coreSymlinks

    Invoke-Git -WorkingDirectory $PyTorchSource -Arguments @(
        "submodule",
        "update",
        "--init",
        "--jobs",
        "1",
        "--",
        "external/TheRock"
    ) -CoreSymlinks $coreSymlinks
}

function Test-RequiredPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Path,
        [Parameter(Mandatory = $true)]
        [string]$Message
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw $Message
    }
}

function Get-PinnedVisionCommit {
    $pinFile = Join-Path $PyTorchSource ".github\ci_commit_pins\vision.txt"
    Test-RequiredPath -Path $pinFile -Message "PyTorch vision pin not found: $pinFile. Run: pixi run checkout-pytorch"
    return (Get-Content -LiteralPath $pinFile -Raw).Trim()
}

switch ($Action) {
    "validate" {
        Clear-ConflictingRocmEnvironment
        Enter-VsDevShell
        Push-Location $TheRockRoot
        try {
            powershell -ExecutionPolicy Bypass -File ".\build_tools\validate_windows_install.ps1"
        }
        finally {
            Pop-Location
        }
    }

    "checkout" {
        New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null
        Invoke-RepoPython -WorkingDirectory $TheRockPyTorch -Arguments @(
            ".\pytorch_torch_repo.py",
            "checkout",
            "--gitrepo-origin",
            $RepoRoot,
            "--repo-hashtag",
            "main",
            "--checkout-dir",
            $PyTorchSource,
            "--no-submodules",
            "--no-hipify"
        )

        Initialize-PyTorchSubmodules

        Invoke-RepoPython -WorkingDirectory $TheRockPyTorch -Arguments @(
            ".\pytorch_torch_repo.py",
            "hipify",
            "--checkout-dir",
            $PyTorchSource
        )
    }

    "checkout-vision" {
        Test-RequiredPath -Path $PyTorchSource -Message "PyTorch source checkout not found. Run: pixi run checkout-pytorch"
        New-Item -ItemType Directory -Force -Path $BuildRoot | Out-Null
        $visionCommit = Get-PinnedVisionCommit

        Invoke-RepoPython -WorkingDirectory $TheRockPyTorch -Arguments @(
            ".\pytorch_vision_repo.py",
            "checkout",
            "--checkout-dir",
            $VisionSource,
            "--torch-dir",
            $PyTorchSource,
            "--repo-hashtag",
            $visionCommit
        )
    }

    "build" {
        Test-RequiredPath -Path $PyTorchSource -Message "PyTorch source checkout not found. Run: pixi run checkout-pytorch"
        Clear-ConflictingRocmEnvironment
        Enter-VsDevShell
        New-Item -ItemType Directory -Force -Path $WheelOutput | Out-Null

        $env:PYTORCH_ROCM_ARCH = "gfx1030"
        $env:MAX_JOBS = $BuildJobs
        $env:CMAKE_BUILD_PARALLEL_LEVEL = $BuildJobs

        Invoke-RepoPython -WorkingDirectory $TheRockPyTorch -Arguments @(
            ".\build_prod_wheels.py",
            "build",
            "--install-rocm",
            "--index-url",
            $RocmIndexUrl,
            "--pytorch-rocm-arch",
            "gfx1030",
            "--pytorch-dir",
            $PyTorchSource,
            "--output-dir",
            $WheelOutput,
            "--no-build-pytorch-audio",
            "--no-build-pytorch-vision",
            "--no-build-apex",
            "--no-build-triton",
            "--no-enable-pytorch-flash-attention-windows",
            "--clean"
        )
    }

    "build-vision" {
        Test-RequiredPath -Path $VisionSource -Message "PyTorch Vision source checkout not found. Run: pixi run checkout-vision"
        Clear-ConflictingRocmEnvironment
        Enter-VsDevShell
        Set-TorchVisionCodecEnvironment
        Remove-VisionBuildDirectory
        New-Item -ItemType Directory -Force -Path $WheelOutput | Out-Null

        $env:PYTORCH_ROCM_ARCH = "gfx1030"
        $env:MAX_JOBS = $BuildJobs
        $env:CMAKE_BUILD_PARALLEL_LEVEL = $BuildJobs

        Invoke-RepoPython -WorkingDirectory $TheRockPyTorch -Arguments @(
            ".\build_prod_wheels.py",
            "build",
            "--install-rocm",
            "--index-url",
            $RocmIndexUrl,
            "--pytorch-rocm-arch",
            "gfx1030",
            "--pytorch-vision-dir",
            $VisionSource,
            "--output-dir",
            $WheelOutput,
            "--no-build-pytorch-audio",
            "--build-pytorch-vision",
            "--no-build-apex",
            "--no-build-triton",
            "--no-enable-pytorch-flash-attention-windows",
            "--clean"
        )
    }

    "install-wheel" {
        $wheel = Get-ChildItem -LiteralPath $WheelOutput -Filter "torch-*.whl" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1

        if (-not $wheel) {
            throw "No torch wheel found under $WheelOutput. Run: pixi run build-torch-wheel"
        }

        & python -m pip install --force-reinstall --no-deps $wheel.FullName
    }

    "install-vision-wheel" {
        $wheel = Get-ChildItem -LiteralPath $WheelOutput -Filter "torchvision-*.whl" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1

        if (-not $wheel) {
            throw "No torchvision wheel found under $WheelOutput. Run: pixi run build-vision-wheel"
        }

        & python -m pip install --force-reinstall --no-deps $wheel.FullName
    }

    "smoke" {
        Invoke-PythonOutsideSource -Code @'
import os
import pathlib

import torch

source_torch = pathlib.Path(os.environ["RX6900_SOURCE_TORCH"]).resolve()
torch_file = pathlib.Path(torch.__file__).resolve()
if torch_file == source_torch or source_torch in torch_file.parents:
    raise SystemExit(f"Imported torch from source checkout instead of installed wheel: {torch_file}")

print(torch.__version__)
print(torch_file)
assert torch.cuda.is_available(), "torch.cuda is not available"
print(torch.cuda.get_device_name(0))
x = torch.tensor([0.0, 1.0], device="cuda")
print((x == 0).tolist(), (x != 0).tolist())
torch.cuda.synchronize()
print("sync ok")
'@
    }

    "smoke-vision" {
        Invoke-PythonOutsideSource -Code @'
import os
import pathlib
import tempfile

import torch
import torchvision
from torchvision import ops, transforms
from torchvision.io import read_image
from PIL import Image

source_torch = pathlib.Path(os.environ["RX6900_SOURCE_TORCH"]).resolve()
torch_file = pathlib.Path(torch.__file__).resolve()
if torch_file == source_torch or source_torch in torch_file.parents:
    raise SystemExit(f"Imported torch from source checkout instead of installed wheel: {torch_file}")

print(torch.__version__)
print(torchvision.__version__)
print(torch_file)
print(pathlib.Path(torchvision.__file__).resolve())
assert torch.cuda.is_available(), "torch.cuda is not available"
print(torch.cuda.get_device_name(0))

boxes = torch.tensor(
    [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 11.0, 11.0], [20.0, 20.0, 30.0, 30.0]],
    device="cuda",
)
scores = torch.tensor([0.9, 0.8, 0.7], device="cuda")
keep = ops.nms(boxes, scores, 0.5)
torch.cuda.synchronize()
print("nms", keep.tolist())

image = torch.randint(0, 256, (3, 8, 8), dtype=torch.uint8)
converted = transforms.functional.convert_image_dtype(image, torch.float32)
print("transform", str(converted.dtype), tuple(converted.shape))

with tempfile.TemporaryDirectory() as tmpdir:
    png_path = pathlib.Path(tmpdir) / "smoke.png"
    Image.new("RGB", (4, 4), color=(3, 7, 11)).save(png_path)
    decoded = read_image(str(png_path))
    print("read_image", str(decoded.dtype), tuple(decoded.shape), int(decoded.sum().item()))

print("torchvision ok")
'@
    }

    "probe" {
        Invoke-PythonOutsideSource -Code @'
import gc
import json
import os
import pathlib
import time

import torch
import torch.nn.functional as F

source_torch = pathlib.Path(os.environ["RX6900_SOURCE_TORCH"]).resolve()
torch_file = pathlib.Path(torch.__file__).resolve()
if torch_file == source_torch or source_torch in torch_file.parents:
    raise SystemExit(f"Imported torch from source checkout instead of installed wheel: {torch_file}")

if not torch.cuda.is_available():
    raise SystemExit("torch.cuda is not available")

torch.set_num_threads(min(8, max(1, os.cpu_count() or 1)))

summary = {
    "torch_version": torch.__version__,
    "torch_file": str(torch_file),
    "hip_version": getattr(torch.version, "hip", None),
    "cuda_device": torch.cuda.get_device_name(0),
}

mask_input = torch.tensor([0.0, 1.0], device="cuda")
eq_mask = (mask_input == 0).tolist()
ne_mask = (mask_input != 0).tolist()
if eq_mask != [True, False] or ne_mask != [False, True]:
    raise SystemExit(f"unexpected equality masks: eq={eq_mask} ne={ne_mask}")

norm_input = torch.randn(512, 6, device="cuda", requires_grad=True)
F.normalize(norm_input, dim=-1).sum().backward()
torch.cuda.synchronize()
summary["normalize_backward"] = "ok"

def run_matmul_backward(device: str, n: int, warmup: int, iters: int) -> float:
    a = torch.randn(n, n, device=device, requires_grad=True)
    b = torch.randn(n, n, device=device, requires_grad=True)

    def step() -> None:
        loss = (a @ b).relu().sum()
        loss.backward()
        a.grad = None
        b.grad = None

    for _ in range(warmup):
        step()

    if device == "cuda":
        torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        step()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return iters / elapsed

matrix_n = int(os.environ.get("RX6900_PROBE_MATRIX_N", "1024"))
warmup = int(os.environ.get("RX6900_PROBE_WARMUP", "2"))
iters = int(os.environ.get("RX6900_PROBE_ITERS", "5"))
min_ratio = float(os.environ.get("RX6900_PROBE_MIN_GPU_CPU_RATIO", "1.2"))

cpu_sps = run_matmul_backward("cpu", matrix_n, warmup=1, iters=max(2, min(iters, 3)))
gc.collect()
cuda_sps = run_matmul_backward("cuda", matrix_n, warmup=warmup, iters=iters)
ratio = cuda_sps / cpu_sps if cpu_sps else 0.0

summary.update(
    {
        "matrix_n": matrix_n,
        "cpu_steps_per_sec": round(cpu_sps, 4),
        "cuda_steps_per_sec": round(cuda_sps, 4),
        "cuda_vs_cpu_ratio": round(ratio, 4),
        "min_required_ratio": min_ratio,
    }
)

print(json.dumps(summary, indent=2, sort_keys=True))

if ratio < min_ratio:
    raise SystemExit(
        f"CUDA probe is too slow: ratio={ratio:.3f}, required>={min_ratio:.3f}. "
        "Do not publish this wheel or update downstream environments."
    )
'@
    }
}
