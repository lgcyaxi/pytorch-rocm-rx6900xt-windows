# PyTorch ROCm RX 6900 XT Windows

This fork builds PyTorch ROCm wheels for AMD RX 6900 XT / `gfx1030` on Windows.
It tracks upstream PyTorch `main` and carries the Windows ROCm fixes needed for
this target.

## 1. Clone

```powershell
git clone --recurse-submodules https://github.com/lgcyaxi/pytorch-rocm-rx6900xt-windows.git
cd pytorch-rocm-rx6900xt-windows
```

If the repo is already cloned:

```powershell
git submodule update --init --recursive external/TheRock
```

## 2. Prepare Visual Studio

Install:

- Visual Studio 2022 Build Tools
- MSVC C++ tools
- Windows SDK
- C++ ATL
- CMake
- Ninja
- Python 3.12
- Git

Load the Visual Studio x64 build environment:

```powershell
$vs = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\2022\BuildTools\Common7\Tools\Launch-VsDevShell.ps1"
& $vs -Arch amd64 -HostArch amd64
```

Validate from the repository root:

```powershell
cd external\TheRock
powershell -ExecutionPolicy Bypass -File .\build_tools\validate_windows_install.ps1
cd ..\..
```

The validator should end with `0 FAIL`. If ATL is missing, add this Visual
Studio component:

```powershell
Microsoft.VisualStudio.Component.VC.ATL
```

## 3. Prepare TheRock

```powershell
cd external\TheRock
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cd ..\..
```

`fetch_sources.py` is only needed when building ROCm itself from TheRock source.
For PyTorch wheels using `--install-rocm`, the TheRock Python requirements are
the required preparation step.

## 4. Checkout PyTorch Sources

```powershell
cd external\TheRock\external-builds\pytorch
$buildRoot = "..\..\..\..\build"
New-Item -ItemType Directory -Force $buildRoot | Out-Null

python .\pytorch_torch_repo.py checkout `
  --gitrepo-origin ..\..\..\.. `
  --repo-hashtag main `
  --checkout-dir "$buildRoot\pytorch-src"
```

This checkout uses this fork as the PyTorch source, so it includes the RX 6900 XT
Windows ROCm kernel fix.

## 5. Build The Torch Wheel

Keep the first build torch-only. It is the shortest path to a working wheel and
avoids optional Windows failures in triton, audio, vision, or flash attention.

```powershell
$buildRoot = "..\..\..\..\build"
$env:PYTORCH_ROCM_ARCH = "gfx1030"
$env:MAX_JOBS = "1"
$env:CMAKE_BUILD_PARALLEL_LEVEL = "1"

python .\build_prod_wheels.py build `
  --install-rocm `
  --index-url https://rocm.nightlies.amd.com/v2/gfx103X-dgpu/ `
  --pytorch-rocm-arch gfx1030 `
  --pytorch-dir "$buildRoot\pytorch-src" `
  --output-dir "$buildRoot\wheels" `
  --no-build-pytorch-audio `
  --no-build-pytorch-vision `
  --no-build-apex `
  --no-build-triton `
  --no-enable-pytorch-flash-attention-windows `
  --clean
```

If the `gfx103X-dgpu` package index is unavailable, use a ROCm/TheRock index
that contains `gfx1030` packages and keep `--pytorch-rocm-arch gfx1030`.

## 6. Use With Pixi

Create or edit a pixi environment and point `torch` at the built wheel:

```toml
[pypi-dependencies]
torch = { path = "build/wheels/<torch-wheel-file>.whl" }
```

Then lock and run your environment:

```powershell
pixi lock
pixi run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```

## 7. Smoke Test

```powershell
pixi run python -c "import torch; x=torch.tensor([0.,1.],device='cuda'); print(torch.cuda.get_device_name(0)); print((x==0).tolist(), (x!=0).tolist()); torch.cuda.synchronize(); print('sync ok')"
```

Expected on RX 6900 XT:

```text
AMD Radeon RX 6900 XT
[True, False] [False, True]
sync ok
```

## Sync From Upstream

```powershell
git fetch upstream main
git rebase upstream/main
git push --force-with-lease origin main
```

Keep build outputs under `build\` or another ignored local directory. Do not
commit wheels or local package archives.
