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

## 2. Install Windows Build Tools

Install:

- Visual Studio 2022 Build Tools
- MSVC C++ tools
- Windows SDK
- C++ ATL
- CMake
- Ninja
- Python 3.12
- Git

Validate the machine:

```powershell
cd external\TheRock
.\build_tools\validate_windows_install.ps1
cd ..\..
```

If ATL is missing, add the Visual Studio component:

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
python .\build_tools\fetch_sources.py
```

## 4. Checkout Build Sources

Run from `external\TheRock\external-builds\pytorch`:

```powershell
cd external-builds\pytorch

python .\pytorch_torch_repo.py checkout `
  --gitrepo-origin ..\..\..\.. `
  --repo-hashtag main `
  --checkout-dir ..\..\..\..\build\pytorch-src

python .\pytorch_audio_repo.py checkout `
  --checkout-dir ..\..\..\..\build\audio-src

python .\pytorch_vision_repo.py checkout `
  --checkout-dir ..\..\..\..\build\vision-src
```

## 5. Build The Wheel

```powershell
$env:PYTORCH_ROCM_ARCH = "gfx1030"
$env:MAX_JOBS = "1"
$env:CMAKE_BUILD_PARALLEL_LEVEL = "1"

python .\build_prod_wheels.py build `
  --install-rocm `
  --index-url https://rocm.nightlies.amd.com/v2/gfx103X-dgpu/ `
  --pytorch-rocm-arch gfx1030 `
  --pytorch-dir ..\..\..\..\build\pytorch-src `
  --pytorch-audio-dir ..\..\..\..\build\audio-src `
  --pytorch-vision-dir ..\..\..\..\build\vision-src `
  --output-dir ..\..\..\..\build\wheels
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
