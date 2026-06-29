# PyTorch ROCm RX 6900 XT Windows

This fork builds PyTorch ROCm wheels for AMD RX 6900 XT / `gfx1030` on Windows.
It tracks upstream PyTorch `main` and carries the Windows ROCm fixes needed for
this target.

## 1. Clone

```powershell
git clone https://github.com/lgcyaxi/pytorch-rocm-rx6900xt-windows.git
cd pytorch-rocm-rx6900xt-windows
git submodule update --init external/TheRock
```

If the repo is already cloned:

```powershell
git submodule update --init external/TheRock
```

Do not use `--recursive` for `external/TheRock` unless you are building ROCm
itself from TheRock sources. The torch-only wheel path installs ROCm packages
with `--install-rocm` and does not need the large nested ROCm/LLVM submodules.

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
- Pixi

Pixi manages the Python/build helper environment for this repo. Visual Studio,
the Windows SDK, and ATL still need to be installed system-wide.

Create the pixi environment:

```powershell
pixi install
```

Build outputs default to a short path on the same drive, such as
`<repo-drive>\b\rx6900`, to avoid Windows filename-length failures in PyTorch
third-party submodules. Override it when needed:

```powershell
$env:RX6900_BUILD_ROOT = "D:\b\rx6900"
```

Validate from the repository root:

```powershell
pixi run validate-windows
```

The validator should end with `0 FAIL`. If ATL is missing, add this Visual
Studio component:

```powershell
Microsoft.VisualStudio.Component.VC.ATL
```

Developer Mode is optional. When symlink creation is unavailable, the checkout
task falls back to `core.symlinks=false` for PyTorch submodules.

## 3. Prepare TheRock

```powershell
pixi run check-therock-reqs
```

TheRock's Python requirements are locked in `pixi.lock`. Do not run
`fetch_sources.py` for the normal PyTorch wheel path; it is only needed when
building ROCm itself from TheRock source.

## 4. Checkout PyTorch Sources

```powershell
pixi run checkout-pytorch
```

This checkout uses this fork as the PyTorch source, so it includes the RX 6900 XT
Windows ROCm kernel fix. The task initializes PyTorch's required third-party
submodules and initializes the build checkout's `external/TheRock` root
submodule without recursing into its ROCm/LLVM source submodules.

## 5. Build The Torch Wheel

Keep the first build torch-only. It is the shortest path to a working wheel and
avoids optional Windows failures in triton, audio, vision, or flash attention.

```powershell
pixi run build-torch-wheel
```

The build defaults to one job for stability. Set `RX6900_BUILD_JOBS` only if the
machine is stable under parallel C++/HIP compilation.

If the `gfx103X-dgpu` package index is unavailable, use a ROCm/TheRock index
that contains `gfx1030` packages and keep `--pytorch-rocm-arch gfx1030`.

## 6. Install And Smoke Test

```powershell
pixi run install-built-wheel
pixi run smoke-test
pixi run probe-built-wheel
```

Expected on RX 6900 XT:

```text
<torch version>
<installed torch path>
AMD Radeon RX 6900 XT
[True, False] [False, True]
sync ok
```

`probe-built-wheel` runs the equality-mask fix, `torch.nn.functional.normalize`
backward, and a small CPU/GPU matmul-backward timing gate from outside the
source checkout. Do not publish the wheel or point downstream projects at it if
this probe fails.

## 7. Build Torchvision

Build torchvision only after the torch wheel passes the smoke/probe step:

```powershell
pixi run checkout-vision
pixi run build-vision-wheel
pixi run install-vision-wheel
pixi run smoke-vision
```

`checkout-vision` uses PyTorch's pinned torchvision commit from the build
checkout. `smoke-vision` verifies import, CUDA NMS, tensor transforms, and PNG
image IO from outside the source checkout.

## 8. Use The Wheels Elsewhere

Point another pixi environment at the built wheel if you do not want to install
it into this build environment. Include torchvision only after `smoke-vision`
passes:

```toml
[pypi-dependencies]
torch = { path = "<RX6900_BUILD_ROOT>/wheels/<torch-wheel-file>.whl" }
torchvision = { path = "<RX6900_BUILD_ROOT>/wheels/<torchvision-wheel-file>.whl" }
```

## Sync From Upstream

```powershell
git fetch upstream main
git rebase upstream/main
git push --force-with-lease origin main
```

Keep build outputs in the local build root. Do not commit wheels or local
package archives.
