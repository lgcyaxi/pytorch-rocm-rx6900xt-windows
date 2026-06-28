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
- Pixi

Pixi manages the Python/build helper environment for this repo. Visual Studio,
the Windows SDK, and ATL still need to be installed system-wide.

Create the pixi environment:

```powershell
pixi install
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
Windows ROCm kernel fix.

## 5. Build The Torch Wheel

Keep the first build torch-only. It is the shortest path to a working wheel and
avoids optional Windows failures in triton, audio, vision, or flash attention.

```powershell
pixi run build-torch-wheel
```

If the `gfx103X-dgpu` package index is unavailable, use a ROCm/TheRock index
that contains `gfx1030` packages and keep `--pytorch-rocm-arch gfx1030`.

## 6. Install And Smoke Test

```powershell
pixi run install-built-wheel
pixi run smoke-test
```

Expected on RX 6900 XT:

```text
<torch version>
AMD Radeon RX 6900 XT
[True, False] [False, True]
sync ok
```

## 7. Use The Wheel Elsewhere

Point another pixi environment at the built wheel if you do not want to install
it into this build environment:

```toml
[pypi-dependencies]
torch = { path = "build/wheels/<torch-wheel-file>.whl" }
```

## Sync From Upstream

```powershell
git fetch upstream main
git rebase upstream/main
git push --force-with-lease origin main
```

Keep build outputs under `build\` or another ignored local directory. Do not
commit wheels or local package archives.
