# PyTorch ROCm RX 6900 XT Windows

This fork builds PyTorch ROCm wheels for AMD RX 6900 XT / `gfx1030` on Windows.
The overlay `main` branch and the latest GitHub Release are the stable runtime
line. Upstream PyTorch is merged periodically when we rebuild torch+vision and
publish a new date-tagged release, not on every upstream commit.

## Install from GitHub Releases

If you only need to run torch (or torchvision) on this card, download the
wheels from [Releases](https://github.com/lgcyaxi/pytorch-rocm-rx6900xt-windows/releases).
You do not need Visual Studio, pixi, or a source build.

```powershell
pip install https://github.com/lgcyaxi/pytorch-rocm-rx6900xt-windows/releases/download/v2026.08.28-rocm7.13.0-gfx1030/torch-2.15.0a0+rocm7.13.0-cp312-cp312-win_amd64.whl
pip install https://github.com/lgcyaxi/pytorch-rocm-rx6900xt-windows/releases/download/v2026.08.28-rocm7.13.0-gfx1030/torchvision-0.30.0a0+rocm7.13.0-cp312-cp312-win_amd64.whl
pip install --index-url https://repo.amd.com/rocm/whl/gfx103X-all/ `
    "rocm[libraries]==7.13.0" rocm-sdk-core==7.13.0 rocm-sdk-libraries-gfx103x-all==7.13.0
```

Python 3.12 on 64-bit Windows. Do not install `rocm-sdk-devel`. After a local
wheel build, maintainers publish with `pixi run publish-release`.

The rest of this README is the source-build path used to produce those wheels.

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
itself from TheRock sources. The torch-only wheel path uses the pinned TheRock
ROCm 7.13.0 `gfx103X-all` wheels from pixi and does not need the large nested
ROCm/LLVM submodules. Do not install the official Windows HIP SDK and do not
set `HIP_PATH` or `ROCM_PATH`.

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

This also downloads the pinned TheRock ROCm 7.13.0 SDK for `gfx103X-all`
(`rocm`, `rocm-sdk-core`, `rocm-sdk-devel`, `rocm-sdk-libraries-gfx103x-all`)
from `https://repo.amd.com/rocm/whl/gfx103X-all/`. Expect about 4.7 GiB of
wheels; `rocm-sdk-devel` is most of that.

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
pixi run check-rocm-sdk
```

TheRock's Python requirements and the ROCm 7.13.0 SDK are locked in
`pixi.lock`. Do not run `fetch_sources.py` for the normal PyTorch wheel path;
it is only needed when building ROCm itself from TheRock source.

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

The build uses the pixi-locked ROCm 7.13.0 SDK already in this environment. It
does not call TheRock `--install-rocm` and does not float on nightly indexes.
The build defaults to 12 jobs, which fits this i9-12900K (24 threads) and 96 GB
RAM without filling the machine during HIP compiles. Override with
`RX6900_BUILD_JOBS` if a run OOMs or you want to leave more headroom. Set
`RX6900_BUILD_CLEAN=0` to resume an interrupted ninja tree without `--clean`.

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

After the probe passes, run the reusable CPU vs GPU baseline:

```powershell
pixi run bench
pixi run bench -- --quick
pixi run bench -- --json-out agent_space/bench.json
```

`bench` times FP32/FP16 GEMM, a 3x3 conv, a short ViT-like SDPA+MLP block,
ViT `F.linear` shapes, and a DRAM copy. JSON rows include `pct_paper_peak`
against 23.04 TFLOPS FP32 / 46.08 TFLOPS FP16 / 512 GB/s. CPU comparisons use
FP32 only. GEMM 4096 is GPU-only. It is a scoreboard, not a publish gate.

Measured on this RX 6900 XT with `2.15.0a0+rocm7.13.0` (rocBLAS Tensile, NCHW):

| Op | GPU FP32 | % peak | GPU FP16 | % peak |
|---|---|---|---|---|
| GEMM 4096 burst | 22.9 TFLOPS | 99% | 37.9 TFLOPS | 82% |
| GEMM 4096 sustained | 19.5 TFLOPS | 85% | 33.9 TFLOPS | 74% |
| GEMM 2048 | 20.9 TFLOPS | 91% | 30.3 TFLOPS | 66% |
| Conv 3x3 NCHW | 19.3-21.9 TFLOPS | 84-95% | 22-28 TFLOPS | 49-61% |
| DRAM copy | 431 GB/s | 84% | 432 GB/s | 84% |
| ViT linear 8x49x768->3072 | 7.5 TFLOPS | 32% | 11.9 TFLOPS | 26% |
| ViT-like seq 49, batch 8 | 0.58 ms | launch | 0.43 ms | launch |

What is already at the wall:

1. Large FP32 GEMM and 3x3 NCHW conv hit paper peak on a short burst. Longer
   runs drop with clocks/power, not with a worse kernel.
2. hipBLASLt has no gfx1030 images in `gfx103X-all` (only gfx1100). Forcing it
   crashes TunableOp. Keep `PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0` if you tune.
3. rocBLAS TunableOp, `cudnn.benchmark`, `channels_last`, FP16 accumulate, and
   a bigger `HIPBLAS_WORKSPACE_CONFIG` did not beat the defaults. `channels_last`
   is about half speed. Do not enable them.
4. `torch.compile` / Inductor cannot autotune GEMM here: no Triton on this
   Windows ROCm wheel, and the GPU arch is rejected for `max_autotune_gemm`.
5. Flash Attention / AOTriton / CK SDPA are not built for gfx1030. Math SDPA
   is the path. Seq 49 is launch-bound; FA can wait.

What can still move:

1. FP16 large GEMM/conv sits at about 80% / 60% of the 2x packed peak. The
   shipped rocBLAS library for gfx1030 is almost all `fallback_*` Tensile
   images, and PyTorch's ROCm GEMM always uses FP32 accumulate. Closing that
   needs better gfx1030 Tensile/hipBLASLt images or a custom packed-FMA kernel,
   not another runtime switch.
2. Batch up ViT work. `32x256x768->3072` FP16 reached about 37 TFLOPS; `8x49`
   cannot. HIP graphs did not help the short block.
3. Keep the GPU in a high-performance power limit if you care about sustained
   4096+ GEMM. Burst already shows the kernel is at the FP32 ceiling.

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

## 8. Export The Runtime Bundle

After a good probe, export the wheels to a destination directory. That
directory **is** the runtime home. Do not export and then copy again unless
you want a second, edited pixi project.

```powershell
pixi run export-runtime-bundle
pixi run export-runtime-bundle -- D:\rocm-runtime
```

Default destination is `F:\rocm-build\packages`. Override with the argument
above or `$env:RX6900_RUNTIME_BUNDLE`. Then either:

1. Use it as a pixi project: `cd <dest>; pixi install`
2. Point another repo at `<dest>\dependencies\*.whl` (Ware-care AMD does this)
3. Optionally copy `<dest>` and edit that copy's `pixi.toml`

The bundle is runtime-only: `torch-2.15.0a0+rocm7.13.0`, matching
`torchvision-0.30.0a0+rocm7.13.0`, plus TheRock ROCm 7.13.0 `rocm`,
`rocm-sdk-core`, and `rocm-sdk-libraries-gfx103x-all`. No `rocm-sdk-devel`.

The `pixi.toml` / README templates live in `runtime-bundle/`.

## Sync From Upstream

Stay on this overlay `main` between releases. Merge upstream only when you
intend to rebuild both wheels and publish a new `vYYYY.MM.DD-rocm7.13.0-gfx1030`
tag. Merge, do not rebase: the overlay and pixi/TheRock files live on this
branch; rebasing only the overlay commit onto upstream drops them.

```powershell
git fetch upstream main
git merge --no-ff upstream/main
git push origin main
```

TheRock is a separate fork (`lgcyaxi/therock-rocm-rx6900xt-windows`). Fast-forward
it from `ROCm/TheRock` `main`, then bump `external/TheRock` here.

Keep build outputs in the local build root. Do not commit wheels or local
package archives. Publish them with `pixi run publish-release`.
