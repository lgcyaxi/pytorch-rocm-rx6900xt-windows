# RX 6900 XT runtime pixi bundle

This directory is the **exported runtime home**, not a staging folder you
have to copy again. After a good torch wheel:

```powershell
pixi run export-runtime-bundle
pixi run export-runtime-bundle -- D:\rocm-runtime
```

Default destination is `F:\rocm-build\packages`. Override with the path
argument or `$env:RX6900_RUNTIME_BUNDLE`.

Then pick one:

1. **Use this folder as a pixi project**
   ```powershell
   cd <dest>
   pixi install
   ```
2. **Point another repo at the wheels** (preferred for Ware-care). Pin
   `torch` / `rocm*` path deps to `<dest>\dependencies\`.
3. **Optional copy.** Copy `<dest>` and edit that copy's `pixi.toml` only
   when the new project needs a different dependency set.

The bundle is runtime-only:

- `torch-2.15.0a0+rocm7.13.0`
- `torchvision-0.30.0a0+rocm7.13.0`
- TheRock ROCm `7.13.0` (`rocm`, `rocm-sdk-core`, `rocm-sdk-libraries-gfx103x-all`)
- The usual torch Python deps as local wheels

It does **not** include `rocm-sdk-devel`. Devel is only for building the fork.

## Peak settings

Leave these at the PyTorch defaults. Sweeps on the RX 6900 XT showed they
lose time or crash:

- Do not set `torch.backends.cudnn.benchmark = True` for the 3x3 NCHW shapes
  we run. The MIOpen heuristic is already at about 95% of FP32 peak.
- Do not convert convs to `channels_last`. gfx1030 MIOpen is NCHW; NHWC was
  about half speed. The CK grouped-conv library is also missing for gfx1030.
- Do not enable TunableOp without `PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0`.
  The `gfx103X-all` hipBLASLt tree has gfx1100 images only and `getAllAlgos`
  raises. rocBLAS-only tuning did not beat the default heuristic.
- Do not expect `torch.compile` / Inductor GEMM autotune. This wheel has no
  Triton, and gfx1030 is not in hipBLASLt's supported-arch list.
- `F.scaled_dot_product_attention` uses the math backend. That is expected.

Do this instead:

- Keep activations NCHW and weights contiguous.
- Batch work when you want TFLOPS. Short radar windows and small linears
  are launch-bound on this card.
- Treat long 4096+ FP32 GEMM at about 19 TFLOPS as a power/clock limit. A
  5-iteration burst reaches about 23 TFLOPS (paper peak).
