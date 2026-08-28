param(
    [string]$Tag = "v2.15.0a0-rocm7.13.0-gfx1030",
    [string]$Target = "0b0b07ae21318fd45b4525662373b5df5dc7ae46",
    [string]$Repo = "lgcyaxi/pytorch-rocm-rx6900xt-windows"
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DefaultBuildRoot = Join-Path ([System.IO.Path]::GetPathRoot($RepoRoot)) "b\rx6900"
$BuildRoot = if ($env:RX6900_BUILD_ROOT) { $env:RX6900_BUILD_ROOT } else { $DefaultBuildRoot }
$WheelOutput = Join-Path $BuildRoot "wheels"

$torch = Get-ChildItem -LiteralPath $WheelOutput -Filter "torch-2.15.0a0+rocm7.13.0-cp312-cp312-win_amd64.whl" -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $torch) {
    throw "Missing torch 2.15 wheel under $WheelOutput. Run: pixi run build-torch-wheel"
}

$assets = @($torch.FullName)
$vision = Get-ChildItem -LiteralPath $WheelOutput -Filter "torchvision-*-cp312-cp312-win_amd64.whl" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -notlike "*0.29.0a0+rocm7.13.0a20260421*" } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($vision) {
    $assets += $vision.FullName
}

$notes = @"
Prebuilt RX 6900 XT / gfx1030 Windows wheels. Source: ``$Target``.

Install with Python 3.12 (ROCm runtime from AMD, not this release):

``````powershell
pip install https://github.com/$Repo/releases/download/$Tag/$($torch.Name)
pip install --index-url https://repo.amd.com/rocm/whl/gfx103X-all/ ``
    "rocm[libraries]==7.13.0" rocm-sdk-core==7.13.0 rocm-sdk-libraries-gfx103x-all==7.13.0
``````

Do not install ``rocm-sdk-devel``. That package is build-only.

Torchvision is attached when a wheel matching this torch exists.
"@

$existing = gh release view $Tag --repo $Repo 2>$null
if ($LASTEXITCODE -eq 0) {
    Write-Host "Updating existing release $Tag"
    gh release upload $Tag @assets --repo $Repo --clobber
} else {
    gh release create $Tag @assets --repo $Repo --target $Target --title $Tag --notes $notes
}

Write-Host "Release: https://github.com/$Repo/releases/tag/$Tag"
foreach ($path in $assets) {
    Write-Host "  $(Split-Path $path -Leaf)"
}
