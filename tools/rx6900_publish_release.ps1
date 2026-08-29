param(
    [string]$Tag,
    [string]$Target,
    [string]$Repo = "lgcyaxi/pytorch-rocm-rx6900xt-windows",
    [switch]$RequireVision
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DefaultBuildRoot = Join-Path ([System.IO.Path]::GetPathRoot($RepoRoot)) "b\rx6900"
$BuildRoot = if ($env:RX6900_BUILD_ROOT) { $env:RX6900_BUILD_ROOT } else { $DefaultBuildRoot }
$WheelOutput = Join-Path $BuildRoot "wheels"
$Git = Join-Path $RepoRoot ".pixi\envs\default\Library\mingw64\bin\git.exe"
if (-not (Test-Path -LiteralPath $Git)) {
    $Git = "git"
}

if (-not $Target) {
    $Target = (& $Git -C $RepoRoot rev-parse HEAD).Trim()
}
if (-not $Tag) {
    $Tag = "v{0}-rocm7.13.0-gfx1030" -f (Get-Date -Format "yyyy.MM.dd")
}

$torch = Get-ChildItem -LiteralPath $WheelOutput -Filter "torch-*.whl" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -notlike "torchvision-*" } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if (-not $torch) {
    throw "No torch wheel under $WheelOutput. Run: pixi run build-torch-wheel"
}

$assets = @($torch.FullName)
$vision = Get-ChildItem -LiteralPath $WheelOutput -Filter "torchvision-*.whl" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -notlike "*0.29.0a0+rocm7.13.0a20260421*" } |
    Sort-Object LastWriteTime -Descending |
    Select-Object -First 1
if ($vision) {
    $assets += $vision.FullName
} elseif ($RequireVision) {
    throw "No matching torchvision wheel under $WheelOutput. Run: pixi run build-vision-wheel"
}

$visionInstall = ""
if ($vision) {
    $visionInstall = @"

pip install https://github.com/$Repo/releases/download/$Tag/$($vision.Name)
"@
}

$notes = @"
Prebuilt RX 6900 XT / gfx1030 Windows wheels. Source: ``$Target``.

Python 3.12. ROCm runtime comes from AMD, not this release:

``````powershell
pip install https://github.com/$Repo/releases/download/$Tag/$($torch.Name)$visionInstall
pip install --index-url https://repo.amd.com/rocm/whl/gfx103X-all/ ``
    "rocm[libraries]==7.13.0" rocm-sdk-core==7.13.0 rocm-sdk-libraries-gfx103x-all==7.13.0
``````

Do not install ``rocm-sdk-devel``.
"@

$ErrorActionPreference = "Continue"
gh release view $Tag --repo $Repo 2>$null | Out-Null
$viewCode = $LASTEXITCODE
$ErrorActionPreference = "Stop"
if ($viewCode -eq 0) {
    Write-Host "Updating existing release $Tag"
    gh release upload $Tag @assets --repo $Repo --clobber
} else {
    gh release create $Tag @assets --repo $Repo --target $Target --title $Tag --notes $notes
}

Write-Host "Release: https://github.com/$Repo/releases/tag/$Tag"
foreach ($path in $assets) {
    Write-Host "  $(Split-Path $path -Leaf)"
}
