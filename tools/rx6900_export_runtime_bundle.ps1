param(
    [string]$Destination = $env:RX6900_RUNTIME_BUNDLE
)

$ErrorActionPreference = "Stop"

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$DefaultBuildRoot = Join-Path ([System.IO.Path]::GetPathRoot($RepoRoot)) "b\rx6900"
$BuildRoot = if ($env:RX6900_BUILD_ROOT) { $env:RX6900_BUILD_ROOT } else { $DefaultBuildRoot }
$WheelOutput = Join-Path $BuildRoot "wheels"
$RocmIndexUrl = "https://repo.amd.com/rocm/whl/gfx103X-all/"

if (-not $Destination) {
    $Destination = "F:\rocm-build\packages"
}

$Dependencies = Join-Path $Destination "dependencies"
$Template = Join-Path $RepoRoot "runtime-bundle\pixi.toml"
$TemplateReadme = Join-Path $RepoRoot "runtime-bundle\README.md"

if (-not (Test-Path -LiteralPath $Template)) {
    throw "Missing runtime bundle template: $Template"
}

New-Item -ItemType Directory -Force -Path $Dependencies | Out-Null

$torchWheel = Get-ChildItem -LiteralPath $WheelOutput -Filter "torch-2.15.0a0+rocm7.13.0-cp312-cp312-win_amd64.whl" -ErrorAction SilentlyContinue |
    Select-Object -First 1
if (-not $torchWheel) {
    $torchWheel = Get-ChildItem -LiteralPath $WheelOutput -Filter "torch-*.whl" -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -notlike "torchvision-*" } |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
}
if (-not $torchWheel) {
    throw "No torch wheel found under $WheelOutput. Run: pixi run build-torch-wheel"
}

Copy-Item -LiteralPath $torchWheel.FullName -Destination (Join-Path $Dependencies $torchWheel.Name) -Force
Write-Host "Copied $($torchWheel.Name)"

& python -m pip download --dest $Dependencies --no-deps --only-binary :all: --index-url $RocmIndexUrl "rocm-sdk-core==7.13.0" "rocm-sdk-libraries-gfx103x-all==7.13.0"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to download ROCm 7.13.0 library wheels from $RocmIndexUrl"
}

# The AMD index publishes rocm 7.13.0 as an sdist, not a wheel.
& python -m pip download --dest $Dependencies --no-deps --index-url $RocmIndexUrl "rocm==7.13.0"
if ($LASTEXITCODE -ne 0) {
    throw "Failed to download the rocm 7.13.0 sdist from $RocmIndexUrl"
}

$rocmSdist = Get-ChildItem -LiteralPath $Dependencies -Filter "rocm-7.13.0.tar.gz" -ErrorAction SilentlyContinue
if ($rocmSdist) {
    & python -m pip wheel --no-deps --wheel-dir $Dependencies $rocmSdist.FullName
    if ($LASTEXITCODE -ne 0) {
        throw "Failed to build rocm-7.13.0 from sdist"
    }
    Remove-Item -LiteralPath $rocmSdist.FullName -Force
}

$obsolete = @(
    "torch-2.14.0a0+rocm7.13.0a20260421-cp312-cp312-win_amd64.whl",
    "torchvision-0.29.0a0+rocm7.13.0a20260421-cp312-cp312-win_amd64.whl",
    "rocm-7.13.0a20260421-py3-none-any.whl",
    "rocm_sdk_core-7.13.0a20260421-py3-none-win_amd64.whl",
    "rocm_sdk_libraries_gfx103x_dgpu-7.13.0a20260421-py3-none-win_amd64.whl"
)
foreach ($name in $obsolete) {
    $path = Join-Path $Dependencies $name
    if (Test-Path -LiteralPath $path) {
        Remove-Item -LiteralPath $path -Force
        Write-Host "Removed obsolete $name"
    }
}

Copy-Item -LiteralPath $Template -Destination (Join-Path $Destination "pixi.toml") -Force
Copy-Item -LiteralPath $TemplateReadme -Destination (Join-Path $Destination "README.md") -Force
Write-Host "Exported runtime home: $Destination"
Write-Host "Use it: cd $Destination; pixi install"
Write-Host "Or pin another repo at $Dependencies"
