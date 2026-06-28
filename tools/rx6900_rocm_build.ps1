param(
    [Parameter(Mandatory = $true)]
    [ValidateSet("validate", "checkout", "build", "install-wheel", "smoke")]
    [string]$Action
)

$ErrorActionPreference = "Stop"

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$TheRockRoot = Join-Path $RepoRoot "external\TheRock"
$TheRockPyTorch = Join-Path $TheRockRoot "external-builds\pytorch"
$BuildRoot = Join-Path $RepoRoot "build"
$PyTorchSource = Join-Path $BuildRoot "pytorch-src"
$WheelOutput = Join-Path $BuildRoot "wheels"
$RocmIndexUrl = "https://rocm.nightlies.amd.com/v2/gfx103X-dgpu/"

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
            $PyTorchSource
        )
    }

    "build" {
        Test-RequiredPath -Path $PyTorchSource -Message "PyTorch source checkout not found. Run: pixi run checkout-pytorch"
        Clear-ConflictingRocmEnvironment
        Enter-VsDevShell
        New-Item -ItemType Directory -Force -Path $WheelOutput | Out-Null

        $env:PYTORCH_ROCM_ARCH = "gfx1030"
        $env:MAX_JOBS = "1"
        $env:CMAKE_BUILD_PARALLEL_LEVEL = "1"

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

    "install-wheel" {
        $wheel = Get-ChildItem -LiteralPath $WheelOutput -Filter "torch-*.whl" -ErrorAction SilentlyContinue |
            Sort-Object LastWriteTime -Descending |
            Select-Object -First 1

        if (-not $wheel) {
            throw "No torch wheel found under $WheelOutput. Run: pixi run build-torch-wheel"
        }

        & python -m pip install --force-reinstall --no-deps $wheel.FullName
    }

    "smoke" {
        & python -c "import torch; x=torch.tensor([0.,1.],device='cuda'); print(torch.__version__); print(torch.cuda.get_device_name(0)); print((x==0).tolist(), (x!=0).tolist()); torch.cuda.synchronize(); print('sync ok')"
    }
}
