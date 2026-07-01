# Launch the full WPF demo: FastAPI backend + WPF console.
#
# This is what "Launch Sartorius Demo.cmd" calls. It can also be run directly:
#     .\run_wpf.ps1
#
# Behavior:
#   1. Start the FastAPI backend in a separate PowerShell window
#   2. Poll http://localhost:8000/health until the model is loaded (up to 60s)
#   3. Launch the WPF app via `dotnet run`
#   4. Exit this launcher window (backend keeps running in its own window)

$ErrorActionPreference = "Stop"

$VenvPython = "C:\Users\palla\venvs\sartorius-cell\Scripts\python.exe"
$BackendDir = Join-Path $PSScriptRoot "backend"
$WpfDir     = Join-Path $PSScriptRoot "frontend_wpf"

# ---- Sanity checks ---------------------------------------------------------
if (-not (Test-Path $VenvPython)) {
    Write-Host "Python venv not found at $VenvPython" -ForegroundColor Red
    Write-Host "Recreate it with: py -3.11 -m venv C:\Users\palla\venvs\sartorius-cell"
    Read-Host "Press Enter to close"
    exit 1
}

if (-not (Get-Command dotnet -ErrorAction SilentlyContinue)) {
    Write-Host ".NET SDK not found on PATH." -ForegroundColor Red
    Write-Host "Install from: https://dotnet.microsoft.com/en-us/download/dotnet/8.0"
    Read-Host "Press Enter to close"
    exit 1
}

# ---- 1. Start the backend in a new window ---------------------------------
Write-Host "Starting FastAPI backend on http://localhost:8000 ..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd '$BackendDir'; & '$VenvPython' -m uvicorn main:app --port 8000"
) | Out-Null

# ---- 2. Wait for the backend to come up ----------------------------------
Write-Host "Waiting for backend to load the model (this can take 10-30s on first run)..." -ForegroundColor Yellow

$deadline = (Get-Date).AddSeconds(60)
$ready = $false
while ((Get-Date) -lt $deadline) {
    try {
        $r = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
        if ($r.StatusCode -eq 200) {
            $ready = $true
            break
        }
    } catch {
        # Backend not ready yet; keep polling.
    }
    Start-Sleep -Milliseconds 500
}

if (-not $ready) {
    Write-Host "Backend did not become ready within 60 seconds." -ForegroundColor Red
    Write-Host "Check the backend window for errors. WPF app will still launch but will show 'backend offline'."
} else {
    Write-Host "Backend is ready." -ForegroundColor Green
}

# ---- 3. Launch the WPF app -----------------------------------------------
Write-Host "Launching WPF console..." -ForegroundColor Cyan
Push-Location $WpfDir
try {
    # dotnet run blocks until the window closes. We launch it in a new process
    # so this launcher can exit immediately. The WPF window becomes the main
    # UI; the backend window keeps running for it.
    Start-Process dotnet -ArgumentList "run", "--project", "`"$WpfDir`"" -WindowStyle Hidden | Out-Null
} finally {
    Pop-Location
}

Write-Host ""
Write-Host "Demo running:" -ForegroundColor Green
Write-Host "  - Backend window:  visible PowerShell window running uvicorn"
Write-Host "  - WPF window:      will appear in ~5-15 seconds"
Write-Host ""
Write-Host "To stop everything:"
Write-Host "  - Close the WPF window"
Write-Host "  - Close (or Ctrl+C) the backend PowerShell window"
Write-Host ""
Start-Sleep -Seconds 3
