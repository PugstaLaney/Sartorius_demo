# Quick-start launcher. Right-click this file → "Run with PowerShell".
# Opens two terminal windows (backend + frontend) and your browser.

$VenvPython = "C:\Users\palla\venvs\sartorius-cell\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    Write-Host "venv not found at $VenvPython" -ForegroundColor Red
    Read-Host "Press Enter to close"
    exit 1
}

Write-Host "Starting backend (http://localhost:8000) ..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd '$PSScriptRoot\backend'; & '$VenvPython' -m uvicorn main:app --port 8000"
)

Write-Host "Starting frontend (http://localhost:8080) ..." -ForegroundColor Cyan
Start-Process powershell -ArgumentList @(
    "-NoExit",
    "-Command",
    "cd '$PSScriptRoot\frontend'; & '$VenvPython' -m http.server 8080"
)

Write-Host ""
Write-Host "Waiting 8 seconds for the model to load..." -ForegroundColor Yellow
Start-Sleep -Seconds 8

Write-Host "Opening browser..." -ForegroundColor Green
Start-Process "http://localhost:8080"

Write-Host ""
Write-Host "Both servers are running in the two PowerShell windows that opened."
Write-Host "To stop: close those two windows (or press Ctrl+C in each)."
Write-Host ""
Read-Host "Press Enter to close this launcher window"
