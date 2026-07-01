@echo off
REM Double-click this file to launch the full demo (FastAPI backend + WPF console).
REM This wrapper exists because Windows opens .ps1 files in Notepad by default,
REM so we invoke PowerShell explicitly here.
REM
REM %~dp0 expands to the folder this .cmd file lives in, so the launcher works
REM no matter where you double-click from.

powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0run_wpf.ps1"
