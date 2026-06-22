@echo off
cd /d %~dp0

REM ── venv ───────────────────────────────────────────────────────────────
if not exist venv (echo Creating venv... & python -m venv venv || (echo [ERROR] Failed to create venv. & pause & exit /b 1))
set "VENV_PYTHON=venv\Scripts\python.exe"

REM ── Install ────────────────────────────────────────────────────────────
echo Installing ROCm PyTorch...
"%VENV_PYTHON%" -m pip install torch torchvision torchaudio --index-url https://rocm.nightlies.amd.com/v2/gfx110X-all/ || (echo [ERROR] ROCm PyTorch install failed. & pause & exit /b 1)

echo Installing requirements...
"%VENV_PYTHON%" -m pip install -r requirements.txt || (echo [ERROR] pip install failed. & pause & exit /b 1)

echo Installing UI dependencies...
cd training-ui && call npm install || (echo [ERROR] npm install failed. & cd .. & pause & exit /b 1)

echo.
echo Installation complete!
pause
