@echo off
set HIP_VISIBLE_DEVICES=1
cd /d "%~dp0"
echo Starting Anima Training UI...
call npm start
echo.
echo Application exited (check for errors above).
pause
