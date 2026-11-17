@echo off
cd /d "%~dp0"
".venv\Scripts\python.exe" "src\run_experiment.py" --execute
if errorlevel 1 echo Execution blocked or failed. Check OSF registration and local logs.
pause
