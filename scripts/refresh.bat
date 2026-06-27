@echo off
REM ─────────────────────────────────────────────────────────────────────────
REM jse-radar daily refresh script
REM
REM This script is called by Windows Task Scheduler every weekday morning.
REM It runs the full pipeline using the jse-radar conda environment.
REM
REM Why call Python directly via full path?
REM   conda activate inside a batch script requires conda to be fully
REM   initialised for cmd.exe, which depends on how conda was installed.
REM   Calling the environment's python.exe directly is simpler and
REM   guaranteed to use the right interpreter with all its packages —
REM   no activation step needed at all.
REM ─────────────────────────────────────────────────────────────────────────

REM ── Paths — update these if your install locations differ ─────────────────
set REPO=D:\jse-radar
set PYTHON=D:\miniforge3_main\envs\jse-radar\python.exe
set LOGFILE=%REPO%\logs\scheduler.log

REM ── Write run header to log ───────────────────────────────────────────────
echo. >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"
echo jse-radar scheduled refresh: %DATE% %TIME% >> "%LOGFILE%"
echo ============================================================ >> "%LOGFILE%"

REM ── Run the pipeline ──────────────────────────────────────────────────────
cd /d "%REPO%"
"%PYTHON%" scripts\run_pipeline.py --start 2000-01-01 >> "%LOGFILE%" 2>&1

REM ── Write completion to log ───────────────────────────────────────────────
echo Refresh completed: %DATE% %TIME% >> "%LOGFILE%"
echo. >> "%LOGFILE%"