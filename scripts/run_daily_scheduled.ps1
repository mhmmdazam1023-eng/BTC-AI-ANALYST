# Wrapper for Task Scheduler (2026-08-09).
#
# Task Scheduler invokes this instead of python.exe directly for one reason:
# a bare "python.exe run_daily.py" action has no console attached, so ALL of
# run_daily.py's stdout — the entire daily summary, every stage result — is
# discarded silently, success or failure. The previous scheduled task ran
# this way and its Arguments field was independently broken (a duplicated,
# truncated python.exe path fed to python.exe as the script argument), so it
# never once ran a real pipeline; nothing surfaced either failure because
# there was no log to look at.
#
# One log file per calendar month, appended to (not overwritten), so a
# week's worth of runs stays in one place worth opening, and the file count
# stays bounded over a year of 2x/day runs.

$ErrorActionPreference = 'Continue'

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

$logDir = Join-Path $root 'logs'
if (-not (Test-Path $logDir)) {
    New-Item -ItemType Directory -Path $logDir -Force | Out-Null
}
$logFile = Join-Path $logDir ('run_daily_{0}.log' -f (Get-Date -Format 'yyyy-MM'))
$stamp = Get-Date -Format 'yyyy-MM-dd HH:mm:ss'

Add-Content -Path $logFile -Value ''
Add-Content -Path $logFile -Value "===== SCHEDULED RUN START $stamp ====="

# Windows PowerShell 5.1's `*>>` redirection defaults to UTF-16LE for
# captured child-process output. run_daily.py writes UTF-8 (it calls
# sys.stdout.reconfigure(encoding="utf-8") itself). Without this line,
# PowerShell reinterprets those UTF-8 bytes through the system codepage one
# byte at a time before re-encoding to UTF-16LE, which turns every character
# into "P O R T O F O L I O" — a real bug, caught in the first live test run
# (2026-08-09) by actually reading the resulting log file rather than only
# checking the exit code.
[Console]::OutputEncoding = [System.Text.Encoding]::UTF8

$python = Join-Path $root '.venv\Scripts\python.exe'

& $python 'scripts\run_daily.py' 2>&1 | Out-File -FilePath $logFile -Append -Encoding utf8
$exitCode = $LASTEXITCODE

# Read-only, no API cost — appended regardless of run_daily's own exit code,
# so a failed run still shows where the sample counts stand.
& $python 'scripts\check_sample_readiness.py' 2>&1 | Out-File -FilePath $logFile -Append -Encoding utf8

Add-Content -Path $logFile -Value "===== SCHEDULED RUN END $stamp (run_daily.py exit $exitCode) ====="

exit $exitCode
