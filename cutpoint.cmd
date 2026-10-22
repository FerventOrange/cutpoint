@echo off
rem Run Cutpoint with the project venv: cutpoint ^<analyze^|clips^|assemble^|...^> ...
rem (needs the .venv created by uv sync; run it once in the repo)
"%~dp0.venv\Scripts\python.exe" -m cutpoint %*
exit /b %ERRORLEVEL%
