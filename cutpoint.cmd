@echo off
rem Run Cutpoint with the project venv: cutpoint ^<analyze^|clips^|assemble^|...^> ...
rem (needs the editable install: uv pip install --python .venv\Scripts\python.exe -e .)
"%~dp0.venv\Scripts\python.exe" -m cutpoint %*
exit /b %ERRORLEVEL%
