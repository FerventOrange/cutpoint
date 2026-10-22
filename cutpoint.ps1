# Run Cutpoint with the project venv: .\cutpoint <analyze|clips|assemble|...> ...
# (needs the .venv created by uv sync; run it once in the repo)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
& "$here\.venv\Scripts\python.exe" -m cutpoint @args
exit $LASTEXITCODE
