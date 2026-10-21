# Run Cutpoint with the project venv: .\cutpoint <analyze|clips|assemble|...> ...
# (needs the editable install: uv pip install --python .venv\Scripts\python.exe -e .)
$here = Split-Path -Parent $MyInvocation.MyCommand.Path
& "$here\.venv\Scripts\python.exe" -m cutpoint @args
exit $LASTEXITCODE
