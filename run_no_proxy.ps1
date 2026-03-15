param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PythonArgs
)

if (-not $PythonArgs -or $PythonArgs.Count -eq 0) {
    $PythonArgs = @("main.py")
}

Remove-Item Env:HTTP_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:HTTPS_PROXY -ErrorAction SilentlyContinue
Remove-Item Env:ALL_PROXY -ErrorAction SilentlyContinue

$env:EMBEDDING_LOCAL_ONLY = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:HF_HUB_OFFLINE = "1"

& ".\.venv\Scripts\python.exe" @PythonArgs
