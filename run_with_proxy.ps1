param(
    [string]$ProxyUrl = "http://127.0.0.1:7890",
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$PythonArgs
)

if (-not $PythonArgs -or $PythonArgs.Count -eq 0) {
    $PythonArgs = @("main.py")
}

$env:HTTP_PROXY = $ProxyUrl
$env:HTTPS_PROXY = $ProxyUrl
$env:ALL_PROXY = $ProxyUrl

$env:EMBEDDING_LOCAL_ONLY = "1"

& ".\.venv\Scripts\python.exe" @PythonArgs
