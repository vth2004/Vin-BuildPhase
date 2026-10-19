$ErrorActionPreference = 'Stop'

$bashScript = Join-Path $PSScriptRoot 'deploy-vps.sh'
$gitBash = 'C:\Program Files\Git\bin\bash.exe'

if (-not (Test-Path -LiteralPath $bashScript)) {
    throw "Deployment script not found: $bashScript"
}

if (-not (Test-Path -LiteralPath $gitBash)) {
    throw 'Git Bash was not found. Install Git for Windows, then run this script again.'
}

Write-Host 'Starting Coolify deployment through Git Bash...' -ForegroundColor Cyan
$env:PUSH_APP_ENV = 'true'
& $gitBash $bashScript

if ($LASTEXITCODE -ne 0) {
    throw "Deployment failed (exit code $LASTEXITCODE)."
}
