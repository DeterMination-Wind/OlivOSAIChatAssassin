param(
    [string]$TargetDir = "dist"
)

$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path -Parent $PSScriptRoot
$OutDir = Join-Path $RepoRoot $TargetDir
$BinaryName = "napcat-aichat-assassin-rs.exe"
$ReleaseBinary = Join-Path $RepoRoot "target\release\$BinaryName"
$PackageDir = Join-Path $OutDir "napcat-aichat-assassin-rs"

Write-Host "Building release binary..."
Push-Location $RepoRoot
try {
    cargo build --release
} finally {
    Pop-Location
}

if (!(Test-Path $ReleaseBinary)) {
    throw "Release binary not found: $ReleaseBinary"
}

New-Item -ItemType Directory -Force -Path $OutDir | Out-Null
if (Test-Path $PackageDir) {
    Remove-Item -Recurse -Force $PackageDir
}
New-Item -ItemType Directory -Force -Path $PackageDir | Out-Null
New-Item -ItemType Directory -Force -Path (Join-Path $PackageDir "data\Knowledge") | Out-Null

Copy-Item $ReleaseBinary (Join-Path $PackageDir $BinaryName)
Copy-Item (Join-Path $RepoRoot "README.md") (Join-Path $PackageDir "README.md")
Copy-Item (Join-Path $RepoRoot "LICENSE") (Join-Path $PackageDir "LICENSE")

if (Test-Path (Join-Path $RepoRoot "data\config.json")) {
    Copy-Item (Join-Path $RepoRoot "data\config.json") (Join-Path $PackageDir "data\config.json")
}
if (Test-Path (Join-Path $RepoRoot "data\memory.json")) {
    Copy-Item (Join-Path $RepoRoot "data\memory.json") (Join-Path $PackageDir "data\memory.json")
}

$ZipPath = Join-Path $OutDir "napcat-aichat-assassin-rs-windows-x64.zip"
if (Test-Path $ZipPath) {
    Remove-Item -Force $ZipPath
}
Compress-Archive -Path (Join-Path $PackageDir "*") -DestinationPath $ZipPath

Write-Host "Release package created:"
Write-Host "  $ZipPath"
