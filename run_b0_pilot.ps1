# B0 pilot (skripsi Iam): ukur throughput HCA Softmax per jumlah env.
# Salinan berparameter dari run_all_experiments_50M.ps1; skrip kating tidak diubah.
# Proses training berjalan terpisah dari sesi ini; log ke run_logs/<run-id>.out.log (di-ignore git).
#
# Contoh:
#   .\run_b0_pilot.ps1 -EnvPath "F:\...\Code Crusader.exe" -NumEnvs 1
#   .\run_b0_pilot.ps1 -EnvPath "F:\...\Code Crusader.exe" -NumEnvs 1 -RunId B0pilot-e1-s1-20261016 -Resume

param(
    [Parameter(Mandatory = $true)][string]$EnvPath,
    [int]$NumEnvs = 1,
    [int]$Seed = 1,
    [string]$RunId = "B0pilot-e$NumEnvs-s$Seed-$(Get-Date -Format yyyyMMdd)",
    [string]$Config = "config/hca/B0_pilot_HCA_Softmax_200k.yaml",
    [string]$MlagentsLearn = "$env:USERPROFILE\miniconda3\envs\mlagents-hca\Scripts\mlagents-learn.exe",
    [switch]$Resume
)

if (-not (Test-Path $EnvPath)) { throw "Build tidak ditemukan: $EnvPath" }
if (-not (Test-Path $MlagentsLearn)) { throw "mlagents-learn tidak ditemukan: $MlagentsLearn" }
# Run-id tidak boleh dipakai ulang kecuali -Resume (aturan AGENTS.md; tanpa --force).
if (-not $Resume -and (Test-Path "results/$RunId")) { throw "results/$RunId sudah ada; pakai run-id baru atau -Resume" }

New-Item -ItemType Directory -Force run_logs | Out-Null
$suffix = if ($Resume) { "-resume-$(Get-Date -Format HHmmss)" } else { "" }
$out = "run_logs/$RunId$suffix.out.log"
$err = "run_logs/$RunId$suffix.err.log"

$learnArgs = @($Config, "--env=`"$EnvPath`"", "--run-id=$RunId", "--no-graphics",
               "--num-envs=$NumEnvs", "--seed=$Seed", "--torch-device=cuda")
if ($Resume) { $learnArgs += "--resume" }

$env:PYTHONIOENCODING = "utf-8"
$p = Start-Process -FilePath $MlagentsLearn -ArgumentList $learnArgs -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $out -RedirectStandardError $err

Write-Output "commit : $(git rev-parse --short HEAD)"
Write-Output "start  : $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Output "PID    : $($p.Id)"
Write-Output "cmd    : $MlagentsLearn $($learnArgs -join ' ')"
Write-Output "log    : $out | $err"
