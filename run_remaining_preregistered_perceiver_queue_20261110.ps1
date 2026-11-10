$ErrorActionPreference='Stop'
$root='C:\Users\J0124\DynaCPI-Backend'
$runroot=Join-Path $root 'generalization_suite_20261110'
$runner=Join-Path $root 'run_preregistered_official_perceiver_generic_20261110.ps1'
$logfile=Join-Path $runroot 'OFFICIAL_TRAINING_QUEUE_20261110.log'
if(Test-Path $logfile){throw 'A training queue log already exists; never launch a duplicate queue'}
function LogLine([string]$message){
  $line='['+(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')+'] '+$message
  Add-Content -LiteralPath $logfile -Value $line -Encoding UTF8
  Write-Output $line
}
function IsComplete([string]$directory,[int]$seed){
  $log=Join-Path $directory 'verbose.log'
  $ckpt=Join-Path $directory 'fold_0\model_0\model.pt'
  if(!(Test-Path $log) -or !(Test-Path $ckpt)){return $false}
  $t=Get-Content $log -Raw
  if(!($t.Contains('Epoch 29') -and $t.Contains('Overall test mse') -and $t.Contains('Elapsed time'))){
    return $false
  }
  $args=Get-Content (Join-Path $directory 'args.json') -Raw|ConvertFrom-Json
  if($args.epochs -ne 30 -or $args.pytorch_seed -ne $seed){return $false}
  return $true
}
LogLine 'START_QUEUED_FIXED_OFFICIAL_TRAINING; ONLY TRAIN+VAL DATA; NO SEALED TEST EVALUATION'
$first=Join-Path $runroot 'protein_strict_20261113'
$deadline=(Get-Date).AddHours(8)
while($true){
  if((Get-Date) -gt $deadline){throw 'Exceeded 8h guard while waiting for strict A baseline jobs'}
  $running=@(Get-CimInstance Win32_Process |Where-Object{
    $_.Name -eq 'python.exe' -and $_.CommandLine -match 'train\.py' -and
    $_.CommandLine.Contains('protein_strict_20261113\perceiver_official_seed293')})
  if($running.Count -eq 0){break}
  LogLine ('WAIT_STRICT_A_EXISTING_RUNNING_PROCESSES='+$running.Count)
  Start-Sleep -Seconds 60
}
foreach($seed in @(2931,2932)){
  $path=Join-Path $first ('perceiver_official_seed'+$seed+'_30ep')
  if(!(IsComplete $path $seed)){
    throw ('Existing strict A seed '+$seed+' did not finish; abort queued follow-up instead of contaminating comparison')
  }
}
LogLine 'STRICT_A_BOTH_SEEDS_COMPLETED_AND_VALIDATION_SELECTED; SEALED HOLDOUT STILL UNTOUCHED'
$jobs=@(
 @{scenario='protein_strict_20261114';seed=2931},
 @{scenario='protein_strict_20261114';seed=2932},
 @{scenario='compound_scaffold_20261115';seed=2931},
 @{scenario='compound_scaffold_20261115';seed=2932},
 @{scenario='double_cold_20261116';seed=2931},
 @{scenario='double_cold_20261116';seed=2932}
)
foreach($job in $jobs){
  $scenario=[string]$job.scenario
  $seed=[int]$job.seed
  $target=Join-Path $runroot ($scenario+'\perceiver_official_seed'+$seed+'_30ep')
  if(Test-Path $target){throw ('Queued official run directory already exists: '+$target)}
  $output=Join-Path $runroot ($scenario+'_seed'+$seed+'_runner.stdout.log')
  $errorlog=Join-Path $runroot ($scenario+'_seed'+$seed+'_runner.stderr.log')
  if(Test-Path $output){throw ('Queued runner output already exists: '+$output)}
  if(Test-Path $errorlog){throw ('Queued runner error output already exists: '+$errorlog)}
  LogLine ('START_OFFICIAL_SAME_SPLIT_TRAIN scenario='+$scenario+' seed='+$seed)
  $cli=@('-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass','-File',$runner,
          '-Scenario',$scenario,'-Seed',[string]$seed)
  $p=Start-Process -FilePath 'powershell.exe' -ArgumentList $cli -PassThru -Wait -RedirectStandardOutput $output -RedirectStandardError $errorlog
  if($p.ExitCode -ne 0){
     LogLine ('TRAIN_FAILED scenario='+$scenario+' seed='+$seed+' exit='+$p.ExitCode)
     throw 'Official training failed; stop pipeline. No test data opened.'
  }
  if(!(IsComplete $target $seed)){
     throw ('Official training exit 0 but checkpoint/log is incomplete: '+$target)
  }
  LogLine ('COMPLETED_OFFICIAL_VALIDATION_SELECTED_TRAIN scenario='+$scenario+' seed='+$seed+
           ' checkpoint_SHA256='+((Get-FileHash (Join-Path $target 'fold_0\model_0\model.pt') -Algorithm SHA256).Hash))
}
LogLine 'ALL_EIGHT_NEW_OFFICIAL_30_EPOCH_RUNS_COMPLETE; ONLY TRAIN+VAL USED; ALL_TESTS_STILL_SEALED'
