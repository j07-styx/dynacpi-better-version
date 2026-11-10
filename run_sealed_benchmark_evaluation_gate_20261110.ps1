$ErrorActionPreference='Stop'
$root='C:\Users\J0124\DynaCPI-Backend'
$suite=Join-Path $root 'generalization_suite_20261110'
$queuelog=Join-Path $suite 'OFFICIAL_TRAINING_QUEUE_20261110.log'
$guard=Join-Path $suite 'SEALED_BENCHMARK_EVALUATION_GATE_20261110.log'
$final=Join-Path $suite 'ONE_TIME_GENERALIZATION_ALL_SCENARIOS\ALL_FOUR_BLIND_TEST_RESULTS.json'
$py='C:\Users\J0124\dynacpi_v4\.venv\Scripts\python.exe'
$evaluator=Join-Path $root 'evaluate_all_four_preregistered_davis_once_20261110.py'
if(Test-Path $guard){throw 'Gate already exists: refusing duplicate evaluator'}
if(Test-Path $final){throw 'All four frozen test results already completed: refusing duplicate evaluator'}
if(!(Test-Path $queuelog)){throw 'Upstream official training queue not launched'}
function Note([string]$msg){
    $stamp='['+(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')+'] '+$msg
    Add-Content -LiteralPath $guard -Value $stamp -Encoding UTF8
    Write-Output $stamp
}
Note 'GATE_START; ONLY_CHECKING_QUEUE_LOG; NO_TEST_LABELS_READ'
$deadline=(Get-Date).AddHours(9)
$successmarker='ALL_EIGHT_NEW_OFFICIAL_30_EPOCH_RUNS_COMPLETE'
while($true){
    $txt=Get-Content -LiteralPath $queuelog -Raw -ErrorAction SilentlyContinue
    if($txt.Contains($successmarker)){break}
    if($txt -match 'TRAIN_FAILED|Exceeded 8h guard'){
        Note 'QUEUE_FAILED: ABORT_BEFORE_OPENING_ANY_TEST'
        throw 'Official training queue failed; preserved sealed test set'
    }
    if((Get-Date) -gt $deadline){
        Note 'GATE_TIMEOUT: ABORT_BEFORE_TEST'
        throw 'Training jobs did not finish before gate timeout'
    }
    $queueProcesses=@(Get-CimInstance Win32_Process|Where-Object{
      $_.Name -eq 'powershell.exe' -and
      $_.CommandLine -match 'run_remaining_preregistered_perceiver_queue_20261110'
    })
    if($queueProcesses.Count -eq 0){
        Note 'QUEUE_PROCESS_MISSING_BEFORE_SUCCESS: ABORT'
        throw 'Training queue terminated before completion marker'
    }
    Note 'WAITING_FOR_ALL_FOUR_PREDECLARED_OFFICIAL_TWO_SEED_MODELS_TO_COMPLETE'
    Start-Sleep -Seconds 120
}
Note 'ALL_EIGHT_OFFICIAL_TRAININGS_COMPLETED; PRE_FLIGHT_NO_TEST'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='2'
$checkLog=Join-Path $suite 'ALL_FOUR_BENCHMARK_PREFLIGHT.log'
if(Test-Path $checkLog){throw 'Preflight log exists: refusing overwrite'}
& $py -u $evaluator --mode preflight *> $checkLog
if($LASTEXITCODE -ne 0){
    Note ('PREFLIGHT_FAILED exit='+$LASTEXITCODE+'; NO_TEST_ACCESS')
    throw 'Final original-model checkpoint preflight failed'
}
$check=(Get-Content $checkLog -Raw)
if(!$check.Contains('"all_eligible": true')){
    Note 'NOT_ALL_ELIGIBLE: ABORT_SEALED_TEST'
    throw 'One or more original or DynaCPI checkpoints not frozen'
}
Note 'ALL_CHECKPOINTS_FROZEN; STARTING_PREREGISTERED_SEALED_TEST_SUITE_ONCE'
$reportlog=Join-Path $suite 'FOUR_SCENARIOS_ONCE_EVALUATOR_STDOUT.log'
if(Test-Path $reportlog){throw 'Existing test evaluator log: refuse duplicate'}
& $py -u $evaluator --mode evaluate *> $reportlog
$code=$LASTEXITCODE
if($code -ne 0){
   Note ('EVALUATOR_FAILED exit='+$code+'; PRESERVE_ALL_PARTIAL_TEST_OUTPUTS_NO_RERUN')
   throw ('Sealed test evaluator ended with exit '+$code)
}
if(!(Test-Path $final)){
   Note 'EVALUATOR_EXITED_ZERO_BUT_NO_FINAL_METRIC_FILE'
   throw 'No final results despite zero exit'
}
Note ('ALL_FOUR_FROZEN_TESTS_SCORED_COMPLETE SHA256='+((Get-FileHash $final -Algorithm SHA256).Hash))
