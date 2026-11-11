# Proactively score two intermediate KIBA/Metz cohorts after BOTH original
# validation-selected checkpoints are ready. The LAST original cohort,
# KIBA double-cold, remains reserved for the existing all-six final gate:
# avoids a last-checkpoint race with that evaluator.
# No TEST labels accessed while waiting. Abort on any incomplete once-only run.
$ErrorActionPreference='Stop'
$root='C:\Users\J0124\DynaCPI-Backend'
$suite=Join-Path $root 'external_benchmarks_20261110\locked_external_protocol_v1'
$queue=Join-Path $suite 'OFFICIAL_PERCEIVER_KIBA_METZ_QUEUE_20261110.log'
$finalGate=Join-Path $suite 'ALL_SIX_EXTERNAL_SAME_SPLIT_TESTS_ONCE'
$once=Join-Path $suite 'INDIVIDUAL_EXTERNAL_SAME_SPLIT_TESTS_ONCE_20261111'
$log=Join-Path $suite 'INCREMENTAL_EXTERNAL_READY_GATE_20261111.log'
$py='C:\Users\J0124\dynacpi_v4\.venv\Scripts\python.exe'
$evaluator=Join-Path $root 'evaluate_external_ready_scenario_once_20261111.py'
$bootstrap=Join-Path $root 'paired_bootstrap_external_kiba_metz_20261110.py'
if(Test-Path $log){throw 'Existing incremental gate log. Refuse duplicate watcher.'}
if(!(Test-Path $queue)){throw 'Original model training queue missing'}
if(!(Test-Path $evaluator) -or !(Test-Path $bootstrap)){throw 'Frozen evaluator unavailable'}
function Note([string]$message){
 $t='['+(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')+'] '+$message
 Add-Content -LiteralPath $log -Value $t -Encoding UTF8
 Write-Output $t
}
function Ready([string]$ds,[string]$sc){
 foreach($seed in @(2931,2932)){
  $run=Join-Path $suite ($ds+'\'+$sc+'\perceiver_official_seed'+$seed+'_30ep')
  $lf=Join-Path $run 'verbose.log'
  if(!(Test-Path (Join-Path $run 'fold_0\model_0\model.pt')) -or
     !(Test-Path (Join-Path $run 'args.json')) -or !(Test-Path $lf)){return $false}
  $text=Get-Content -LiteralPath $lf -Raw
  if(!($text.Contains('Epoch 29') -and $text.Contains('Overall test mse') -and
        $text.Contains('Elapsed time'))){return $false}
 }
 return $true
}
Note 'INCREMENTAL_WAIT_GATE_START NO_TEST_ACCESS'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='2'
$tasks=@(
 @{dataset='kiba';scenario='compound_cold'},
 @{dataset='metz';scenario='double_cold'}
)
foreach($t in $tasks){
 $ds=[string]$t.dataset
 $sc=[string]$t.scenario
 $name=$ds+'_'+$sc
 $dest=Join-Path $once $name
 $deadline=(Get-Date).AddHours(18)
 while(!(Ready $ds $sc)){
   if(Test-Path $finalGate){
     Note ('FINAL_AGGREGATE_STARTED; EARLY_GATE_EXIT_NO_RESCORING '+$name)
     exit 0
   }
   if((Get-Date) -gt $deadline){
     Note ('TIMEOUT_NO_TEST '+$name)
     throw 'Waiting for frozen original checkpoints timed out'
   }
   $contents=Get-Content -LiteralPath $queue -Raw
   if($contents -match 'FAILED |INCOMPLETE |MEMORY_GUARD_TIMEOUT'){
     Note ('ORIGINAL_QUEUE_FAILED_ABORT_NO_TEST '+$name)
     throw 'Original training queue failed; no test access allowed'
   }
   if($contents.Contains('ALL_12_ORIGINAL_EXTERNAL_MODELS_TRAINED_AND_VALIDATION_SELECTED')){
     Note ('ALL_TWELVE_READY_USE_EXISTING_FINAL_GATE '+$name)
     exit 0
   }
   Note ('WAITING_TWO_ORIGINAL_VALIDATION_SELECTED_MODELS '+$name)
   Start-Sleep -Seconds 120
 }
 if(Test-Path $finalGate){
   Note ('AGGREGATE_ALREADY_STARTED_ABORT_INDIVIDUAL '+$name)
   exit 0
 }
 if(Test-Path $dest){
   $results=Join-Path $dest 'TEST_RESULTS.json'
   if(!(Test-Path $results)){
     Note ('PARTIAL_INDIVIDUAL_TEST_FOUND_ABORT '+$name)
     throw 'Existing partial sealed test - never restart'
   }
   Note ('ALREADY_ONCE_SCORED_NO_RESCORING '+$name)
   continue
 }
 $pre=Join-Path $suite ('INCREMENTAL_'+$name+'_PREFLIGHT_20261111.log')
 $scored=Join-Path $suite ('INCREMENTAL_'+$name+'_TEST_ONCE_20261111.log')
 $bootlog=Join-Path $suite ('INCREMENTAL_'+$name+'_BOOTSTRAP_20261111.log')
 if((Test-Path $pre) -or (Test-Path $scored) -or (Test-Path $bootlog)){
   throw ('Existing partial incremental log; refuse duplicate '+$name)
 }
 Note ('RUN_VALIDATION_ONLY_PREFLIGHT '+$name)
 & $py -u $evaluator --dataset $ds --scenario $sc --mode preflight *> $pre
 if($LASTEXITCODE -ne 0){
   Note ('PREFLIGHT_ABORT_NO_TEST '+$name)
   throw 'No-test preflight refused'
 }
 $pt=Get-Content $pre -Raw
 if(!$pt.Contains('"both_baselines_ready": true')){
   Note ('PREFLIGHT_MISSING_TWO_SEEDS_ABORT '+$name)
   throw 'Frozen two-seed preflight marker absent'
 }
 if(Test-Path $finalGate){Note ('FINAL_GATE_STARTED_AFTER_PREFLIGHT '+$name);exit 0}
 Note ('EVALUATE_IMMUTABLE_HELDOUT_ONCE '+$name)
 & $py -u $evaluator --dataset $ds --scenario $sc --mode evaluate *> $scored
 if($LASTEXITCODE -ne 0){
    Note ('ONCE_TEST_FAILED_PRESERVE_PARTIAL_NO_RETRY '+$name)
    throw 'Once-only test failed; preserve evidence'
 }
 $result=Join-Path $dest 'TEST_RESULTS.json'
 if(!(Test-Path $result)){throw ('Missing once-only test result '+$name)}
 Note ('ONCE_TEST_COMPLETE SHA256='+((Get-FileHash $result -Algorithm SHA256).Hash)+' '+$name)
 Note ('COMPUTE_PREDECLARED_PAIRED_BOOTSTRAP '+$name)
 & $py -u $bootstrap --dataset $ds --scenario $sc --individual *> $bootlog
 if($LASTEXITCODE -ne 0){
   Note ('BOOTSTRAP_FAILED_TEST_UNCHANGED '+$name)
   throw 'Bootstrap failure; test remains frozen and untouched'
 }
 Note ('INDIVIDUAL_COMPLETE_WITH_UNCERTAINTY '+$name)
}
Note 'DONE_TWO_INTERMEDIATE_SCENARIOS; FINAL_KIBA_DOUBLE_COLD_RESERVED_FOR_ALL_SIX_GATE'
