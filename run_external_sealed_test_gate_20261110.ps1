$ErrorActionPreference='Stop'
$root='C:\Users\J0124\DynaCPI-Backend'
$suite=Join-Path $root 'external_benchmarks_20261110\locked_external_protocol_v1'
$queue=Join-Path $suite 'OFFICIAL_PERCEIVER_KIBA_METZ_QUEUE_20261110.log'
$gate=Join-Path $suite 'SEALED_SIX_EXTERNAL_EVALUATION_GATE_20261110.log'
$final=Join-Path $suite 'ALL_SIX_EXTERNAL_SAME_SPLIT_TESTS_ONCE\COMPLETE_SIX_RESULTS.json'
$script=Join-Path $root 'evaluate_external_kiba_metz_all_six_once_20261110.py'
$py='C:\Users\J0124\dynacpi_v4\.venv\Scripts\python.exe'
if(Test-Path $gate){throw 'Existing external test gate; refuse duplicate'}
if(Test-Path $final){throw 'All six test already scored; refuse duplicate'}
if(!(Test-Path $queue)){throw 'External model training queue is missing'}
function Note([string]$message){
 $t='['+(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')+'] '+$message
 Add-Content -LiteralPath $gate -Value $t -Encoding UTF8
 Write-Output $t
}
Note 'GATE_WAITING_FOR_ALL_12_VALIDATION_SELECTED_ORIGINAL_EXTERNAL_MODELS; NO_TEST_ACCESS'
$deadline=(Get-Date).AddHours(24)
while($true){
 $s=Get-Content $queue -Raw -ErrorAction SilentlyContinue
 if($s.Contains('ALL_12_ORIGINAL_EXTERNAL_MODELS_TRAINED_AND_VALIDATION_SELECTED')){break}
 if($s -match 'FAILED |INCOMPLETE |MEMORY_GUARD_TIMEOUT'){
   Note 'ORIGINAL_EXTERNAL_QUEUE_FAILED_ABORT_NO_TEST'
   throw 'Original baseline failed; sealed datasets remain untouched'
 }
 if((Get-Date) -gt $deadline){
   Note 'EXTERNAL_QUEUE_TIMED_OUT_ABORT_TEST'
   throw 'External official model training exceeded gate deadline'
 }
 $p=@(Get-CimInstance Win32_Process|Where-Object{
   $_.Name -eq 'powershell.exe' -and
   $_.CommandLine -match 'run_remaining_external_perceiver_queue_20261110'
 })
 if($p.Count -eq 0){
   Note 'OFFICIAL_QUEUE_PROCESS_MISSING_ABORT_TEST'
   throw 'Original queue stopped without completion'
 }
 Note 'WAITING_FOR_ALL_SIX_COLD_GENERALIZATION_EXTERNAL_BASELINE_PAIRS'
 Start-Sleep -Seconds 120
}
Note 'ALL_12_CHECKPOINTS_COMPLETED; CHECKING_SOURCE_AND_SPLITS_WITHOUT_TEST'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='2'
$preflight=Join-Path $suite 'SIX_EXTERNAL_FINAL_PREFLIGHT.log'
$valReplay=Join-Path $suite 'SIX_EXTERNAL_FINAL_VALIDATION_REPLAY.log'
$evaluation=Join-Path $suite 'SIX_EXTERNAL_MATCHED_TEST_ONCE.log'
if((Test-Path $preflight) -or (Test-Path $valReplay) -or (Test-Path $evaluation)){
 throw 'Existing external eval log; no overwrite'
}
& $py -u $script --mode preflight *> $preflight
if($LASTEXITCODE -ne 0){
 Note ('PRE_FLIGHT_ERROR_EXIT='+$LASTEXITCODE)
 throw 'Official full train/val preflight failed; no test accessed'
}
$preflightText=Get-Content $preflight -Raw
if(!$preflightText.Contains('"all_12_ready": true')){
 Note 'NOT_ALL_12_OFFICIAL_VALIDATION_CHECKPOINTS_VERIFIED_ABORT'
 throw 'External source/preflight gate refused to open tests'
}
Note 'PRE_FLIGHT_PASS; REPLAYING_FROZEN_DYNACPI_VALIDATIONS_BEFORE_TEST'
& $py -u $script --mode validate-only *> $valReplay
if($LASTEXITCODE -ne 0){
 Note ('FROZEN_VALIDATION_REPLAY_ERROR='+$LASTEXITCODE)
 throw 'Frozen DynaCPI replay failed; no test'
}
$valText=Get-Content $valReplay -Raw
if(!$valText.Contains('PASS_ALL_SIX_EXTERNAL_FROZEN_VALIDATION_REPLAYS_NO_TEST_SCORING')){
 Note 'MISSING_ALL_SIX_VALIDATION_REPLAY_PASS_ABORT'
 throw 'Expected validated checkpoint replay marker absent'
}
Note 'ALL_CHECKPOINTS_SELECTED_ON_VALIDATION; OPENING_SIX_TESTS_ONCE'
& $py -u $script --mode evaluate *> $evaluation
$evalcode=$LASTEXITCODE
if($evalcode -ne 0){
 Note ('EVALUATOR_FAILED_EXIT='+$evalcode+'; PRESERVE_PARTIAL_TEST_OUTPUTS')
 throw 'Evaluation script failed. Do NOT edit/tune/re-run on exposed tests.'
}
if(!(Test-Path $final)){
 Note 'EVALUATOR_ZERO_EXIT_BUT_MISSING_FINAL_RESULTS'
 throw 'No completed six-scenario test result output'
}
Note ('ALL_SIX_EXTERNAL_ORIGINAL_VS_DYNACPI_TESTS_COMPLETE_AND_LOCKED SHA256='+((Get-FileHash $final -Algorithm SHA256).Hash))
