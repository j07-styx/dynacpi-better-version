param(
 [Parameter(Mandatory=$true)]
 [ValidateSet('protein_strict_20261113','protein_strict_20261114','compound_scaffold_20261115','double_cold_20261116')]
 [string]$Scenario,
 [Parameter(Mandatory=$true)]
 [ValidateSet(2931,2932)]
 [int]$Seed
)
$ErrorActionPreference='Stop'
$root='C:\Users\J0124\DynaCPI-Backend'
$up=Join-Path $root 'matched_perceivercpi_20261110\upstream'
$dir=Join-Path $root ('generalization_suite_20261110\'+$Scenario)
$target=Join-Path $dir ('perceiver_official_seed'+$Seed+'_30ep')
$python='C:\Users\J0124\dynacpi_v4\.venv\Scripts\python.exe'
if(Test-Path $target){throw ('Refusing to overwrite or resume already-created run '+$target)}
$spec=Get-Content (Join-Path $dir 'SPLIT_LOCKED.json') -Raw|ConvertFrom-Json
if($spec.scenario -ne $Scenario -or !$spec.test_labels_not_exported -or !$spec.selection_locked_before_test){
 throw 'Preregistered scenario or test separation invalid'
}
if(@($spec.training_seeds) -join ',' -ne '2931,2932' -or $spec.perceiver_epochs -ne 30){
 throw 'Training seed list or epochs differ from predeclared protocol'
}
if((git -C $up rev-parse HEAD).Trim() -ne '755f076ef51099a90332fd1ec8437f94e3750c0b'){
 throw 'Official upstream commit mismatch'
}
$modified=(git -C $up diff --name-only -- '*.py' |Out-String).Trim()
if($modified){throw ('Upstream Python code changed: '+$modified)}
foreach($filename in @('train.csv','validation.csv','validation_proxy_test.csv')){
 $expected=$spec.input_csv_sha256.$filename
 $current=(Get-FileHash (Join-Path $dir $filename) -Algorithm SHA256).Hash.ToLowerInvariant()
 if($current -ne $expected){throw ('Frozen training/validation CSV changed '+$filename)}
}
if($spec.input_csv_sha256.'validation.csv' -ne $spec.input_csv_sha256.'validation_proxy_test.csv'){
 throw 'Test proxy is not byte-identical validation'
}
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='2'
$env:PYTHONHASHSEED=[string]$Seed
$env:PYTHONWARNINGS='ignore::UserWarning'
$env:TOKENIZERS_PARALLELISM='false'
Set-Location $up
$cmd=@('-u','train.py',
 '--data_path',"$dir\train.csv",
 '--separate_val_path',"$dir\validation.csv",
 '--separate_test_path',"$dir\validation_proxy_test.csv",
 '--metric','mse',
 '--dataset_type','regression',
 '--save_dir',$target,
 '--target_columns','label',
 '--epochs','30',
 '--ensemble_size','1',
 '--num_folds','1',
 '--batch_size','128',
 '--num_workers','0',
 '--cache_cutoff','100000',
 '--aggregation','mean',
 '--dropout','0.1',
 '--pytorch_seed',[string]$Seed,
 '--seed',[string]$Seed,
 '--save_preds')
Write-Output ('FROZEN_SCENARIO='+$Scenario+' SEED='+$Seed+' START='+(Get-Date))
Write-Output ('NO_TEST_LABELS_AVAILABLE_TO_UPSTREAM='+$true)
& $python @cmd
$exitcode=$LASTEXITCODE
Write-Output ('OFFICIAL_TRAIN_EXIT='+$exitcode+' FINISHED_AT='+(Get-Date))
if($exitcode -ne 0){exit $exitcode}
Write-Output 'OFFICIAL_VALIDATION_SELECTED_CHECKPOINT_COMPLETE'
