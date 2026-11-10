$ErrorActionPreference='Stop'
$root='C:\Users\J0124\DynaCPI-Backend'
$suite=Join-Path $root 'generalization_suite_20261110\protein_strict_20261113'
$up=Join-Path $root 'matched_perceivercpi_20261110\upstream'
$py='C:\Users\J0124\dynacpi_v4\.venv\Scripts\python.exe'
$dest=Join-Path $suite 'perceiver_official_seed2931_30ep'
if(Test-Path $dest){throw "Run dir already exists: $dest"}
$lock=Get-Content (Join-Path $suite 'SPLIT_LOCKED.json') -Raw |ConvertFrom-Json
if($lock.scenario -ne 'protein_strict_20261113' -or !$lock.test_labels_not_exported -or $lock.train_pairs -ne 20332 -or $lock.validation_pairs -ne 5236){throw 'Prerequisite split lock invalid'}
$src=(git -C $up rev-parse HEAD).Trim()
if($src -ne '755f076ef51099a90332fd1ec8437f94e3750c0b'){throw 'Upstream SHA changed'}
$code=(git -C $up diff --name-only -- '*.py'|Out-String).Trim()
if($code){throw "Official source modified: $code"}
foreach($f in @('train.csv','validation.csv','validation_proxy_test.csv')){
 $hash=(Get-FileHash (Join-Path $suite $f) -Algorithm SHA256).Hash.ToLowerInvariant()
 if($hash -ne $lock.input_csv_sha256.$f){throw "Frozen CSV changed: $f"}
}
if($lock.input_csv_sha256.'validation.csv' -ne $lock.input_csv_sha256.'validation_proxy_test.csv'){throw 'Validation proxy mismatch'}
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='2'
$env:PYTHONWARNINGS='ignore::UserWarning'
$env:PYTHONHASHSEED='2931'
Set-Location $up
$args=@('-u','train.py',
'--data_path',"$suite\train.csv",
'--separate_val_path',"$suite\validation.csv",
'--separate_test_path',"$suite\validation_proxy_test.csv",
'--metric','mse','--dataset_type','regression',
'--save_dir',$dest,'--target_columns','label',
'--epochs','30','--ensemble_size','1','--num_folds','1',
'--batch_size','128','--num_workers','0','--cache_cutoff','100000',
'--aggregation','mean','--dropout','0.1','--pytorch_seed','2931','--seed','2931','--save_preds')
Write-Output ('EXECUTING_NEW_STRICT_PROTEIN_MATCHED_OFFICIAL_BASELINE='+$dest)
Write-Output ('START_TIME='+(Get-Date))
& $py @args
Write-Output ('STRICT_PROTEIN_OFFICIAL_EXIT='+$LASTEXITCODE)
if($LASTEXITCODE -ne 0){exit $LASTEXITCODE}
Write-Output 'STRICT_PROTEIN_OFFICIAL_FULL_30EPOCHS_COMPLETE'
