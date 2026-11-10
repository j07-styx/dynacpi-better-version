$ErrorActionPreference='Stop'
$root='C:\Users\J0124\DynaCPI-Backend\matched_perceivercpi_20261110'
$up=Join-Path $root 'upstream'
$dest=Join-Path $root 'official_matched_v4_seed2932_30ep'
$py='C:\Users\J0124\dynacpi_v4\.venv\Scripts\python.exe'
if(Test-Path $dest){throw "Refusing to re-run or overwrite $dest"}
if((git -C $up rev-parse HEAD).Trim() -ne '755f076ef51099a90332fd1ec8437f94e3750c0b'){
    throw 'Official PerceiverCPI source checkout differs from preregistered commit'
}
$srcModified=(git -C $up diff --name-only -- '*.py' |Out-String).Trim()
if($srcModified){throw "Modified original source $srcModified"}
$j=Get-Content "$root\data_v2\SPLIT_EXPORT.json" -Raw |ConvertFrom-Json
if($j.n_training_pairs -ne 20536 -or $j.n_validation_pairs -ne 4012 -or !$j.test_labels_not_exported){
    throw 'Frozen train/validation exports invalid'
}
foreach($file in @('train.csv','val.csv','val_proxy_test.csv')){
    $expected=$j.csv_sha256.$file
    $actual=(Get-FileHash "$root\data_v2\$file" -Algorithm SHA256).Hash.ToLowerInvariant()
    if($actual -ne $expected){throw "Modified input export: $file"}
}
if((Get-FileHash "$root\data_v2\val.csv" -Algorithm SHA256).Hash -ne (Get-FileHash "$root\data_v2\val_proxy_test.csv" -Algorithm SHA256).Hash){
    throw 'Validation-proxy test is NOT a byte-identical validation copy'
}
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='2'
$env:PYTHONHASHSEED='2932'
$env:PYTHONWARNINGS='ignore::UserWarning'
$env:TOKENIZERS_PARALLELISM='false'
$cfg=@{
 experiment='seed-parity second official PerceiverCPI model'
 purpose='two-seed official ensemble comparison against fixed v5 two-seed ensemble'
 seed=2932
 train_entries=302
 validation_entries=59
 heldout_entries=81
 test_labels_never_supplied_to_upstream_training=$true
 upstream_git_sha='755f076ef51099a90332fd1ec8437f94e3750c0b'
 max_epochs=30
 batch_size=128
 optimizer='official LAMB default'
 selection='validation pooled MSE, official best checkpoint'
 status='PREDECLARED_AND_EXECUTING'
}
$protocol=Join-Path $root 'SEED_PARITY_PROTOCOL_2932.json'
if(Test-Path $protocol){throw 'Seed protocol exists; do not launch again'}
$cfg|ConvertTo-Json |Set-Content -LiteralPath $protocol -Encoding UTF8
Set-Location $up
$a=@('-u','train.py',
'--data_path',"$root\data_v2\train.csv",
'--separate_val_path',"$root\data_v2\val.csv",
'--separate_test_path',"$root\data_v2\val_proxy_test.csv",
'--metric','mse','--dataset_type','regression',
'--save_dir',$dest,'--target_columns','label',
'--epochs','30','--ensemble_size','1','--num_folds','1',
'--batch_size','128','--num_workers','0','--cache_cutoff','100000',
'--aggregation','mean','--dropout','0.1','--pytorch_seed','2932','--seed','2932','--save_preds')
Write-Output ('RUN_PREDECLARED_SEED2932='+$dest)
Write-Output ('STARTED_AT='+(Get-Date))
& $py @a
$exit=$LASTEXITCODE
Write-Output ('TRAIN_SEED2932_EXIT='+$exit)
if($exit -ne 0){exit $exit}
Write-Output 'TRAIN_SEED2932_COMPLETED_VALIDATION_PROXY_ONLY'
