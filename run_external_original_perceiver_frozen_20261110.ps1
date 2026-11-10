param(
 [Parameter(Mandatory=$true)]
 [ValidateSet('kiba','metz')]
 [string]$Dataset,
 [Parameter(Mandatory=$true)]
 [ValidateSet('protein_cold','compound_cold','double_cold')]
 [string]$Scenario,
 [Parameter(Mandatory=$true)]
 [ValidateSet(2931,2932)]
 [int]$Seed
)
$ErrorActionPreference='Stop'
$root='C:\Users\J0124\DynaCPI-Backend'
$local=Join-Path $root ('external_benchmarks_20261110\locked_external_protocol_v1\'+$Dataset)
$dir=Join-Path $local $Scenario
$up=Join-Path $root 'matched_perceivercpi_20261110\upstream'
$py='C:\Users\J0124\dynacpi_v4\.venv\Scripts\python.exe'
$target=Join-Path $dir ('perceiver_official_seed'+$Seed+'_30ep')
if(Test-Path $target){throw ('An official training directory already exists; no overwrite: '+$target)}
$specfile=Join-Path $dir 'SPLIT_LOCKED.json'
if(!(Test-Path $specfile)){throw 'Frozen split does not exist'}
$spec=Get-Content $specfile -Raw|ConvertFrom-Json
$audit=Get-Content (Join-Path $local 'SOURCE_AUDIT.json') -Raw|ConvertFrom-Json
$match=@($audit.splits|Where-Object{$_.name -eq $Scenario})
if($match.Count -ne 1){throw 'No exactly matching locked source partition'}
$sha=(Get-FileHash -LiteralPath $specfile -Algorithm SHA256).Hash.ToLowerInvariant()
if($match[0].manifest_sha256 -ne $sha){throw 'External frozen split hash differs'}
if($spec.dataset -ne $Dataset -or $spec.scenario -ne $Scenario -or !$spec.test_labels_not_exported){
 throw 'External test split/source metadata invalid'
}
if((@($spec.training_seeds) -join ',') -ne '2931,2932'){
 throw 'Changed preregistered seed protocol'
}
$original=(git -C $up rev-parse HEAD).Trim()
if($original -ne '755f076ef51099a90332fd1ec8437f94e3750c0b'){
 throw 'Upstream original PerceiverCPI git commit differs'
}
$changes=(git -C $up diff --name-only -- '*.py' |Out-String).Trim()
if($changes){throw ('Upstream PerceiverCPI Python source modified: '+$changes)}
foreach($file in @('train.csv','validation.csv','validation_proxy_test.csv')){
 $h=(Get-FileHash (Join-Path $dir $file) -Algorithm SHA256).Hash.ToLowerInvariant()
 if($spec.csv_sha256.$file -ne $h){throw ('Frozen CSV hash mismatch '+$file)}
}
if($spec.csv_sha256.'validation.csv' -ne $spec.csv_sha256.'validation_proxy_test.csv'){
 throw 'Proxy is NOT a byte-identical copy of validation'
}
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='2'
$env:PYTHONHASHSEED=[string]$Seed
$env:PYTHONWARNINGS='ignore::UserWarning'
$env:TOKENIZERS_PARALLELISM='false'
$cmd=@('-u','train.py',
'--data_path',"$dir\train.csv",
'--separate_val_path',"$dir\validation.csv",
'--separate_test_path',"$dir\validation_proxy_test.csv",
'--metric','mse','--dataset_type','regression',
'--save_dir',$target,'--target_columns','label',
'--epochs','30','--ensemble_size','1','--num_folds','1',
'--batch_size','128','--num_workers','0','--cache_cutoff','100000',
'--aggregation','mean','--dropout','0.1',
'--pytorch_seed',[string]$Seed,'--seed',[string]$Seed,
'--save_preds')
Write-Output ('EXTERNAL_ORIGINAL_PERCEIVER_START '+$Dataset+'/'+$Scenario+' SEED='+$Seed+
     ' Time='+(Get-Date))
Write-Output ('LABEL_UNITS='+$spec.label_definition)
Write-Output 'TEST_LABELS_NOT_EXPORTED=TRUE'
Set-Location $up
& $py @cmd
$exit=$LASTEXITCODE
Write-Output ('ORIGINAL_PERCEIVER_FINISHED EXIT='+$exit+' AT='+(Get-Date))
if($exit -ne 0){exit $exit}
Write-Output 'ORIGINAL_PERCEIVER_VALIDATION_SELECTED_CHECKPOINT_READY_NO_TRUE_TEST'
