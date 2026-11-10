$ErrorActionPreference='Stop'
$root='C:\Users\J0124\DynaCPI-Backend'
$source=Join-Path $root 'external_benchmarks_20261110\locked_external_protocol_v1'
$script=Join-Path $root 'train_external_sparse_dynacpi_v5_20261110.py'
$py='C:\Users\J0124\dynacpi_v4\.venv\Scripts\python.exe'
$log=Join-Path $source 'EXTERNAL_DYNACPI_VALIDATION_QUEUE.log'
if(Test-Path $log){throw 'External training queue log exists; never double-run'}
function Note([string]$msg){
 $t='['+(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')+'] '+$msg
 Add-Content -LiteralPath $log -Value $t -Encoding UTF8
 Write-Output $t
}
$jobs=@(
 @{name='metz';scenario='compound_cold'},
 @{name='kiba';scenario='compound_cold'},
 @{name='metz';scenario='double_cold'},
 @{name='kiba';scenario='double_cold'}
)
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='2'
Note 'START_DYNACPI_ONLY_FROZEN_EXTERNAL_TRAINING_NO_TEST_ACCESS'
foreach($j in $jobs){
 $dataset=[string]$j.name
 $scenario=[string]$j.scenario
 $dest=Join-Path $source ($dataset+'\'+$scenario+'\dynacpi_fixed_sparse_train')
 if(Test-Path $dest){throw ('Existing DynaCPI split checkpoint: refuse overwrite '+$dest)}
 Note ('START_EXTERNAL_TRAIN '+$dataset+'/'+$scenario)
 $args=@('-u',$script,'--dataset',$dataset,'--scenario',$scenario)
 & $py @args
 if($LASTEXITCODE -ne 0){
    Note ('FAILED '+$dataset+'/'+$scenario+' exit='+$LASTEXITCODE)
    throw 'Stopped on training error without exposing test labels'
 }
 $lock=Join-Path $dest 'VALIDATION_FROZEN.json'
 if(!(Test-Path $lock)){throw ('No validation-frozen outputs after train '+$dataset+'/'+$scenario)}
 $data=Get-Content $lock -Raw|ConvertFrom-Json
 if($data.status -ne 'VALIDATION_LOCKED_TWO_SEEDS_NO_TEST' -or $data.validation_two_seed.n_compound_protein_pairs -lt 100){
    throw 'Validation selection not safely frozen'
 }
 Note ('FINISHED_VALIDATION_ONLY '+$dataset+'/'+$scenario+
   ' macro='+$data.validation_two_seed.macro_rmse+
   ' pooled='+$data.validation_two_seed.pooled_rmse)
}
Note 'ALL_FOUR_REMAINING_DYNACPI_EXTERNAL_VALIDATIONS_COMPLETE; ALL_SIX_TESTS_SEALED'
