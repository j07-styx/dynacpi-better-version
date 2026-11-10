$ErrorActionPreference='Stop'
$root='C:\Users\J0124\DynaCPI-Backend'
$suite=Join-Path $root 'external_benchmarks_20261110\locked_external_protocol_v1'
$runner=Join-Path $root 'run_external_original_perceiver_frozen_20261110.ps1'
$log=Join-Path $suite 'OFFICIAL_PERCEIVER_KIBA_METZ_QUEUE_20261110.log'
if(Test-Path $log){throw 'External original-model training queue exists; cannot launch twice'}
function Note([string]$message){
 $line='['+(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')+'] '+$message
 Add-Content -LiteralPath $log -Value $line -Encoding UTF8
 Write-Output $line
}
function IsComplete([string]$dataset,[string]$scenario,[int]$seed){
 $target=Join-Path $suite ($dataset+'\'+$scenario+'\perceiver_official_seed'+$seed+'_30ep')
 $logpath=Join-Path $target 'verbose.log'
 $args=Join-Path $target 'args.json'
 $checkpoint=Join-Path $target 'fold_0\model_0\model.pt'
 if(!(Test-Path $logpath) -or !(Test-Path $args) -or !(Test-Path $checkpoint)){return $false}
 $contents=Get-Content $logpath -Raw
 if(!($contents.Contains('Epoch 29') -and $contents.Contains('Elapsed time') -and $contents.Contains('Overall test mse'))){return $false}
 $data=Get-Content $args -Raw |ConvertFrom-Json
 if($data.epochs -ne 30 -or $data.pytorch_seed -ne $seed -or $data.seed -ne $seed){return $false}
 return $true
}
Note 'START_EXTERNAL_OFFICIAL_CHECKPOINT_QUEUE TRAIN_VALIDATION_ONLY; TESTS_STAY_SEALED'
$cutoff=(Get-Date).AddHours(18)
while(!(IsComplete 'metz' 'protein_cold' 2931)){
 if((Get-Date) -gt $cutoff){throw 'Timeout waiting for first original Metz model'}
 $active=@(Get-CimInstance Win32_Process |Where-Object{
   $_.Name -eq 'python.exe' -and $_.CommandLine -match 'train\.py' -and
   $_.CommandLine -match 'metz\\protein_cold\\' -and
   $_.CommandLine -match 'seed2931_30ep'
 })
 if($active.Count -eq 0){throw 'First Metz original run exited without a verified checkpoint'}
 Note 'WAIT_FIRST_METZ_OFFICIAL_SEED_2931'
 Start-Sleep -Seconds 90
}
Note 'FIRST_METZ_OFFICIAL_SEED2931_COMPLETED_VERIFIED'
$jobs=@(
 @{name='metz';split='protein_cold';seed=2932},
 @{name='kiba';split='protein_cold';seed=2931},
 @{name='kiba';split='protein_cold';seed=2932},
 @{name='metz';split='compound_cold';seed=2931},
 @{name='metz';split='compound_cold';seed=2932},
 @{name='kiba';split='compound_cold';seed=2931},
 @{name='kiba';split='compound_cold';seed=2932},
 @{name='metz';split='double_cold';seed=2931},
 @{name='metz';split='double_cold';seed=2932},
 @{name='kiba';split='double_cold';seed=2931},
 @{name='kiba';split='double_cold';seed=2932}
)
foreach($job in $jobs){
 $dataset=[string]$job.name
 $split=[string]$job.split
 $seed=[int]$job.seed
 $target=Join-Path $suite ($dataset+'\'+$split+'\perceiver_official_seed'+$seed+'_30ep')
 if(Test-Path $target){throw ('Official external model directory exists: '+$target)}
 $outlog=Join-Path $suite ($dataset+'_'+$split+'_seed'+$seed+'.stdout.log')
 $errlog=Join-Path $suite ($dataset+'_'+$split+'_seed'+$seed+'.stderr.log')
 if(Test-Path $outlog){throw ('External stdout log exists: '+$outlog)}
 if(Test-Path $errlog){throw ('External stderr log exists: '+$errlog)}
 $limit=(Get-Date).AddHours(4)
 while($true){
   $os=Get-CimInstance Win32_OperatingSystem
   $freeGB=[math]::Round($os.FreePhysicalMemory/1MB,2)
   if($freeGB -ge 2.80){break}
   if((Get-Date) -gt $limit){
     Note ('MEMORY_GUARD_TIMEOUT '+$dataset+'/'+$split+' SEED='+$seed)
     throw 'Insufficient free system RAM. No sealed test opened.'
   }
   Note ('WAIT_FOR_RAM '+$freeGB+'GB BELOW 2.80GB '+$dataset+'/'+$split)
   Start-Sleep -Seconds 120
 }
 Note ('RUN_ORIGINAL_TRAIN '+$dataset+'/'+$split+' seed='+$seed+' free_RAM_GB='+$freeGB)
 $args=@('-NoProfile','-NonInteractive','-ExecutionPolicy','Bypass',
         '-File',$runner,'-Dataset',$dataset,'-Scenario',$split,
         '-Seed',[string]$seed)
 $proc=Start-Process -FilePath 'powershell.exe' -ArgumentList $args -PassThru -Wait -RedirectStandardOutput $outlog -RedirectStandardError $errlog
 if($proc.ExitCode -ne 0){
     Note ('FAILED '+$dataset+'/'+$split+' seed='+$seed+' exit='+$proc.ExitCode)
     throw 'Original external model training failed; no test opened'
 }
 if(!(IsComplete $dataset $split $seed)){
     Note ('INCOMPLETE '+$dataset+'/'+$split+' seed='+$seed)
     throw 'Training exit 0 but no completed selected checkpoint'
 }
 Note ('COMPLETE_ORIGINAL_MODEL '+$dataset+'/'+$split+' seed='+$seed+
       ' checkpoint_SHA256='+((Get-FileHash (Join-Path $target 'fold_0\model_0\model.pt') -Algorithm SHA256).Hash))
}
Note 'ALL_12_ORIGINAL_EXTERNAL_MODELS_TRAINED_AND_VALIDATION_SELECTED; ALL_SIX_TESTS_SEALED'
