"""Fetch versioned public BALM BindingDB arrow data into ignored reference directory.

Do not commit the ~20 MB source file. Its content SHA is retained in a
reproducibility manifest; this staging step never trains or evaluates models.
"""
import hashlib,json,urllib.request,time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
REF=ROOT/"data"/"reference"/"balm_bindingdb_202611"
MANIFEST=ROOT/"research"/"balm_bindingdb_public_source_manifest_dev.json"
URL="https://huggingface.co/datasets/BALM/BALM-benchmark/resolve/main/BindingDB_filtered/train/data-00000-of-00001.arrow"
EXPECTED_SIZE=19581136
def main():
 REF.mkdir(parents=True,exist_ok=True)
 target=REF/"data-00000-of-00001.arrow"
 start=time.time()
 if not target.exists():
  req=urllib.request.Request(URL,headers={"User-Agent":"DynaCPI-Research/1.0"})
  with urllib.request.urlopen(req,timeout=60) as response,target.with_suffix(".part").open("wb") as dst:
   while True:
    block=response.read(1024*1024)
    if not block:break
    dst.write(block)
  target.with_suffix(".part").replace(target)
 assert target.stat().st_size==EXPECTED_SIZE,(target.stat().st_size,EXPECTED_SIZE)
 import pyarrow as pa
 with pa.memory_map(str(target),"r") as source:
  try:reader=pa.ipc.RecordBatchFileReader(source)
  except pa.ArrowInvalid:
   source.seek(0)
   reader=pa.ipc.RecordBatchStreamReader(source)
  table=reader.read_all()
 schema={field.name:str(field.type) for field in table.schema}
 report={
  "status":"PUBLIC_BENCHMARK_SOURCE_STAGED_UNSPLIT_NO_MODEL_RESULTS",
  "dataset":"BALM/BALM-benchmark BindingDB_filtered",
  "source_url":URL,
  "upstream_license":"CC-BY-4.0 per publisher dataset card; preserve attribution",
  "source_bytes":target.stat().st_size,
  "source_sha256":hashlib.sha256(target.read_bytes()).hexdigest(),
  "record_rows":table.num_rows,
  "columns":schema,
  "source_path_relative":"data/reference/balm_bindingdb_202611/data-00000-of-00001.arrow",
  "seconds":round(time.time()-start,1),
  "warnings":["Upstream dataset card labels source split 'train' only; it is an UNSPLIT corpus, not a model train partition.",
              "Construct and assay provenance not verified by this staging action.",
              "No prospectively held out records have been evaluated."]
 }
 MANIFEST.write_text(json.dumps(report,indent=2))
 print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()
