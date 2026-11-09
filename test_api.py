"""HTTP integration tests for isolated DynaCPI backend."""
import json
import math
from urllib.request import Request, urlopen
from urllib.error import HTTPError

BASE="http://127.0.0.1:8781"

def call(route,body=None,method=None,origin=None):
    data=json.dumps(body).encode() if body is not None else None
    headers={"Content-Type":"application/json"}
    if origin:headers["Origin"]=origin
    req=Request(BASE+route,data=data,headers=headers,method=method)
    try:
        with urlopen(req,timeout=180) as r:
            return r.status,json.loads(r.read()) if r.status!=204 else {},r.headers
    except HTTPError as exc:
        return exc.code,json.loads(exc.read()),exc.headers

status,h,_=call("/api/v1/health")
assert status==200 and h["ok"] and h["backend_version"].startswith("v4.")
print("PASS server health",flush=True)
status,m,_=call("/api/v1/models")
assert status==200 and m["default_model_id"]=="sequence_baseline"
names={x["id"] for x in m["models"]}
assert {"sequence_baseline","repaired_sequence_2831","repaired_sequence_2832","repaired_sequence_2833"}<=names
print("PASS candidate registry and conservative default",flush=True)
status,t,_=call("/api/v1/targets")
assert status==200 and len(t["targets"])==4
print("PASS known DAVIS development proteins",flush=True)
status,tr,_=call("/api/v1/training/repaired")
assert status==200 and tr["candidate_count"]==3 and not tr["independent_scientific_validation"]
print("PASS saved model training metadata",flush=True)
status,invalid,_=call("/api/v1/predict",{"target":"FRK","smiles":"invalid?","model_id":"repaired_sequence_2831"},method="POST")
assert status==400 and "error" in invalid
print("PASS invalid molecule rejection",flush=True)
smi="Cn1c(=O)c2c(ncn2C)n(C)c1=O"
preds={}
for name in ["sequence_baseline","repaired_sequence_2831","repaired_sequence_2832","repaired_sequence_2833"]:
    status,result,_=call("/api/v1/predict",{"target":"FRK","smiles":smi,"model_id":name,"support_k":0},method="POST")
    if status!=200:raise AssertionError((name,status,result))
    assert result["model_id"]==name and not result["geometry_available"]
    assert math.isfinite(result["predictions"]["D_full_adapted_pkd"])
    preds[name]=result["predictions"]["sequence_pkd"]
    print("PASS live model",name,"predicted pKd",round(preds[name],5),flush=True)
status,result,_=call("/api/v1/predict",{"target":"ADCK4","smiles":smi,
    "model_id":"repaired_sequence_2833","support_k":5},method="POST")
assert status==200 and result["support_k"]==5
assert math.isfinite(result["predictions"]["D_full_adapted_pkd"])
print("PASS K=5 support calibration for repaired model",flush=True)
status,result,_=call("/api/v1/predict",method="OPTIONS",origin="http://localhost:5173")
assert status==204 and result=={}
print("PASS CORS preflight for external frontend",flush=True)
status,result,headers=call("/api/v1/models",origin="http://localhost:5173")
assert status==200 and headers.get("Access-Control-Allow-Origin")=="http://localhost:5173"
print("PASS CORS on GET response",flush=True)
print("ALL API CHECKS PASSED",flush=True)