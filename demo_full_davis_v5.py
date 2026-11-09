"""Local benchmark-only demonstration of full-DAVIS low-rank DynaCPI v5.

- Loads the real, validation-selected two-seed checkpoints.
- No arbitrary SMILES / drug ranking / medical decision functionality.
- Demonstration uses fixed anonymized DAVIS benchmark pairs only.
- No changes to the older backend on localhost:8781.
"""
import json
import os
from pathlib import Path
import numpy as np
import torch
from flask import Flask, jsonify, request, Response

from train_full_davis_v5 import (
    OUT,PREP,SPLIT,RUN,SEEDS,CONFIGS,LowRankDynaCPI,
    normalize_protein,load_local_checkpoint,sha
)
app=Flask(__name__)
torch.set_num_threads(2)
DEVICE="cpu"
result=json.loads((RUN/"RESULTS.json").read_text(encoding="utf-8"))
verified=json.loads((RUN/"VERIFICATION.json").read_text(encoding="utf-8"))
selection=json.loads((RUN/"SELECTION_FROZEN.json").read_text(encoding="utf-8"))
manifest=json.loads(SPLIT.read_text(encoding="utf-8"))
if selection["selected_family"]!=result["selected_variant"]:
    raise RuntimeError("Serving selection mismatch")
if sha(RUN/"RESULTS.json")!=verified["train_results_sha256"]:
    raise RuntimeError("Serving verification provenance mismatch")
with np.load(PREP,allow_pickle=False) as z:
    pfeat=z["pfeat"];bits=z["bits"];y=z["y"]
    train=z["train"].astype(int)
    val=z["val"].astype(int)
    test=z["test"].astype(int)
scaled,mu,std=normalize_protein(pfeat,train)
mode=next(m for k,m,_ in CONFIGS if k==result["selected_variant"])
protein_vectors=torch.tensor(scaled[mode],dtype=torch.float32,device=DEVICE)
compound_vectors=torch.tensor(bits,dtype=torch.float32,device=DEVICE)
chemical_prior=torch.tensor(y[train].mean(0),dtype=torch.float32,device=DEVICE)
label_sd=float(y[train].std())
models=[]
for seed in SEEDS:
    checkpoint=load_local_checkpoint(RUN/"checkpoints"/f"{result['selected_variant']}_seed{seed}.pt")
    if checkpoint["training_data_sha256"]!=manifest["raw_Y_sha256"]:
        raise RuntimeError("Serving model data identity mismatch")
    model=LowRankDynaCPI().to(DEVICE)
    model.load_state_dict(checkpoint["state_dict"],strict=True)
    model.eval()
    models.append(model)
# Frozen representative sample positions, by arithmetic only, not selected
# based on labels, model accuracy, chemical names, or target identities.
positions=np.array([0],dtype=np.int64)
examples={"DV-01":0}  # one fixed, anonymized benchmark demonstration
example_options=[{"demo_id":"DV-01","label":"Fixed benchmark demonstration"}]

def raw_inference(demo_id):
    if demo_id not in examples:
        raise ValueError("This API only supports the 12 fixed anonymized benchmark examples.")
    pos=examples[demo_id]
    protein=int(test[pos//68]);compound=int(pos%68)
    with torch.inference_mode():
        perseed=[]
        for model in models:
            resid=model(protein_vectors[protein:protein+1],
                        compound_vectors[compound:compound+1])[0]*label_sd
            perseed.append(max(5.,float((chemical_prior[compound]+resid).item())))
    prediction=float(np.mean(perseed))
    truth=float(y[protein,compound])
    return {"demo_id":demo_id,
        "prediction_pKd":round(prediction,5),
        "reference_pKd":round(truth,5),
        "absolute_error":round(abs(prediction-truth),5),
        "seed_prediction_std":round(float(np.std(perseed)),6),
        "benchmark":"Public DeepDTA DAVIS",
        "heldout":"protein-grouped test; ligand identity seen in training",
        "label":"Research benchmark only; not evidence of therapeutic efficacy"}

@app.after_request
def cors(resp):
    resp.headers["Access-Control-Allow-Origin"]="*"
    resp.headers["Access-Control-Allow-Headers"]="Content-Type"
    resp.headers["Access-Control-Allow-Methods"]="GET,POST,OPTIONS"
    resp.headers["Cache-Control"]="no-store"
    return resp

@app.route("/api/v1/health")
def health():
    return jsonify({"status":"ok","service":"dynacpi-full-davis-v5",
       "model":"LowRankDynaCPI","checkpoint_count":len(models),
       "active_variant":result["selected_variant"],
       "source":"public DeepDTA DAVIS",
       "full_pairs":30056,
       "heldout_proteins":int(len(test)),
       "training_proteins":int(len(train)),
       "validation_proteins":int(len(val)),
       "note":"Cold-protein proxy split; no matched PerceiverCPI claim"})

@app.route("/api/v1/metrics")
def get_metrics():
    return jsonify({"status":"complete",
      "protocol":manifest["status"],
      "validation_baseline":result["chemical_prior_validation"],
      "validation_variants":result["validation_all_configurations"],
      "selected":result["selected_variant"],
      "test_chemical_only":result["baseline_test"],
      "test_selected":result["selected_test"],
      "relative_rmse_reduction_pct":
          verified["test_macro_rmse_relative_reduction_percent"],
      "n_better":verified["n_heldout_targets_better_than_baseline"],
      "n_worse":verified["n_heldout_targets_worse_than_baseline"],
      "test_group_bootstrap_delta_interval":
          verified["descriptive_group_bootstrap_95pct_mean_rmse_delta"],
      "limitations":manifest["caveats"]})

@app.route("/api/v1/training")
def training():
    return jsonify({"variant":result["selected_variant"],
        "runs":[{"seed":r["seed"],"variant":r["variant"],
                 "best_epoch":r["best_epoch"],"best_val":r["best_validation_macro_rmse"],
                 "history":r["history"]} for r in result["per_run"]]})

@app.route("/api/v1/examples")
def list_examples():
    return jsonify({"examples":example_options,
        "selection":"Single first-index held-out benchmark pair, not selected by affinity or prediction accuracy"})

@app.route("/api/v1/predict",methods=["POST","OPTIONS"])
def predict():
    if request.method=="OPTIONS":return Response(status=204)
    data=request.get_json(silent=True) or {}
    key=str(data.get("demo_id",""))
    try:return jsonify(raw_inference(key))
    except ValueError as exc:return jsonify({"error":str(exc)}),400

@app.route("/")
def index():
    return Response(PAGE,mimetype="text/html")

PAGE=r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DynaCPI — DAVIS Research Console</title>
<style>
:root{color-scheme:dark;--bg:#080916;--panel:#101327;--border:#2b2d46;--text:#ebecff;--mute:#9296b1;--violet:#9a83fb;--cyan:#75d9dd}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;font-family:Inter,"Segoe UI",system-ui,sans-serif;color:var(--text);background:radial-gradient(1100px 520px at 90% -10%,#35235350,transparent),var(--bg)}
.wrap{max-width:1130px;margin:auto;padding:0 28px 80px}header{height:92px;display:flex;align-items:center;justify-content:space-between;border-bottom:1px solid #ffffff1c}
.brand{display:flex;align-items:center;gap:14px;letter-spacing:.02em;font-size:20px;font-weight:750}.logo{width:38px;height:38px;border:1px solid #aa89fa;border-radius:12px;display:grid;place-items:center;background:#272041;box-shadow:0 0 25px #9068ff33;color:#dfd0ff}
.tag{border:1px solid #665394;background:#31294a55;border-radius:100px;padding:7px 12px;font-size:11px;letter-spacing:.07em;color:#beadfa;text-transform:uppercase}
.hero{padding:69px 0 38px;display:grid;grid-template-columns:1.4fr .6fr;gap:30px;align-items:center}
.eyebrow{color:var(--cyan);font-size:12px;letter-spacing:.19em;font-weight:750;text-transform:uppercase}
h1{font-size:clamp(36px,5vw,65px);line-height:1.07;letter-spacing:-.052em;margin:17px 0 20px;font-weight:720}
h1 span{color:#ab91ff}.lede{font-size:15px;color:#acafc7;line-height:1.8;max-width:650px}
.orb{width:min(260px,80%);aspect-ratio:1;border-radius:50%;margin:auto;position:relative;background:radial-gradient(circle at 34% 28%,#b59dff9a 0%,#5636aa 24%,#19102a 65%,#06050d 95%);box-shadow:0 0 92px #9f72fb25,inset -20px -32px 42px #0009;animation:float 7s ease-in-out infinite}
.orb:after,.orb:before{content:"";position:absolute;inset:-24px;border:1px solid #a380ed48;border-radius:50%;transform:rotate(-22deg) scaleY(.39);box-shadow:0 0 20px #a481ec20}
.orb:after{transform:rotate(55deg) scaleY(.29);inset:-34px;border-color:#bfa3ff4a}
@keyframes float{0%,100%{transform:translateY(0)}50%{transform:translateY(-12px)}}
.sectionhead{display:flex;justify-content:space-between;align-items:baseline;gap:14px;margin:26px 0 16px}h2{font-size:21px;letter-spacing:-.03em;margin:0}small,.quiet{color:var(--mute)}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px}.metric,.panel{background:linear-gradient(160deg,#15182c,#0f1120);border:1px solid var(--border);border-radius:18px;padding:22px;box-shadow:0 12px 26px #0002}
.metric .label{font-size:12px;color:var(--mute);margin-bottom:22px}.metric .num{font-size:37px;letter-spacing:-.045em;font-weight:690}.metric .sub{font-size:12px;color:#9e9fbb;margin-top:12px}.num.primary{color:#a995ff}.num.good{color:#7ae1d4}
.two{display:grid;grid-template-columns:1.2fr 1fr;gap:16px;margin-top:16px}.panel h3{font-weight:650;font-size:17px;margin:0 0 8px}.panel p{line-height:1.65;color:#acb0c8;font-size:13px}
.barline{margin:25px 0 4px}.barlabels{display:flex;justify-content:space-between;font-size:12px;color:#b5b8ce;margin-bottom:9px}.bar{width:100%;height:11px;background:#27263b;border-radius:20px;overflow:hidden}.bar span{height:100%;display:block;border-radius:20px;background:linear-gradient(90deg,#6955b0,#bb9dff)}
.bar.good span{background:linear-gradient(90deg,#369f9b,#6febcb)}
button,select{font:inherit}.btn{padding:12px 18px;background:linear-gradient(120deg,#8d6fec,#6a54ce);color:#fff;font-weight:650;border:0;border-radius:10px;cursor:pointer;box-shadow:0 6px 20px #8263ff24;transition:transform .15s}.btn:hover{transform:translateY(-1px)}.btn:disabled{opacity:.6}
select{min-width:210px;padding:12px;border-radius:10px;border:1px solid #393950;background:#16182c;color:#e5e4ff}
.controls{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin:20px 0}
.readout{display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin:22px 0}.readout div{border:1px solid #3b3659;border-radius:12px;padding:16px;background:#16152955}.readout label{font-size:11px;color:#a7a6c4}.readout strong{font-size:26px;display:block;margin-top:9px}
.list{display:grid;gap:14px;margin-top:20px}.item{display:grid;grid-template-columns:150px 1fr auto;gap:12px;align-items:center;color:#aaaec8;font-size:13px}.sbar{height:8px;border-radius:6px;background:#2c2940}.sbar b{display:block;height:100%;background:linear-gradient(90deg,#8972e3,#7ad7df);border-radius:6px}
.alert{padding:16px;border:1px solid #423853;background:#29203555;border-radius:14px;font-size:12px;color:#bcb9d1;line-height:1.6;margin-top:20px}
footer{padding-top:32px;color:#7e819c;font-size:12px}.status{display:inline-flex;gap:8px;align-items:center}.status:before{content:"";width:8px;height:8px;border-radius:50%;background:#65d9bd;box-shadow:0 0 12px #65d9bd}
@media(max-width:850px){.grid{grid-template-columns:repeat(2,1fr)}.two{grid-template-columns:1fr}.hero{grid-template-columns:1fr}.orb{display:none}}
@media(max-width:510px){.wrap{padding:0 16px 55px}.grid{grid-template-columns:1fr 1fr}.metric{padding:15px}.metric .num{font-size:26px}.readout{grid-template-columns:1fr}header{height:75px}}
</style></head>
<body><main class="wrap">
<header><div class="brand"><div class="logo">D</div>DynaCPI <span style="font-size:11px;font-weight:500;color:#8889a7">/ RESEARCH</span></div><div class="tag">V5 · FULL DAVIS</div></header>
<section class="hero"><div><div class="eyebrow"><span class="status">Trained model online</span> · FULL DAVIS</div><h1>Protein-aware.<br><span>Measured.</span></h1><p class="lede">A working compound–protein affinity prediction benchmark, trained on 30,056 DAVIS observations. Frozen ESM-2 features, a compact interaction network, and transparent held-out evaluation.</p><div style="margin-top:24px;color:#b8afdc;font-size:12px">442 proteins &nbsp;·&nbsp; 68 compounds &nbsp;·&nbsp; 2 independently trained seeds</div></div><div><div class="orb"></div></div></section>
<div class="sectionhead"><h2>Evaluation overview</h2><small>81 held-out protein entries · 5,508 pairs</small></div>
<div class="grid">
<div class="metric"><div class="label">Chemical-only · test RMSE</div><div class="num">0.8236</div><div class="sub">Same split, training-only prior</div></div>
<div class="metric"><div class="label">DynaCPI v5 · test RMSE</div><div class="num primary">0.5852</div><div class="sub">Validation-selected two-seed ensemble</div></div>
<div class="metric"><div class="label">Relative RMSE reduction</div><div class="num good">28.9%</div><div class="sub">Versus chemical-only control</div></div>
<div class="metric"><div class="label">Held-out proteins improved</div><div class="num">68<span style="font-size:19px;color:#898ca7"> / 81</span></div><div class="sub">Descriptive, one frozen split</div></div>
</div>
<div class="two">
<div class="panel"><h3>Measured error reduction</h3><p>Identical DAVIS protein-grouped test subset. Lower RMSE is better.</p>
<div class="barline"><div class="barlabels"><span>Chemical-only baseline</span><strong>0.8236</strong></div><div class="bar"><span style="width:82.36%"></span></div></div>
<div class="barline"><div class="barlabels"><span>Selected DynaCPI v5 ensemble</span><strong>0.5852</strong></div><div class="bar good"><span style="width:58.52%"></span></div></div>
<div class="alert">All compounds are warm; the held-out partition groups similar protein sequences using a 3-mer proxy. This is not a curated family split or an equivalent PerceiverCPI benchmark.</div></div>
<div class="panel"><h3>Validation selection</h3><p>Three fixed training configurations, each with two seeds. The selected version was determined without test scores.</p>
<div class="list">
<div class="item"><span>Raw ESM + MSE</span><div class="sbar"><b style="width:80.6%"></b></div><strong>0.5285</strong></div>
<div class="item"><span>Centered ESM + MSE</span><div class="sbar"><b style="width:79.6%"></b></div><strong>0.5229</strong></div>
<div class="item"><span>Centered + censored</span><div class="sbar"><b style="width:78.7%"></b></div><strong style="color:#a995ff">0.5169</strong></div>
</div></div></div>
<div class="sectionhead" style="margin-top:43px"><h2>Live model inference</h2><small>Checkpoint-backed · anonymized benchmark examples</small></div>
<div class="panel"><h3>Run a held-out pair through the trained network</h3><p>Run the single fixed, anonymized test pair. Each request executes the actual two-seed model, then compares the prediction to the published benchmark label. The example is the first held-out pair by index, not chosen by model accuracy.</p>
<div class="controls"><select id="case" disabled></select><button class="btn" id="run">Run inference</button><span class="quiet" id="state">Ready</span></div>
<div class="readout"><div><label>Predicted pKd</label><strong id="pred">—</strong></div><div><label>Benchmark reference</label><strong id="reference">—</strong></div><div><label>Absolute error</label><strong id="error">—</strong></div></div>
<p class="quiet" id="detail">Predictions are research estimates, not medical or therapeutic guidance.</p>
</div>
<div class="sectionhead" style="margin-top:42px"><h2>Training trajectory</h2><small>Selected configuration · seed 2931</small></div>
<div class="panel"><p>Validation macro RMSE by epoch. Model epoch was chosen using validation only.</p><svg viewBox="0 0 900 220" preserveAspectRatio="none" id="curve" style="width:100%;height:220px;display:block"></svg><div class="quiet" id="curveinfo" style="font-size:12px"></div></div>
<footer>DYNA CPI · 2026 research engineering demo · Local inference only · Not independently benchmarked against PerceiverCPI · No clinical use</footer>
</main>
<script>
const $=(id)=>document.getElementById(id);
async function get(url){let r=await fetch(url);if(!r.ok)throw new Error("Request failed");return await r.json()}
async function init(){let data=await get("/api/v1/examples"); $("case").innerHTML=data.examples.map(e=>'<option value="'+e.demo_id+'">'+e.label+'</option>').join("");await draw();$("run").click()}
$("run").addEventListener("click",async()=>{try{$("run").disabled=true;$("state").textContent="Running checkpoint inference...";let r=await fetch("/api/v1/predict",{method:"POST",headers:{"Content-Type":"application/json"},body:JSON.stringify({demo_id:$("case").value})});let data=await r.json();if(!r.ok)throw Error(data.error);$("pred").textContent=data.prediction_pKd.toFixed(3);$("reference").textContent=data.reference_pKd.toFixed(3);$("error").textContent=data.absolute_error.toFixed(3);$("detail").textContent="Actual two-seed model inference · "+data.heldout+" · fixed, non-cherry-picked sample";$("state").textContent="Inference complete"}catch(e){$("state").textContent="Error: "+e.message}finally{$("run").disabled=false}})
async function draw(){try{let data=await get("/api/v1/training");let h=data.runs.find(r=>r.variant===data.variant&&r.seed===2931);if(!h)return;let points=h.history.map(x=>[x.epoch,x.validation_macro_rmse]);let maxX=Math.max(...points.map(p=>p[0]));let ylo=Math.min(...points.map(p=>p[1]))-.015,yhi=Math.max(...points.map(p=>p[1]))+.015;let px=(x)=>50+(x-1)/Math.max(1,maxX-1)*820;let py=(y)=>180-(y-ylo)/(yhi-ylo)*135;let svg='<line x1="50" y1="185" x2="870" y2="185" stroke="#373753"/><line x1="50" y1="40" x2="50" y2="185" stroke="#373753"/>';let pstr=points.map(p=>px(p[0]).toFixed(1)+","+py(p[1]).toFixed(1)).join(" ");svg+='<polyline fill="none" stroke="#a48cf5" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" points="'+pstr+'"/>';let b=points.find(p=>p[0]===h.best_epoch);svg+='<circle cx="'+px(b[0])+'" cy="'+py(b[1])+'" r="6" fill="#76dbcb" stroke="#070916" stroke-width="3"/>';svg+='<text x="52" y="20" fill="#a5a7c4" font-size="14">Validation RMSE</text><text x="796" y="211" fill="#a5a7c4" font-size="12">Epoch '+maxX+'</text>';$("curve").innerHTML=svg;$("curveinfo").textContent="Best checkpoint: epoch "+h.best_epoch+" · validation macro RMSE "+h.best_val.toFixed(4)}catch(e){$("curveinfo").textContent="Unable to load history"}}
init().catch(e=>{$("state").textContent="Initialization error: "+e.message})
</script></body></html>"""

if __name__=="__main__":
    print(json.dumps({"status":"SERVING_SELECTED_FULL_DAVIS_V5","port":8782,
      "variant":result["selected_variant"],"example_count":len(examples),
      "test_rmse":result["selected_test"]["macro_rmse"]}),flush=True)
    app.run(host="127.0.0.1",port=8782,debug=False,use_reloader=False,threaded=True)
