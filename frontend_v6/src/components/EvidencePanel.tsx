import { useId, useState } from "react";
import { ChevronDown } from "lucide-react";
import type { MetricsResponse, TrainingResponse } from "@/lib/api";

const configNames: Record<string,string> = {
  raw_esm_mse:"Raw ESM · MSE",
  centered_esm_mse:"Centered ESM · MSE",
  centered_esm_censored:"Centered ESM · censored loss",
};
const fmt=(n:number)=>Number.isFinite(n)?n.toFixed(4):"—";

export default function EvidencePanel({metrics,training,connected}:{metrics:MetricsResponse|null;training:TrainingResponse|null;connected:boolean}) {
  const [open,setOpen]=useState(true);
  const id=useId();
  const heading=id+"-heading";
  const primary=training?.runs.find(r=>r.variant===training.variant&&r.seed===2931)??training?.runs.find(r=>r.variant===training.variant);
  const h=primary?.history??[];
  const bottom=Math.min(...h.map(x=>x.validation_macro_rmse));
  const top=Math.max(...h.map(x=>x.validation_macro_rmse));
  const points=h.map((r,i)=>(18+i/Math.max(1,h.length-1)*565).toFixed(1)+","+(100-(r.validation_macro_rmse-bottom)/Math.max(.0001,top-bottom)*82).toFixed(1)).join(" ");
  return <section className="surface" aria-labelledby={heading}>
    <h2 className="m-0" id={heading}>
      <button type="button" aria-expanded={open} aria-controls={id} onClick={()=>setOpen(x=>!x)}
        className="flex w-full items-center justify-between gap-4 rounded-[inherit] px-5 py-4 text-left transition-colors hover:bg-panel/50">
        <span className="flex flex-col gap-1">
          <span className="micro !text-foreground/90">Earlier v5 · held-out evaluation (separate checkpoints)</span>
          <span className="text-xs text-muted-foreground">302 training · 59 validation · 81 held-out protein entries · 68 warm compounds</span>
        </span>
        <div className="flex items-center gap-2"><span className="micro">{connected?"Live backend":"Offline"}</span><ChevronDown size={16} className="text-muted-foreground" style={{transform:open?"rotate(180deg)":"none"}}/></div>
      </button>
    </h2>
    <div id={id} role="region" aria-labelledby={heading} hidden={!open} className="border-t border-border px-5 pb-5 pt-5">
      {!metrics ? <p className="panel p-5 text-sm text-muted-foreground">Waiting for the separately evaluated v5 model metrics. Run the backend on 127.0.0.1:8782. No results are simulated.</p> : <>
        <div className="grid grid-cols-2 gap-3 lg:grid-cols-4">
          <Card label="Chemical-only RMSE" value={fmt(metrics.test_chemical_only.macro_rmse)} desc="Held-out target-macro"/>
          <Card label="DynaCPI v5 RMSE" value={fmt(metrics.test_selected.macro_rmse)} desc="Lower is better" active/>
          <Card label="Relative improvement" value={fmt(metrics.relative_rmse_reduction_pct)+"%"} desc="Vs chemical-only" active/>
          <Card label="Protein targets improved" value={metrics.n_better+"/"+(metrics.n_better+metrics.n_worse)} desc="Held-out group test"/>
        </div>
        <div className="mt-4 grid gap-4 lg:grid-cols-2">
          <div className="panel p-4">
            <h3 className="text-sm font-semibold">Validation candidates</h3>
            <p className="mt-1 text-xs text-muted-foreground">Only validation chose model family and checkpoint.</p>
            <div className="mt-4 grid gap-3">
              {Object.entries(configNames).map(([key,name])=>{
                const score=metrics.validation_variants?.[key]?.macro_rmse;
                return <div key={key} className="flex items-center justify-between gap-2 text-xs">
                  <span className={metrics.selected===key?"text-primary":"text-muted-foreground"}>{name}</span>
                  <span className="font-mono">{typeof score==="number"?fmt(score):"—"}</span>
                </div>;
              })}
            </div>
          </div>
          <div className="panel p-4">
            <div className="flex justify-between gap-2"><h3 className="text-sm font-semibold">Training trajectory</h3><span className="micro">{primary?"Seed "+primary.seed:"No history"}</span></div>
            {h.length>=2 ? <>
              <svg className="mt-4 w-full" viewBox="0 0 600 122" role="img" aria-label="Actual validation RMSE per training epoch">
                <line x1="18" x2="583" y1="100" y2="100" stroke="currentColor" opacity=".2"/>
                <polyline points={points} fill="none" stroke="var(--primary)" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"/>
                <text x="18" y="117" fontSize="11" fill="currentColor" opacity=".6">Epoch 1</text><text x="582" y="117" textAnchor="end" fontSize="11" fill="currentColor" opacity=".6">{h.length} epochs</text>
              </svg>
              <p className="mt-1 text-xs text-muted-foreground">Best saved epoch {primary?.best_epoch}; recorded validation RMSE {fmt(primary?.best_val??NaN)}.</p>
            </> : <p className="mt-4 text-xs text-muted-foreground">Waiting for training records…</p>}
          </div>
        </div>
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          <Card label="Pooled RMSE" value={fmt(metrics.test_chemical_only.pooled_rmse)+" → "+fmt(metrics.test_selected.pooled_rmse)} desc="Baseline → trained"/>
          <Card label="Centered MSE" value={fmt(metrics.test_chemical_only.centered_mse)+" → "+fmt(metrics.test_selected.centered_mse)} desc="Within-protein variation"/>
        </div>
      </>}
      <p className="mt-4 border-l-2 border-primary/50 pl-3 text-xs leading-relaxed text-muted-foreground">
        These v5 scores come from DIFFERENT, earlier checkpoint files (not the all-data final refit). Full DAVIS, protein-grouped by an approximate sequence-similarity proxy; all compounds were seen during training. Not a double-cold benchmark, a family-curated split, a clinical predictor, or a matched PerceiverCPI comparison. The pKd=5 censoring marker is a heuristic.
      </p>
    </div>
  </section>;
}
function Card({label,value,desc,active=false}:{label:string;value:string;desc:string;active?:boolean}) {
  return <div className="panel p-4"><p className="micro">{label}</p><p className={"mt-3 font-mono text-xl "+(active?"text-primary":"text-foreground")}>{value}</p><p className="mt-2 text-xs text-muted-foreground">{desc}</p></div>;
}
