import { AlertTriangle, RefreshCw, Activity, ShieldAlert, Database } from "lucide-react";
import type { V6Prediction } from "@/lib/api";

export type PredState =
  | {kind:"empty"}
  | {kind:"loading"}
  | {kind:"error";message:string}
  | {kind:"success";result:V6Prediction;latencyMs:number};

export default function PredictionPanel({state,onRetry}:{state:PredState;onRetry:()=>void}) {
  return <section aria-labelledby="pred-h" className="surface p-5">
    <div className="mb-4 flex items-baseline justify-between gap-3">
      <h2 id="pred-h" className="micro !text-foreground/90">Prediction output · v6 final</h2>
      <span className="micro">Two real neural checkpoints</span>
    </div>
    <div className="panel relative min-h-[165px] overflow-hidden p-5" aria-live="polite">
      {state.kind==="empty"&&<div className="flex flex-col gap-3">
        <Activity size={22} className="text-primary"/>
        <p className="text-sm font-medium">Ready to predict binding affinity</p>
        <p className="text-sm text-muted-foreground">Enter any valid SMILES and a single protein sequence, or load an original DAVIS example. The model runs locally; it does not generate illustrative values.</p>
      </div>}
      {state.kind==="loading"&&<div role="status" className="flex flex-col gap-4">
        <p className="text-sm text-muted-foreground">Running frozen ESM-2 features and two trained v6 checkpoint heads…</p>
        <div className="relative h-px overflow-hidden bg-border"><div className="anim-scan absolute inset-y-0 w-1/3 bg-primary"/></div>
        <div className="h-10 w-36 animate-pulse rounded bg-background/70"/>
        <p className="text-xs text-muted-foreground">Unseen protein sequences require a fresh ESM-2 pass and may run more slowly.</p>
      </div>}
      {state.kind==="error"&&<div role="alert" className="flex flex-col gap-3">
        <div className="flex items-start gap-3">
          <AlertTriangle size={18} className="shrink-0 text-destructive"/>
          <div><p className="text-sm font-medium">Inference failed</p>
            <p className="mt-1 break-words text-sm text-muted-foreground">{state.message}</p></div>
        </div>
        <button type="button" className="btn self-start" onClick={onRetry}><RefreshCw size={14}/>Retry</button>
      </div>}
      {state.kind==="success"&&<div className="grid gap-4">
        <div className="flex flex-wrap items-start justify-between gap-2">
          <div>
            <p className="micro">Predicted binding affinity</p>
            <p className="mt-2 flex items-baseline gap-2">
              <span className="font-mono text-4xl font-semibold tracking-tight text-primary">{state.result.prediction_pKd.toFixed(4)}</span>
              <span className="font-mono text-sm text-muted-foreground">pKd</span>
            </p>
          </div>
          <div className={"rounded-md border px-2.5 py-1.5 text-xs "+(state.result.known_training_pair?"border-border text-muted-foreground":"border-primary/40 text-primary")}>
            {state.result.known_training_pair?"In-sample pair":"Extrapolative prediction"}
          </div>
        </div>
        <dl className="grid grid-cols-2 gap-3 border-t border-border pt-3 sm:grid-cols-3">
          <Meta label="Protein in training" value={state.result.protein_in_original_davis?"Yes":"No"}/>
          <Meta label="Compound in training" value={state.result.compound_in_original_davis?"Yes":"No"}/>
          <Meta label="Seed disagreement" value={state.result.seed_prediction_std.toFixed(4)}/>
          <Meta label="Processing time" value={state.latencyMs+" ms"}/>
          <Meta label="Protein features" value={state.result.protein_in_original_davis?"ESM-2 cache":"ESM-2 encoded"}/>
          <Meta label="Molecular prior" value={state.result.compound_in_original_davis?"DAVIS mean":"Tanimoto extrapolation"}/>
        </dl>
        <div className="flex items-start gap-2 border-t border-border pt-3">
          {state.result.known_training_pair?<Database size={16} className="mt-0.5 shrink-0 text-muted-foreground"/>:
            <ShieldAlert size={16} className="mt-0.5 shrink-0 text-primary"/>}
          <p className="text-xs leading-relaxed text-muted-foreground">{state.result.notice}</p>
        </div>
      </div>}
    </div>
    <p className="mt-3 text-xs leading-relaxed text-muted-foreground">
      Predicted pKd is a model output, not an experimental measurement. Seed disagreement is not a calibrated uncertainty interval.
      v6 used all DAVIS labels for refitting; its predictions are not independently validated on new compounds or proteins.
    </p>
  </section>;
}
function Meta({label,value}:{label:string;value:string}) {
  return <div className="min-w-0"><dt className="micro">{label}</dt><dd className="mt-1 break-words font-mono text-xs">{value}</dd></div>;
}
