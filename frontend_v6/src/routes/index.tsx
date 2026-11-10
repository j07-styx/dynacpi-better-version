import { createFileRoute, ClientOnly } from "@tanstack/react-router";
import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Eraser, FlaskConical, Loader2, Play, RefreshCw } from "lucide-react";
import Header, { type BackendState } from "@/components/Header";
import PredictionPanel, { type PredState } from "@/components/PredictionPanel";
import EvidencePanel from "@/components/EvidencePanel";
import { OrbitMark } from "@/components/OrbitMark";
import { ApiError, fetchMetrics, fetchTraining, fetchV6Example, fetchV6FitMetrics,
  fetchV6Health, getV6ApiUrl, predictV6, type MetricsResponse, type TrainingResponse, type V6FitMetrics } from "@/lib/api";
import { cleanProtein, validateProtein, validateSmiles, PROTEIN_MAX } from "@/lib/validation";

const MoleculeViewer=lazy(()=>import("@/components/MoleculeViewer"));
export const Route=createFileRoute("/")({
  head:()=>({meta:[{title:"DynaCPI v6 — Full DAVIS affinity research"},
    {name:"description",content:"Full DAVIS final-refit inference, verified training diagnostics, and separately evaluated v5 test results."}]}),
  component:Workspace,
});

const VISUAL_EXAMPLE={
  smiles:"CC(=O)NC1=NN=C(S1)S(N)(=O)=O",
  protein:"MSHHWGYGKHNGPEHWHKDFPIAKGERQSPVDIDTHTAKYDPSLKPLSVSYDQATSLRILNNGHAFNVEFDDSQDKAVLKGGPLDGTYRLIQFHFHWGSLDGQGSEHTVDKKKYAAELHLVHWNTKYGDFGKAVQQPDGLAVLGIFLKVGSAKPGLQKVVDVLDSIKTKGKSADFTNFDPRGLLPESLDYWTYPGSLTTPPLLECVTWIVLKEPISVSSEQVLKFRKLNFNGEGEPEELMVDNWRPAQPLKNRQIKASFK"
};
function useDebounced<T>(v:T,ms:number){
  const [d,setD]=useState(v);
  useEffect(()=>{const t=setTimeout(()=>setD(v),ms);return()=>clearTimeout(t)},[v,ms]);
  return d;
}
function Workspace(){
  const [smiles,setSmiles]=useState("");
  const [protein,setProtein]=useState("");
  const [touched,setTouched]=useState({s:false,p:false});
  const [pred,setPred]=useState<PredState>({kind:"empty"});
  const [runs,setRuns]=useState(0);
  const [backend,setBackend]=useState<BackendState>("checking");
  const [apiUrl,setApiUrl]=useState<string|null>(null);
  const [finalFit,setFinalFit]=useState<V6FitMetrics|null>(null);
  const [oldMetrics,setOldMetrics]=useState<MetricsResponse|null>(null);
  const [oldTraining,setOldTraining]=useState<TrainingResponse|null>(null);
  const [exampleLoading,setExampleLoading]=useState(false);
  const abort=useRef<AbortController|null>(null);
  const sv=useMemo(()=>validateSmiles(smiles),[smiles]);
  const pv=useMemo(()=>validateProtein(protein),[protein]);
  const residues=useMemo(()=>cleanProtein(protein).length,[protein]);
  const viewerSmiles=useDebounced(sv.ok?sv.value:null,350);
  const loading=pred.kind==="loading";

  const recheck=useCallback(async()=>{
    const url=getV6ApiUrl();
    setApiUrl(url);
    if(!url){setBackend("unconfigured");setFinalFit(null);setOldMetrics(null);setOldTraining(null);return}
    setBackend("checking");
    try{
      const h=await fetchV6Health();
      if(h.status!=="ok"||h.service!=="dynacpi-v6-all-davis"||h.n_checkpoints!==2||h.n_pairs_trained!==30056)
        throw new Error("Unexpected model or training identity");
      setBackend("connected");
    }catch{
      setBackend("unreachable");setFinalFit(null);setOldMetrics(null);setOldTraining(null);return;
    }
    const [fit,prev,history]=await Promise.allSettled([fetchV6FitMetrics(),fetchMetrics(),fetchTraining()]);
    setFinalFit(fit.status==="fulfilled"&&fit.value.status==="VERIFIED_ALL_DATA_TRAINING_FIT_NOT_HELDOUT"?fit.value:null);
    setOldMetrics(prev.status==="fulfilled"?prev.value:null);
    setOldTraining(history.status==="fulfilled"?history.value:null);
  },[]);
  useEffect(()=>{void recheck()},[recheck]);

  const reset=(s:string,p:string)=>{
    abort.current?.abort();setSmiles(s);setProtein(p);setTouched({s:true,p:true});setPred({kind:"empty"});
  };
  const loadDAVIS=async()=>{
    setExampleLoading(true);
    try{const e=await fetchV6Example();reset(e.smiles,e.sequence);}
    catch(e){setPred({kind:"error",message:"Unable to load DAVIS example: "+String((e as Error).message??e)})}
    finally{setExampleLoading(false)}
  };
  const run=useCallback(async()=>{
    setTouched({s:true,p:true});
    if(!sv.ok||!pv.ok)return;
    abort.current?.abort();
    const controller=new AbortController();
    abort.current=controller;
    setPred({kind:"loading"});
    const started=performance.now();
    try{
      const result=await predictV6({smiles:sv.value,sequence:pv.value},controller.signal);
      if(controller.signal.aborted)return;
      setPred({kind:"success",result,latencyMs:Math.round(performance.now()-started)});
      setRuns(n=>n+1);setBackend("connected");
    }catch(e){
      if(controller.signal.aborted)return;
      const err=e instanceof ApiError?e:new ApiError(String((e as Error).message??e),"network");
      setPred({kind:"error",message:err.message});
      if(err.kind==="network")setBackend("unreachable");
    }
  },[sv,pv]);
  const onKey=(e:React.KeyboardEvent)=>{if((e.metaKey||e.ctrlKey)&&e.key==="Enter"){e.preventDefault();void run()}};

  return <div className="min-h-screen">
    <Header backend={backend} apiUrl={apiUrl} onRecheck={()=>void recheck()} runs={runs}/>
    <main className="mx-auto grid max-w-[1440px] gap-5 px-4 py-6 md:px-8 md:py-8">
      <section className="surface relative overflow-hidden p-5 md:p-7">
        <div className="pointer-events-none absolute -right-14 -top-12 h-48 w-48 rounded-full bg-primary/10 blur-3xl"/>
        <div className="relative flex flex-wrap items-center justify-between gap-4">
          <div>
            <p className="micro text-primary">DYNACPI · ORIGINAL DAVIS · FINAL V6</p>
            <h1 className="mt-2 text-2xl font-semibold tracking-tight md:text-3xl">Compound–protein affinity prediction</h1>
            <p className="mt-2 max-w-2xl text-sm leading-relaxed text-muted-foreground">
              Frozen ESM-2 protein embeddings · low-rank interaction head · two-seed ensemble trained on all DAVIS pairs.
              Enter a SMILES and a protein sequence to execute the real model.
            </p>
          </div>
          <div className="flex flex-wrap gap-3">
            <Summary k="Protein entries" v="442"/><Summary k="Compounds" v="68"/><Summary k="Training pairs" v="30,056"/>
          </div>
        </div>
      </section>
      <div className="grid gap-5 lg:grid-cols-[minmax(0,5fr)_minmax(0,6fr)]">
        <form className="surface flex flex-col gap-5 p-5" onSubmit={e=>{e.preventDefault();void run()}} onKeyDown={onKey} noValidate aria-label="Prediction inputs">
          <div className="flex items-baseline justify-between"><h2 className="micro !text-foreground/90">Model inputs</h2><span className="micro hidden sm:block">Ctrl / ⌘ + Enter</span></div>
          <div className="flex flex-col gap-2">
            <div className="flex items-baseline justify-between">
              <label htmlFor="smiles" className="text-sm font-medium">Compound <span className="text-muted-foreground">SMILES</span></label>
              <span className="font-mono text-[11px] text-muted-foreground">{smiles.length} / 500</span>
            </div>
            <input id="smiles" className="field" value={smiles} spellCheck={false} autoComplete="off" placeholder="CC(=O)Oc1ccccc1C(=O)O"
              aria-invalid={touched.s&&!sv.ok}
              onChange={e=>{abort.current?.abort();setSmiles(e.target.value);setPred({kind:"empty"})}}
              onBlur={()=>setTouched(t=>({...t,s:true}))}/>
            <p className={"min-h-4 text-xs "+(touched.s&&!sv.ok?"text-destructive":"text-muted-foreground")}>
              {touched.s&&!sv.ok?(sv as {error:string}).error:sv.ok?"Ready for RDKit validation in the model backend.":"Enter one molecular SMILES."}
            </p>
          </div>
          <div className="flex flex-col gap-2">
            <div className="flex items-baseline justify-between">
              <label htmlFor="protein" className="text-sm font-medium">Protein <span className="text-muted-foreground">amino-acid sequence</span></label>
              <span className="font-mono text-[11px] text-muted-foreground">{residues} / {PROTEIN_MAX} aa</span>
            </div>
            <textarea id="protein" className="field min-h-[190px] resize-y break-all leading-relaxed" value={protein}
              spellCheck={false} placeholder="One sequence or a FASTA record" aria-invalid={touched.p&&!pv.ok}
              onChange={e=>{abort.current?.abort();setProtein(e.target.value);setPred({kind:"empty"})}}
              onBlur={()=>setTouched(t=>({...t,p:true}))}/>
            <p className={"min-h-4 text-xs "+(touched.p&&!pv.ok?"text-destructive":"text-muted-foreground")}>
              {touched.p&&!pv.ok?(pv as {error:string}).error:pv.ok?"Sequence accepted. Unseen sequences use frozen ESM-2.":"One protein sequence or single-record FASTA."}
            </p>
          </div>
          <div className="mt-auto flex flex-wrap items-center gap-2 border-t border-border pt-4">
            <button type="button" className="btn" disabled={exampleLoading} onClick={()=>void loadDAVIS()}>
              {exampleLoading?<Loader2 size={14} className="animate-spin"/>:<FlaskConical size={14}/>} Load DAVIS example
            </button>
            <button type="button" className="btn btn-ghost" onClick={()=>reset(VISUAL_EXAMPLE.smiles,VISUAL_EXAMPLE.protein)}>Load visual example</button>
            <button type="button" className="btn btn-ghost" disabled={!smiles&&!protein} onClick={()=>reset("","")}><Eraser size={14}/>Clear</button>
            <button type="submit" className="btn btn-primary ml-auto" disabled={loading||backend!=="connected"} aria-busy={loading}>
              {loading?<Loader2 size={14} className="animate-spin"/>:<Play size={14}/>} {loading?"Predicting…":"Predict affinity"}
            </button>
          </div>
          {backend!=="connected"&&<p className="text-xs text-muted-foreground">Final model backend unavailable on port 8783. <button type="button" className="underline" onClick={()=>void recheck()}><RefreshCw size={12} className="mr-1 inline"/>Retry connection</button></p>}
          <p className="-mt-1 text-xs leading-relaxed text-muted-foreground">
            The DAVIS example was used in training; its prediction is in-sample. Predictions for unseen proteins or compounds are exploratory and not independently validated.
          </p>
        </form>
        <div className="flex flex-col gap-5">
          <section className="surface flex flex-col p-1.5" aria-label="Compound 3D preview">
            <div className="flex items-baseline justify-between px-3.5 py-2.5">
              <h2 className="micro !text-foreground/90">Compound structure</h2><span className="micro">Visualization only</span>
            </div>
            <div className="h-[420px]">
              <ClientOnly fallback={<ViewerFallback/>}>
                <Suspense fallback={<ViewerFallback/>}><MoleculeViewer smiles={viewerSmiles}/></Suspense>
              </ClientOnly>
            </div>
          </section>
          <PredictionPanel state={pred} onRetry={()=>void run()}/>
        </div>
      </div>
      <FinalFitPanel metrics={finalFit}/>
      <EvidencePanel metrics={oldMetrics} training={oldTraining} connected={Boolean(oldMetrics&&oldTraining)}/>
      <footer className="flex flex-wrap justify-between gap-2 py-2 font-mono text-[11px] text-muted-foreground">
        <span>v6 inference endpoint: {apiUrl??"offline"}</span>
        <span>Research use only · separate v5 held-out evidence · no matched PerceiverCPI</span>
      </footer>
    </main>
  </div>;
}
function Summary({k,v}:{k:string;v:string}) {
  return <div className="rounded-lg border border-border bg-card/70 px-3 py-2"><div className="micro">{k}</div><div className="mt-1 font-mono text-xl">{v}</div></div>;
}
function ViewerFallback(){return <div className="grid-bg flex h-full items-center justify-center rounded-md bg-background"><OrbitMark active/></div>}
function FinalFitPanel({metrics}:{metrics:V6FitMetrics|null}) {
  const primary=metrics?.training_runs.find(r=>r.seed===2931);
  const hist=primary?.history??[];
  const min=Math.min(...hist.map(r=>r.standardized_training_objective));
  const max=Math.max(...hist.map(r=>r.standardized_training_objective));
  const poly=hist.map((v,i)=>(16+i/Math.max(1,hist.length-1)*560).toFixed(1)+","+(100-(v.standardized_training_objective-min)/Math.max(.001,max-min)*80).toFixed(1)).join(" ");
  return <section className="surface p-5">
    <div className="flex flex-wrap items-center justify-between gap-3"><h2 className="micro !text-foreground/90">v6 · final training-fit metrics</h2><span className="micro">Not an independent test</span></div>
    {!metrics?<p className="mt-4 text-sm text-muted-foreground">Waiting for verified all-data training-fit metrics from v6.</p>:<>
      <div className="mt-4 grid grid-cols-2 gap-3 lg:grid-cols-4">
        <Stat k="Chemical mean · training RMSE" v={metrics.all_data_fit_baseline_chemical_mean.macro_rmse.toFixed(4)}/>
        <Stat k="DynaCPI v6 · training RMSE" v={metrics.all_data_fit_v6_ensemble.macro_rmse.toFixed(4)} accent/>
        <Stat k="Training-fit RMSE reduction" v={metrics.all_data_fit_improvement_relative_baseline_percent.toFixed(2)+"%"}/>
        <Stat k="Training centered MSE" v={metrics.all_data_fit_v6_ensemble.centered_mse.toFixed(4)}/>
      </div>
      <div className="mt-4 grid gap-3 lg:grid-cols-2">
        <div className="panel p-4">
          <p className="text-sm font-medium">Recorded final-refit training objective</p>
          {hist.length>1&&<svg viewBox="0 0 600 120" className="mt-2 w-full" role="img" aria-label="Actual v6 training objective per epoch">
            <line x1="16" x2="580" y1="100" y2="100" stroke="currentColor" opacity=".15"/>
            <polyline points={poly} fill="none" stroke="var(--primary)" strokeWidth="2.5" strokeLinejoin="round"/>
            <text x="16" y="115" fontSize="11" fill="currentColor" opacity=".7">Epoch 1</text>
            <text x="580" y="115" fontSize="11" fill="currentColor" opacity=".7" textAnchor="end">{hist.length} epochs</text>
          </svg>}
          <p className="mt-1 text-xs text-muted-foreground">Seed {primary?.seed}; this is training loss, not validation.</p>
        </div>
        <div className="panel p-4">
          <p className="text-sm font-medium">Trained model</p>
          <p className="mt-3 text-sm text-muted-foreground">Full original DAVIS: {metrics.pairs_train.toLocaleString()} pairs, {metrics.proteins_train_entries} entries ({metrics.unique_protein_sequences} unique proteins), {metrics.compounds_train} compounds.</p>
          <p className="mt-3 text-xs leading-relaxed text-muted-foreground">The selected architecture and two epoch budgets were frozen in the prior v5 validation experiment. The final refit used all labels; no DAVIS holdout remains for these checkpoints.</p>
        </div>
      </div>
    </>}
    <p className="mt-4 border-l-2 border-primary/50 pl-3 text-xs leading-relaxed text-muted-foreground">
      The v6 metrics above are IN-SAMPLE training-fit numbers. The v5 evidence below is a separate held-out evaluation from DIFFERENT, earlier model checkpoints; do not compare them as a same-model test.
    </p>
  </section>;
}
function Stat({k,v,accent=false}:{k:string;v:string;accent?:boolean}) {
  return <div className="panel p-4"><p className="micro">{k}</p><p className={"mt-3 font-mono text-2xl "+(accent?"text-primary":"text-foreground")}>{v}</p></div>;
}
