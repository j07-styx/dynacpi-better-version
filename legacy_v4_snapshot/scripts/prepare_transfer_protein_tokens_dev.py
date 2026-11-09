"""Full-sequence frozen ESM features for frozen train/dev only; no test features."""
import sys,json,hashlib,time,os
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from dynacpi.transfer_data import load_development
from gpu_lease import gpu_lease
def main():
    torch.set_num_threads(4)
    rows,seq,split_hash=load_development()
    out=ROOT/"data/transfer_esm64_dev";out.mkdir(exist_ok=True)
    cache=ROOT/"data/esm2_t33_650M_layer33_w1000_o128_fp16"
    meta=json.loads((cache/"settings.json").read_text())
    assert meta["checkpoint_sha256"]=="ea9d0522b335a8778dea6535a65301f10208dece28cd5865482b0b1fc446168c"
    model=None;started=time.time();computed=0;reused=0
    for n,(pid,s) in enumerate(sorted(seq.items(),key=lambda kv:(len(kv[1]),kv[0])),1):
        dest=out/(pid+".pt")
        if dest.exists():
            old=torch.load(dest,map_location="cpu",weights_only=True)
            assert old["sequence_hash"]==pid and old["source_meta"]==meta and old["pooling"]=="contiguous_nonoverlap_mean_max64"
            continue
        raw=cache/(pid+".pt")
        if raw.exists():
            obj=torch.load(raw,map_location="cpu",weights_only=True)
            assert obj["sequence_hash"]==pid and obj["meta"]==meta
            emb=obj["embeddings"].float();assert emb.shape==(len(s),1280);reused+=1
        else:
            if model is None:
                import esm
                weights=Path(torch.hub.get_dir())/"checkpoints/esm2_t33_650M_UR50D.pt"
                h=hashlib.sha256()
                with weights.open("rb") as f:
                    for b in iter(lambda:f.read(1<<20),b""):h.update(b)
                assert h.hexdigest()==meta["checkpoint_sha256"]
                model,alphabet=esm.pretrained.load_model_and_alphabet_local(str(weights))
                model=model.eval().half().cuda();converter=alphabet.get_batch_converter()
            emb=torch.zeros(len(s),1280);counts=torch.zeros(len(s),1)
            with torch.inference_mode():
                for start in range(0,len(s),872):
                    end=min(start+1000,len(s));_,_,t=converter([(pid,s[start:end])])
                    rep=model(t.cuda(),repr_layers=[33],return_contacts=False)["representations"][33][0,1:1+end-start].float().cpu()
                    emb[start:end]+=rep;counts[start:end]+=1
                    if end==len(s):break
            assert (counts>0).all();emb/=counts;computed+=1
        assert torch.isfinite(emb).all()
        tokens=torch.stack([part.mean(0) for part in torch.tensor_split(emb,min(64,len(s)))])
        temp=dest.with_suffix(".tmp");torch.save({"sequence_hash":pid,"length":len(s),"source_meta":meta,
          "pooling":"contiguous_nonoverlap_mean_max64","tokens":tokens.half()},temp);os.replace(temp,dest)
        progress={"completed":n,"total":len(seq),"computed_new":computed,"reused_raw":reused,
          "elapsed_seconds":round(time.time()-started,1),"last_length":len(s),"split_sha256":split_hash}
        (out/"progress.json").write_text(json.dumps(progress,indent=2))
        if n%10==0:print(json.dumps(progress),flush=True)
    (out/"COMPLETE.json").write_text(json.dumps({"sequences":len(seq),"split_sha256":split_hash,"scope":"train and inner dev only","source_meta":meta},indent=2))
    print("COMPLETE",len(seq),flush=True)
if __name__=="__main__":
    with gpu_lease(ROOT):main()
