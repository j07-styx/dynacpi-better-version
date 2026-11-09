from pathlib import Path
import hashlib,json,os,time,traceback
import torch,esm
root=Path(__file__).resolve().parents[1]
data=root/"data"/"staging";out=root/"data"/"esm2_t33_650M_layer33_w1000_o128_fp16";out.mkdir(parents=True,exist_ok=True)
status=out/"progress.json"
def progress(obj):
 tmp=status.with_suffix(".tmp");tmp.write_text(json.dumps(obj));os.replace(tmp,status)
def main():
 seqs=json.loads((data/"sequences.json").read_text());progress({"stage":"loading_model","total":len(seqs)})
 model,alphabet=esm.pretrained.esm2_t33_650M_UR50D()
 weights=Path(torch.hub.get_dir())/"checkpoints"/"esm2_t33_650M_UR50D.pt"
 digest=hashlib.sha256()
 with weights.open("rb") as f:
  for block in iter(lambda:f.read(1<<20),b""):digest.update(block)
 wd=digest.hexdigest()
 model=model.eval().half().cuda();converter=alphabet.get_batch_converter()
 meta={"model":"esm2_t33_650M_UR50D","layer":33,"checkpoint_sha256":wd,"esm_version":esm.__version__,"dtype":"float16","window":1000,"overlap":128}
 (out/"settings.json").write_text(json.dumps(meta,indent=2))
 started=time.time()
 with torch.inference_mode():
  for n,(pid,seq) in enumerate(sorted(seqs.items(),key=lambda z:(len(z[1]),z[0])),1):
   assert hashlib.sha256(seq.encode()).hexdigest()==pid
   dest=out/(pid+".pt")
   if dest.exists():
    old=torch.load(dest,map_location="cpu",weights_only=True)
    if old["meta"]==meta and old["sequence_hash"]==pid and old["embeddings"].shape==(len(seq),1280):continue
   emb=torch.zeros(len(seq),1280);counts=torch.zeros(len(seq),1);offsets=[]
   for start in range(0,len(seq),872):
    end=min(start+1000,len(seq));chunk=seq[start:end]; _,_,tokens=converter([(pid,chunk)])
    rep=model(tokens.cuda(),repr_layers=[33],return_contacts=False)["representations"][33][0,1:1+len(chunk)].float().cpu()
    if not torch.isfinite(rep).all():raise ValueError("Nonfinite ESM features for "+pid)
    emb[start:end]+=rep;counts[start:end]+=1;offsets.append((start,end))
    if end==len(seq):break
   assert (counts>0).all()
   tmp=dest.with_suffix(".tmp");torch.save({"sequence_hash":pid,"meta":meta,"embeddings":(emb/counts).half(),"windows":offsets},tmp);os.replace(tmp,dest)
   progress({"stage":"extracting","completed":n,"total":len(seqs),"length":len(seq),"peak_MiB":torch.cuda.max_memory_allocated()/2**20,"seconds":round(time.time()-started,1)})
 progress({"stage":"complete","files":len(list(out.glob("*.pt"))),"total":len(seqs),"peak_MiB":torch.cuda.max_memory_allocated()/2**20,"seconds":round(time.time()-started,1)})
if __name__=="__main__":
 try:
  from gpu_lease import gpu_lease
  with gpu_lease(root):main()
 except Exception as e:progress({"stage":"failed","error":repr(e)});raise

