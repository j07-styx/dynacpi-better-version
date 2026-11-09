"""Frozen ChemBERTa property-pretrained encoder, pinned revision, train/dev only."""
import sys,json,hashlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/"data/tooldeps_chemberta"))
import requests,torch
from dynacpi.transfer_data import load_development
def main():
    torch.set_num_threads(2)
    from transformers import AutoTokenizer,RobertaModel
    rows,_,split_hash=load_development();smiles={r["compound"]:r["smiles"] for r in rows}
    dest=ROOT/"data/transfer_chemberta_dev.pt"
    if dest.exists():raise FileExistsError(dest)
    name="DeepChem/ChemBERTa-77M-MTR"
    revision="66b895cab8adebea0cb59a8effa66b2020f204ca"  # Exact encoder used by the recorded experiments.
    tokenizer=AutoTokenizer.from_pretrained(name,revision=revision,cache_dir=str(ROOT/"data/hf_transfer_cache"))
    model,loading=RobertaModel.from_pretrained(name,revision=revision,add_pooling_layer=False,output_loading_info=True,cache_dir=str(ROOT/"data/hf_transfer_cache"))
    if loading["missing_keys"]:raise ValueError("Uninitialized encoder weights: "+str(loading["missing_keys"]))
    model=model.eval();assert model.config.hidden_size==384
    entries=[];long=0
    # Nonoverlapping token windows preserve coverage; no affinity labels enter this stage.
    for cid,sm in sorted(smiles.items()):
        ids=tokenizer.encode(sm,add_special_tokens=False)
        assert ids
        if len(ids)>510:long+=1
        for start in range(0,len(ids),510):
            chunk=ids[start:start+510]
            entries.append((cid,tokenizer.build_inputs_with_special_tokens(chunk),len(chunk)))
    entries.sort(key=lambda x:len(x[1]));sums={};counts={};started=time.time()
    with torch.inference_mode():
        for off in range(0,len(entries),32):
            selected=entries[off:off+32]
            batch=tokenizer.pad({"input_ids":[r[1] for r in selected]},padding=True,return_tensors="pt")
            hidden=model(**batch).last_hidden_state
            for i,(cid,ids,n) in enumerate(selected):
                value=hidden[i,1:1+n].sum(0)
                sums[cid]=sums.get(cid,torch.zeros(384))+value
                counts[cid]=counts.get(cid,0)+n
            if off%1024==0:
                print(json.dumps({"chemical_chunks":off+len(selected),"total":len(entries),"seconds":round(time.time()-started,1)}),flush=True)
    features={cid:(sums[cid]/counts[cid]).half() for cid in smiles}
    assert all(torch.isfinite(x).all() for x in features.values())
    meta={"model":name,"revision":revision,"hidden_size":384,"split_sha256":split_hash,
      "pooling":"mean non-special token states; full nonoverlapping 510-token windows",
      "long_molecules_windowed":long,"molecules":len(features),"missing_encoder_keys":loading["missing_keys"],
      "unused_checkpoint_keys":loading["unexpected_keys"],"device":"cpu","label_use":False}
    torch.save({"features":features,"meta":meta},dest)
    (ROOT/"research/transfer_cpi_dev_20261108/chemical_language_features.json").write_text(json.dumps(meta,indent=2))
    print(json.dumps(meta,indent=2),flush=True)
if __name__=="__main__":main()
