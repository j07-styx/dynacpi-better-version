import joblib,json
from dynacpi.balm_public_data_dev import ROOT,get_train_only
from dynacpi.balm_memory_predictor_dev import predict,AA_ALPHABET

model=joblib.load(ROOT/"data/models/balm_affinity_memory_prototype_exploratory_20261108.joblib")
s=next(r["protein"] for r in get_train_only() if len(r["protein"])>=100 and set(r["protein"]).issubset(AA_ALPHABET))
pos=50
mutated=s[:pos]+("G" if s[pos]!="G" else "A")+s[pos+1:]
assert mutated!=s
out=predict(model,"CCO",mutated,device="cuda")
assert out["status"]=="EXPLORATORY_UNVALIDATED_RESEARCH_ONLY"
print("UNSEEN_SYNTHETIC_PROTEIN_SMOKE",json.dumps(out,indent=2))
