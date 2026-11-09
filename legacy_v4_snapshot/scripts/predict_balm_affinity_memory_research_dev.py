"""Predict experimental BALM p-affinity label from SMILES and protein sequence.

Research-only, no source-assay QC certification. Input a single sequence (not
a FASTA header). Trusted locally generated checkpoint only; joblib is pickle.
"""
import argparse,json
import joblib
from dynacpi.balm_public_data_dev import ROOT
from dynacpi.balm_memory_predictor_dev import predict

DEFAULT=ROOT/"data/models/balm_affinity_memory_prototype_exploratory_20261108.joblib"

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--smiles",required=True)
    protein=p.add_mutually_exclusive_group(required=True)
    protein.add_argument("--sequence")
    protein.add_argument("--sequence-file")
    p.add_argument("--checkpoint",default=str(DEFAULT))
    p.add_argument("--device",choices=("cuda","cpu"),default=None)
    args=p.parse_args()
    seq=args.sequence if args.sequence is not None else open(
        args.sequence_file,encoding="utf-8").read()
    if seq.lstrip().startswith(">"):
        raise ValueError("Pass a sequence only, not a FASTA header")
    # Only load checkpoints generated on a trusted local machine.
    model=joblib.load(args.checkpoint)
    if model.get("model_kind")!="EXPLORATORY_BALM_TRAIN_ONLY_MEMORY_CPI":
        raise ValueError("Unsupported or unapproved checkpoint type")
    result=predict(model,args.smiles,seq,device=args.device)
    print(json.dumps(result,indent=2))

if __name__=="__main__":main()
