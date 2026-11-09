"""Dynamic graph/token collation without molecule or sequence truncation."""
import torch
from torch.nn.utils.rnn import pad_sequence
class TransferCollator:
    def __init__(self,graphs,proteins,mean,std):
        self.graphs=graphs;self.proteins=proteins;self.mean=mean;self.std=std
    def __call__(self,rows):
        gs=[self.graphs[r["compound"]] for r in rows]
        ps=[self.proteins[r["protein"]] for r in rows]
        n=max(len(g["z"]) for g in gs);adj=torch.zeros(len(gs),4,n,n)
        for i,g in enumerate(gs):
            m=len(g["z"]);adj[i,:,:m,:m]=g["adj"]
        z=pad_sequence([g["z"] for g in gs],batch_first=True)
        amask=torch.arange(n)[None,:]<torch.tensor([len(g["z"]) for g in gs])[:,None]
        proteins=pad_sequence(ps,batch_first=True).float()
        pmask=torch.arange(proteins.shape[1])[None,:]<torch.tensor([len(p) for p in ps])[:,None]
        batch={"z":z,"atom_features":pad_sequence([g["atom_features"] for g in gs],batch_first=True),
          "adj":adj,"atom_mask":amask,"protein":proteins,"protein_mask":pmask,
          "properties":((torch.stack([g["properties"] for g in gs])-self.mean)/self.std).clamp(-10,10),
          "fp":torch.stack([g["fp"] for g in gs]).float(),
          "y":torch.tensor([r["y"] for r in rows],dtype=torch.float32)}

        if "chemical_language" in gs[0]:batch["chemical_language"]=torch.stack([g["chemical_language"] for g in gs]).float()
        return batch
