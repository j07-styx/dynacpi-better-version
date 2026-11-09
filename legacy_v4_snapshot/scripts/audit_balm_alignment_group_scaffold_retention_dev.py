"""Audit training retention for Smith-Waterman-alignment group + Murcko core-cold folds."""
import hashlib,json,numpy as np
from collections import Counter
from sklearn.model_selection import GroupKFold
from dynacpi.balm_public_data_dev import ROOT,get_train_only
from scripts.train_balm_nested_scaffold_cold_mean_shrinkage_dev import scaffold_key,scaffold_filter
from scripts.train_balm_bindingdb_morgan_protein_ridge_groupcv_dev import collapse_training

def main():
    source=ROOT/"research/balm_train_only_smith_waterman_protein_group_audit_20261108.json"
    audit=json.loads(source.read_text(encoding="utf-8"))
    raw=get_train_only()
    mapping={}
    for name in ("identity30_cover80","identity40_cover80"):
        group_ids=audit["sequence_sha256_to_alignment_groups"][name]
        if not mapping:
            mapping={s:hashlib.sha256(s.encode()).hexdigest() for s in set(r["protein"] for r in raw)}
        rows=[dict(r,group=group_ids[mapping[r["protein"]]]) for r in raw]
        scaffold={s:scaffold_key(s) for s in set(r["smiles"] for r in rows)}
        groups=np.asarray([r["group"] for r in rows])
        print("METHOD",name,"components",len(set(groups)),flush=True)
        for fold,(it,iv) in enumerate(GroupKFold(4).split(groups,groups=groups),1):
            held=[rows[int(i)] for i in iv]
            retained=scaffold_filter([rows[int(i)] for i in it],held,scaffold)
            fit=collapse_training(retained)
            assert not {r["group"] for r in fit}&{r["group"] for r in held}
            print("outer",fold,"heldout_rows",len(held),"train_before_exclusion",len(it),
                  "train_after_scaffold",len(retained),"unique_fit",len(fit),
                  "heldout_groups",len(set(r["group"] for r in held)),flush=True)
            ing=np.asarray([r["group"] for r in fit])
            for k,(tr,va) in enumerate(GroupKFold(3).split(ing,groups=ing),1):
                held2=[fit[int(i)] for i in va]
                inner=scaffold_filter([fit[int(i)] for i in tr],held2,scaffold)
                print("  inner",k,"unique_fit",len(collapse_training(inner)),flush=True)
if __name__=="__main__":main()
