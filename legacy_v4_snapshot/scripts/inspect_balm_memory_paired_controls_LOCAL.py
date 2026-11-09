import json
from pathlib import Path
root=Path("research")
a="ensemble_plus_memory_joint_esm8_k64"
b="ensemble_plus_memory_chemical_only_k64"
c="ensemble_plus_memory_joint_esm16_k64"
for rule in ("identity30_cover80","identity40_cover80"):
    x=json.loads((root/f"balm_joint_similarity_memory_sw_{rule}_scaffold_cold_traincv_dev_paired_controls.json").read_text())
    boot=x["out_of_fold"]["bootstrap"]["comparisons"]
    print(rule)
    for comparison in (a+"_minus_"+b,a+"_minus_train_mean",c+"_minus_"+b):
        print(comparison,boot[comparison])
