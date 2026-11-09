import sys,random
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from dynacpi.evaluation_axes import plan_axes,summarize
rows=[dict(task_id=f"t{i}",protein_component_id=f"p{i//2}",compound_id=f"c{j}",scaffold_id=f"s{j//2}") for i in range(60) for j in range(60)]
a=plan_axes(rows)
shuffled=rows.copy();random.Random(19).shuffle(shuffled)
assert a==plan_axes(shuffled)
report=summarize(rows,a)
assert report["dual"]["excluded_cross_axis_pairs"]>0
assert sum(x["pairs"] for x in report["protein"]["roles"].values())==len(rows)
assert report["protein"]["assignment_sha256"]==summarize(shuffled,plan_axes(shuffled))["protein"]["assignment_sha256"]
for bad in ([dict(rows[0],y=7.2)],[rows[0],rows[0]],
            [rows[0],dict(rows[1],protein_component_id="other")],
            [rows[0],dict(rows[60],scaffold_id="conflict")]):
    try:plan_axes(bad)
    except ValueError:pass
    else:raise AssertionError("Accepted invalid metadata")
# Acyclic scaffold group is indivisible, not split into artificial pseudo-scaffolds.
acyclic=[dict(r,scaffold_id="ACYCLIC") for r in rows]
assigned=plan_axes(acyclic)["scaffold"]
assert len(set(assigned.values()))==1
print("PASS order invariance, component/scaffold disjointness, dual exclusion, label rejection, identity conflicts")

# Independently corrupt a valid plan: diagnostics must detect leakage.
corrupt={mode:dict(mapping) for mode,mapping in a.items()}
key=(rows[0]['task_id'],rows[0]['compound_id'])
old=corrupt['protein'][key]
corrupt['protein'][key]='test' if old!='test' else 'base_train'
try:summarize(rows,corrupt)
except AssertionError:pass
else:raise AssertionError('Missed deliberately introduced protein leakage')
assert sum(x['pairs'] for x in report['dual']['roles'].values())+report['dual']['excluded_cross_axis_pairs']==len(rows)
print('PASS deliberate leakage detection and dual pair conservation')
