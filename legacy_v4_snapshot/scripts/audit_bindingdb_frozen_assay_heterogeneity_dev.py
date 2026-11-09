"""Audit assay heterogeneity, source evidence and within-target conflicts.

Scopes all reported statistics to TRAIN and INNER-DEV only.
Prospective test is not summarized or used for architecture selection.
"""
import json
import statistics
from collections import defaultdict, Counter
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
A=ROOT/"data"/"audit"
OUT=ROOT/"research"/"bindingdb_frozen_inner_dev_assay_heterogeneity_audit.json"

def read(path):
    with path.open(encoding="utf8") as f:
        for line in f:yield json.loads(line)

def main():
    m=json.loads((ROOT/"research"/"bindingdb_frozen_component_split_manifest_dev.json").read_text())
    tr=set(m["train_components"]);dv=set(m["inner_dev_components"]);test=set(m["test_components"])
    assert tr.isdisjoint(dv|test) and dv.isdisjoint(test)
    cmap={}
    for record in read(A/"bindingdb_full_kd_protein_components.jsonl"):
        for seq in record["members"]:cmap[seq]=record["component_id"]
    allowed={(r["sequence_sha256"],r["assay_key"]) for r in
             read(A/"bindingdb_full_kd_task_summary.jsonl")
             if r["unique_canonical_compounds"]>=15}
    by_partition={"train":[],"inner_dev":[]}
    for r in read(A/"bindingdb_full_kd_compound_median_candidates.jsonl"):
        if (r["protein_seq_sha256"],r["assay_key"]) not in allowed or r["range_pKd"]>.1:continue
        g=cmap[r["protein_seq_sha256"]]
        if g in tr:by_partition["train"].append(r)
        elif g in dv:by_partition["inner_dev"].append(r)
    assert len(by_partition["train"])==m["train_pairs"] and len(by_partition["inner_dev"])==m["inner_dev_pairs"]

    def audit(records, label):
        bygroup=defaultdict(list);bypair=defaultdict(list);byassay=defaultdict(list)
        for r in records:
            bygroup[cmap[r["protein_seq_sha256"]]].append(r)
            byassay[r["assay_key"]].append(r)
            bypair[(r["protein_seq_sha256"],r["compound_sha256"])].append(r)
        comps=[]
        for c,rr in bygroup.items():
            assays=Counter(r["assay_key"] for r in rr)
            yy=np.asarray([r["median_pKd"] for r in rr])
            comps.append({"component_id":c,"pairs":len(rr),
                    "protein_sequences":len({r["protein_seq_sha256"] for r in rr}),
                    "assay_keys":len(assays),
                    "dominant_assay_share":float(max(assays.values())/len(rr)),
                    "top_assay_counts":assays.most_common(5),
                    "compounds":len({r["compound_sha256"] for r in rr}),
                    "label_mean":float(np.mean(yy)),
                    "label_sd":float(np.std(yy)),
                    "label_q05_q50_q95":np.quantile(yy,[.05,.5,.95]).tolist()})
        comps.sort(key=lambda x:x["pairs"],reverse=True)
        repeated=[rr for rr in bypair.values() if len(rr)>1]
        crossassay=[rr for rr in repeated if len({r["assay_key"] for r in rr})>1]
        ranges=[max(x["median_pKd"] for x in rr)-min(x["median_pKd"] for x in rr)
                for rr in crossassay]
        multiple_assay_records=sum(len(rr) for rr in crossassay)
        out={"pairs":len(records),"components":len(comps),
             "unique_assay_keys":len(byassay),
             "unique_protein_sequences":len({r["protein_seq_sha256"] for r in records}),
             "unique_compounds":len({r["compound_sha256"] for r in records}),
             "largest_component_pair_share":comps[0]["pairs"]/len(records),
             "top_3_component_pair_share":sum(x["pairs"] for x in comps[:3])/len(records),
             "components_with_more_than_one_assay":sum(x["assay_keys"]>1 for x in comps),
             "sequence_compound_pairs_measured_multiple_assays":len(crossassay),
             "records_in_multiple_assay_pair_groups":multiple_assay_records,
             "fraction_records_in_multiple_assay_pair_groups":multiple_assay_records/len(records),
             "cross_assay_pKd_range_q50_q90_q99":np.quantile(ranges,[.5,.9,.99]).tolist() if ranges else None,
             "cross_assay_pairs_spread_gt_0_5":sum(v>.5 for v in ranges),
             "cross_assay_pairs_spread_gt_1_0":sum(v>1 for v in ranges),
             "cross_assay_pairs_spread_gt_2_0":sum(v>2 for v in ranges),
             "component_summary":comps}
        return out
    tr_report=audit(by_partition["train"],"train")
    dev_report=audit(by_partition["inner_dev"],"inner_dev")
    out={"status":"PROVISIONAL_DATA_QUALITY_AUDIT_FROZEN_TRAIN_AND_INNER_DEV_ONLY",
         "train":tr_report,"inner_dev":dev_report,
         "global_label_mean_train":float(np.mean([r["median_pKd"] for r in by_partition["train"]])),
         "global_label_mean_dev":float(np.mean([r["median_pKd"] for r in by_partition["inner_dev"]])),
         "warnings":["The source assay_key is not a standardized assay-condition identifier.",
                     "The same target/compound across distinct assay keys need not be experimentally comparable.",
                     "All metrics are on already-inspected dev or training; no prospective-test statistics.",
                     "No full-text assay/construct verification performed."]}
    OUT.write_text(json.dumps(out,indent=2))
    for k in ("train","inner_dev"):
        r=out[k]
        print(json.dumps({"partition":k,
          "pairs":r["pairs"],"components":r["components"],
          "assay_keys":r["unique_assay_keys"],
          "components_with_multiple_assays":r["components_with_more_than_one_assay"],
          "largest_cluster_share":r["largest_component_pair_share"],
          "target_compound_pairs_in_multiple_assays":r["sequence_compound_pairs_measured_multiple_assays"],
          "cross_assay_spread_gt1":r["cross_assay_pairs_spread_gt_1_0"],
          "range_pKd_quantiles":r["cross_assay_pKd_range_q50_q90_q99"],
          "largest_cluster":r["component_summary"][0]}),flush=True)

if __name__=="__main__":
    main()
