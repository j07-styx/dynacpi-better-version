"""DynaCPI v4 — technical-only lineage audit of 30 UNREVIEWED BindingDB tasks.

Verifies grouped-compound -> unaggregated candidate -> original source reactant
identity, exact numeric Kd to pKd conversion, and median/range aggregation.
Does not verify experimental meaning of Kd, approve source constructs, train
models or create any scientific cohort/split. Never exports per-compound
affinity values, individual raw source IDs or chemical structures.
"""
from __future__ import annotations
from collections import defaultdict, Counter
from pathlib import Path
from datetime import datetime, timezone
import csv, hashlib, json, math, re, statistics, argparse

FILES = {
    "review": "bindingdb_assay_manual_review_queue_30_ENRICHED.csv",
    "median": "bindingdb_full_kd_compound_median_candidates.jsonl",
    "candidate": "bindingdb_full_kd_candidates.jsonl",
    "source": "bindingdb_full_kd_source_provenance.jsonl",
}
NUMBER = re.compile(r"^\+?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?$")

def file_sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def records(path):
    with path.open("r", encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if line.strip():
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Malformed JSONL {path.name}:{n}") from exc

def positive_kd(x):
    value = str(x).strip()
    if not NUMBER.fullmatch(value):
        return None
    result = float(value)
    return result if math.isfinite(result) and result > 0 else None

def rid(x):
    text = str(x).strip()
    if not re.fullmatch(r"[0-9]{1,18}", text):
        raise ValueError("Invalid BindingDB reactant identifier")
    return text

def load_review(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 30 or len({r["task_id"] for r in rows}) != 30:
        raise ValueError("Expected exactly 30 distinct review tasks")
    if any(r.get("curation_decision") != "UNREVIEWED" for r in rows):
        raise PermissionError("Human review status changed; fixed-queue audit refuses to run")
    return {r["task_id"]: r for r in rows}

def audit(root):
    root = Path(root).resolve()
    a = root / "data" / "audit"
    paths = {k: a/v for k, v in FILES.items()}
    for name, p in paths.items():
        if not p.is_file():
            raise FileNotFoundError(f"Missing {name}: {p}")
    before = file_sha(paths["review"])
    reviews = load_review(paths["review"])
    groups = defaultdict(list)
    reactant_task = {}
    for grp in records(paths["median"]):
        task = grp["task_id"]
        if task not in reviews:
            continue
        groups[task].append(grp)
        for x in grp["reactant_ids"]:
            key = rid(x)
            if key in reactant_task:
                raise ValueError("Same reactant in more than one grouped observation")
            reactant_task[key] = task
    if set(groups) != set(reviews):
        raise ValueError("Fixed review tasks missing compound groups")
    source, candidate = {}, {}
    dup_source, dup_candidate = 0, 0
    for r in records(paths["source"]):
        key = str(r.get("BindingDB Reactant_set_id", ""))
        if key in reactant_task:
            if key in source: dup_source += 1
            else: source[key] = r
    for r in records(paths["candidate"]):
        key = str(r.get("reactant_set_id", ""))
        if key in reactant_task:
            if key in candidate: dup_candidate += 1
            else: candidate[key] = r
    overall = Counter()
    task_results = []
    for task, meta in sorted(reviews.items()):
        bucket = groups[task]
        ids = [rid(x) for g in bucket for x in g["reactant_ids"]]
        count = Counter()
        doi, pmid, patent = set(), set(), set()
        org, curators = Counter(), Counter()
        for key in ids:
            raw = source.get(key)
            part = candidate.get(key)
            if raw is None:
                count["SOURCE_RECORD_MISSING"] += 1
                continue
            count["SOURCE_RECORD_FOUND"] += 1
            kd = positive_kd(raw.get("Kd (nM)", ""))
            if kd is None: count["RAW_KD_NOT_STRICT_POSITIVE_NUMERIC"] += 1
            else: count["RAW_KD_STRICT_POSITIVE_NUMERIC"] += 1
            if any(str(raw.get(k, "")).strip() for k in ("Ki (nM)", "IC50 (nM)", "EC50 (nM)")):
                count["MULTIPLE_ENDPOINT_COLUMNS_POPULATED"] += 1
            chains = str(raw.get("Number of Protein Chains in Target (>1 implies a multichain complex)", "")).strip()
            if chains != "1": count["RAW_MULTI_OR_UNSPECIFIED_CHAIN_COUNT"] += 1
            if raw.get("protein_construct_sequence_sha256") != meta["protein_sha256"]:
                count["RAW_TARGET_SEQUENCE_HASH_MISMATCH"] += 1
            if part is None: count["UNAGGREGATED_CANDIDATE_MISSING"] += 1
            else:
                if part.get("assay_key") != meta["assay_key"]:
                    count["UNAGGREGATED_ASSAY_KEY_MISMATCH"] += 1
                if hashlib.sha256(part.get("sequence", "").encode("utf-8")).hexdigest() != meta["protein_sha256"]:
                    count["UNAGGREGATED_PROTEIN_HASH_MISMATCH"] += 1
                if kd is not None and abs(float(part["pKd"]) - (9 - math.log10(kd))) > 1e-7:
                    count["RAW_KD_TO_PK_D_CONVERSION_MISMATCH"] += 1
            for field, values in (("Article DOI", doi), ("PMID", pmid), ("Patent Number", patent)):
                v = str(raw.get(field, "")).strip()
                if v: values.add(v.casefold() if field == "Article DOI" else v)
            v = str(raw.get("Target Source Organism According to Curator or DataSource", "")).strip()
            if v: org[v] += 1
            v = str(raw.get("Curation/DataSource", "")).strip()
            if v: curators[v] += 1
        for g in bucket:
            rids = [rid(x) for x in g["reactant_ids"]]
            if len(rids) > 1: count["COMPOUND_GROUPS_WITH_REPLICATES"] += 1
            if len(rids) != int(g["replicates"]):
                count["REPLICATE_NUMBER_DISAGREES_WITH_REACTANT_IDS"] += 1
            vs = [float(candidate[key]["pKd"]) for key in rids if key in candidate]
            if len(vs) == len(rids):
                if abs(statistics.median(vs) - float(g["median_pKd"])) > 1e-7:
                    count["GROUP_MEDIAN_PK_D_MISMATCH"] += 1
                if abs((max(vs) - min(vs)) - float(g["range_pKd"])) > 1e-7:
                    count["GROUP_RANGE_PK_D_MISMATCH"] += 1
        if len(ids) != int(meta["record_count"]):
            count["REVIEW_RECORD_COUNT_MISMATCH"] += 1
        if len(bucket) != int(meta["unique_compounds"]):
            count["REVIEW_COMPOUND_COUNT_MISMATCH"] += 1
        def split_ids(s, lower=False):
            vals = [x.strip() for x in re.split(r"[;|]", str(s)) if x.strip()]
            return {x.casefold() for x in vals} if lower else set(vals)
        if split_ids(meta["article_dois"], True) and doi != split_ids(meta["article_dois"], True):
            count["SOURCE_DOI_SET_DISAGREES_WITH_REVIEW"] += 1
        if split_ids(meta["pmids"]) and pmid != split_ids(meta["pmids"]):
            count["SOURCE_PMID_SET_DISAGREES_WITH_REVIEW"] += 1
        if split_ids(meta["patent_numbers"]) and {x.upper() for x in patent} != {x.upper() for x in split_ids(meta["patent_numbers"])}:
            count["SOURCE_PATENT_SET_DISAGREES_WITH_REVIEW"] += 1
        overall.update(count)
        task_results.append({
            "task_id": task, "assay_key": meta["assay_key"],
            "target_name": meta["target_names"],
            "compound_group_count": len(bucket),
            "referenced_source_records": len(ids),
            "source_record_set_sha256": hashlib.sha256("|".join(sorted(ids)).encode()).hexdigest(),
            "findings": dict(sorted(count.items())),
            "organism_metadata_counts": dict(org),
            "curation_metadata_counts": dict(curators),
            "source_identifier_unique_counts": {"doi":len(doi),"pmid":len(pmid),"patent":len(patent)},
            "review_state":"UNREVIEWED",
            "physical_assayed_construct_verified":False,
            "experimentally_exact_kd_verified":False,
            "scientifically_approved":False
        })
    critical_names = (
        "SOURCE_RECORD_MISSING", "UNAGGREGATED_CANDIDATE_MISSING",
        "RAW_TARGET_SEQUENCE_HASH_MISMATCH", "UNAGGREGATED_PROTEIN_HASH_MISMATCH",
        "UNAGGREGATED_ASSAY_KEY_MISMATCH", "RAW_KD_TO_PK_D_CONVERSION_MISMATCH",
        "REPLICATE_NUMBER_DISAGREES_WITH_REACTANT_IDS", "GROUP_MEDIAN_PK_D_MISMATCH",
        "GROUP_RANGE_PK_D_MISMATCH", "REVIEW_RECORD_COUNT_MISMATCH",
        "REVIEW_COMPOUND_COUNT_MISMATCH")
    if file_sha(paths["review"]) != before:
        raise PermissionError("Review CSV changed during audit")
    report = {
        "status":"TECHNICAL_SOURCE_LINEAGE_AUDIT_NOT_SCIENTIFIC_APPROVAL",
        "generated_utc":datetime.now(timezone.utc).isoformat(),
        "review_queue_sha256":before,
        "source_file_sha256":file_sha(paths["source"]),
        "candidate_file_sha256":file_sha(paths["candidate"]),
        "median_file_sha256":file_sha(paths["median"]),
        "tasks":len(task_results),
        "compound_groups":sum(len(x) for x in groups.values()),
        "source_reactant_references":len(reactant_task),
        "original_reactant_ids_found":len(source),
        "unaggregated_reactant_ids_found":len(candidate),
        "duplicate_source_id_rows":dup_source,
        "duplicate_unaggregated_id_rows":dup_candidate,
        "summary_counts":dict(sorted(overall.items())),
        "critical_lineage_flags":{k:overall[k] for k in critical_names if overall[k]},
        "approved_tasks":0,
        "source_biological_review_completed":False,
        "main_training_permitted":False,
        "per_task":task_results,
        "limitations":[
            "Technical Kd/pKd conversion consistency is not source verification of a physical Kd experiment.",
            "No original source assay/construct has been adjudicated; no task is approved.",
            "No raw Kd or pKd, individual reactant IDs, SMILES or chemical inputs are exported.",
            "Do not construct a scientific training cohort or frozen group split from this report."
        ]
    }
    target = root / "research" / "source_access_v4"
    target.mkdir(parents=True, exist_ok=True)
    (target / "record_level_provenance_30.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    md = [
        "# BindingDB 30-task original-record lineage check", "",
        "**Technical consistency only — zero source-approved assays.**", "",
        f"Tasks: {report['tasks']}; compound groups: {report['compound_groups']}; "
        f"source reactant IDs: {report['source_reactant_references']}.", "",
        "| Technical check | Count |", "| --- | ---: |"
    ]
    for k,v in report["summary_counts"].items():
        md.append(f"| {k} | {v} |")
    md += ["", "## Critical technical flags", "",
           "~~~json", json.dumps(report["critical_lineage_flags"],indent=2), "~~~", "",
           "A matched number is NOT proof that a source experiment measured true equilibrium Kd.",
           "All thirty candidate tasks remain UNREVIEWED, and the scientific cohort is blocked."]
    (target / "RECORD_LEVEL_PROVENANCE_30.md").write_text("\n".join(md)+"\n",encoding="utf-8")
    return report

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    args=p.parse_args()
    report=audit(args.root)
    print(json.dumps({k:report[k] for k in (
        "status","tasks","compound_groups","source_reactant_references",
        "original_reactant_ids_found","unaggregated_reactant_ids_found",
        "duplicate_source_id_rows","duplicate_unaggregated_id_rows",
        "critical_lineage_flags","approved_tasks","main_training_permitted"
    )},indent=2))
