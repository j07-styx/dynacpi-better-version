"""Build source-access evidence ledger WITHOUT affinity labels or held-out outcomes.

- Input: seven-family private metadata review queue made before this script.
- Review-tier classifications are HUMAN-READ source-access notes from public
  bibliographic, publisher index and patent method pages; NOT verified assay rows.
- Private output retains source-family IDs / role memberships. Public output
  reveals only aggregate evidence readiness counts and no group assignments.
- No synthetic "APPROVED" state is emitted or accepted.
"""
from __future__ import annotations
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "data/audit/PROVISIONAL_SOURCE_FAMILY_EVIDENCE_QUEUE_PRIVATE_20261108.json"
LOCAL_OUT = ROOT / "data/audit/SOURCE_FAMILY_PRIMARY_ACCESS_LEDGER_PRIVATE_20261108.json"
PUBLIC_OUT = ROOT / "research/bindingdb_seven_family_primary_access_readiness_20261108.json"

# Each marker is a PREEXISTING metadata DOI/patent or PMID already observed.
# Access tiers are manually checked source-level observations, not task/row QC.
PRIMARY_SOURCE_NOTES = {
    "doi:10.1016/j.ejmech.2019.111656": {
        "tier": "SECONDARY_ASSAY_ANNOTATION",
        "accessible_evidence": "Public assay annotation describes a tagged, truncated experimental protein fragment",
        "remaining_gap": "Primary source methods, original measured table and exact construct/record alignment",
    },
    "doi:10.1016/j.bmc.2013.02.021": {
        "tier": "ARTICLE_ABSTRACT_ONLY",
        "accessible_evidence": "Article abstract distinguishes physical binding and other assay endpoints",
        "remaining_gap": "Full methods, original Kd table, actual construct and exact BindingDB row mapping",
    },
    "doi:10.1016/j.bmcl.2012.08.001": {
        "tier": "ARTICLE_ABSTRACT_ONLY",
        "accessible_evidence": "Article bibliographic identity available; multiple target/assay semantics unresolved",
        "remaining_gap": "Original table, physical construct and direct-vs-indirect method verification",
    },
    "doi:10.1016/j.ejmech.2012.02.050": {
        "tier": "ARTICLE_ABSTRACT_ONLY",
        "accessible_evidence": "Abstract discusses two distinct binding-characterization methods",
        "remaining_gap": "Distinguish exact Kd from proxy signal for each original source row",
    },
    "doi:10.1021/acs.jmedchem.0c01025": {
        "tier": "PUBLISHER_SUPPLEMENT_INDEX",
        "accessible_evidence": "Publisher's freely listed supplementary information sections identify separate binding methods and protein purification",
        "remaining_gap": "Open and examine specific supplementary tables, construct and source row assignments",
    },
    "doi:10.1021/acs.jmedchem.1c01693": {
        "tier": "ARTICLE_ABSTRACT_ONLY",
        "accessible_evidence": "Article abstract distinguishes binding measurement from screening and competition",
        "remaining_gap": "Original method/table and assayed construct per BindingDB source row",
    },
    "patent:us10087187": {
        "tier": "PRIMARY_PATENT_TABLE_METHOD",
        "accessible_evidence": "Original patent contains Table A with multi-target binding Kd columns, inequality qualifiers and missing entries",
        "remaining_gap": "Check exact source-row, assay target, censoring and construct for each retained BindingDB record",
    },
}
TIERS = frozenset({
    "ARTICLE_ABSTRACT_ONLY",
    "SECONDARY_ASSAY_ANNOTATION",
    "PUBLISHER_SUPPLEMENT_INDEX",
    "PRIMARY_PATENT_TABLE_METHOD",
})
REVIEW_FIELDS = (
    "primary_original_row_matched", "primary_endpoint_verified",
    "assayed_protein_construct_verified", "measurement_qualifiers_verified",
    "independent_reviewer_approved",
)


def classify_queue(data: dict) -> tuple[list[dict], dict]:
    if data.get("status") != "PRIVATE_HUMAN_REVIEW_QUEUE_UNAPPROVED":
        raise PermissionError("Expected unreviewed private source queue")
    families = data.get("families")
    if not isinstance(families, list) or len(families) != 7:
        raise ValueError("Expected exactly seven source conflicts")
    seen = set()
    output = []
    tiers = Counter()
    for family in families:
        ident = family.get("source_family_sha256")
        if not isinstance(ident, str) or len(ident) != 64 or ident in seen:
            raise ValueError("Invalid or duplicated private source-family identifier")
        seen.add(ident)
        markers = set(family.get("source_identifier_examples_only_not_exhaustive") or ())
        for lead in family.get("preexisting_ranked_source_leads", []):
            for k, values in (
                ("doi", lead["source_doi_candidates"]),
                ("pmid", lead["source_pmid_candidates"]),
                ("patent", lead["source_patent_candidates"]),
            ):
                markers.update(k+":"+str(v).strip().lower() for v in values if v)
        # The mapping below is only an evidence-access marker. It cannot prove
        # this one source defines every task in a transitive source family.
        matches = sorted(markers & PRIMARY_SOURCE_NOTES.keys())
        if len(matches) != 1:
            raise ValueError("One independently recorded evidence anchor required per family")
        marker = matches[0]
        source = PRIMARY_SOURCE_NOTES[marker]
        if source["tier"] not in TIERS:
            raise ValueError("Unknown evidence tier")
        if family.get("review_status") != "UNREVIEWED_METADATA_CONFLICT_NOT_APPROVED":
            raise PermissionError("Family has changed from unreviewed state")
        fields = {k: False for k in REVIEW_FIELDS}
        out = {
            "source_family_sha256": ident,
            "provisional_role_component_counts_PRIVATE": family["affected_provisional_role_component_counts"],
            "single_example_source_identifier": marker,
            "identifier_coverage_note": "One example only; complete source linkage NOT reviewed",
            "source_access_tier": source["tier"],
            "source_access_note_NOT_EXPERIMENT_VERIFICATION": source["accessible_evidence"],
            "source_review_gap": source["remaining_gap"],
            **fields,
            "status": "UNREVIEWED_NO_SOURCE_ROW_APPROVAL",
        }
        tiers[source["tier"]] += 1
        output.append(out)
    if sum(tiers.values()) != 7:
        raise AssertionError("Source family coverage incomplete")
    public = {
        "status": "SOURCE_ACCESS_TRIAGE_ONLY_NOT_SCIENTIFIC_APPROVAL",
        "families_in_previously_identified_split_conflict": len(output),
        "families_with_one_source_access_evidence_anchor": len(output),
        "evidence_access_tier_counts": dict(sorted(tiers.items())),
        "families_with_independently_verified_original_records": 0,
        "families_with_independently_verified_protein_constructs": 0,
        "families_scientifically_approved": 0,
        "note": "A source method table, article abstract, publisher SI index or assay annotation is not an original BindingDB record crosswalk. Source-family examples do not guarantee complete source lineage. Full assay and construct review remains outstanding. No reserved-test labels or model outcomes accessed.",
        "source_selection": "Existing private seven-family metadata queue only, with separately annotated public source-access tiers",
        "scientific_gate": "BLOCKED_UNAPPROVED_NOT_TRAINING_READY",
    }
    return output, public


def main() -> None:
    if LOCAL_OUT.exists() or PUBLIC_OUT.exists():
        raise FileExistsError("Do not overwrite established source evidence triage")
    raw = INPUT.read_bytes()
    data = json.loads(raw.decode("utf-8-sig"))
    local, public = classify_queue(data)
    ledger = {
        "status": "PRIVATE_SOURCE_ACCESS_REVIEW_REQUIRED",
        "input_private_queue_sha256": sha256(raw).hexdigest(),
        "families": local,
    }
    LOCAL_OUT.parent.mkdir(parents=True, exist_ok=True)
    with LOCAL_OUT.open("x", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2)
        f.write("\n")
    PUBLIC_OUT.parent.mkdir(parents=True, exist_ok=True)
    with PUBLIC_OUT.open("x", encoding="utf-8") as f:
        json.dump(public, f, indent=2)
        f.write("\n")
    print(json.dumps(public, indent=2))


if __name__ == "__main__":
    main()
