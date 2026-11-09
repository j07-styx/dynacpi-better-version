"""Label-blind DOI/PMID/patent source-family leakage preflight for the PROVISIONAL BindingDB split.

Only reads already published metadata linkage JSON and provisional role-manifest JSON.
Never reads assay observations, chemical identities, predictions, or reserved test outcomes.
This cannot approve sources or freeze a new scientific split.
"""
from __future__ import annotations
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FAMILY_PATH = ROOT / "research/bindingdb_publication_connected_protein_review_preflight_20261108.json"
SPLIT_PATH = ROOT / "research/bindingdb_frozen_component_split_manifest_dev.json"
OUT = ROOT / "research/bindingdb_provisional_role_source_overlap_audit_20261108.json"
ROLE_NAMES = ("train", "inner_dev", "test")
ROLE_KEYS = ("train_components", "inner_dev_components", "test_components")


def assess_roles(roles: dict[str, list[str]], family_by_component: dict[str, str]) -> dict:
    """Pure calculation using component and source-family metadata; never labels."""
    if tuple(roles) != ROLE_NAMES:
        raise ValueError("Three predeclared role names and order required")
    seen = set()
    fam_roles = defaultdict(lambda: defaultdict(int))
    per_role = {}
    for role in ROLE_NAMES:
        components = roles[role]
        if len(components) != len(set(components)):
            raise ValueError("Repeated protein component inside a role")
        repeated = set(components) & seen
        if repeated:
            raise ValueError("Protein component appears in multiple roles")
        unknown = set(components) - set(family_by_component)
        if unknown:
            raise ValueError("Some role components lack publication-family mapping")
        seen.update(components)
        for component in components:
            fam_roles[family_by_component[component]][role] += 1
        per_role[role] = {
            "components": len(components),
            "distinct_source_families": len({family_by_component[x] for x in components}),
        }
    mixed = [counts for counts in fam_roles.values() if len(counts) > 1]
    paired = {}
    for i, a in enumerate(ROLE_NAMES):
        for b in ROLE_NAMES[i+1:]:
            crossed = [counts for counts in fam_roles.values() if a in counts and b in counts]
            paired[f"{a}_vs_{b}"] = {
                "families_in_both": len(crossed),
                "role_a_components_in_shared_families": sum(v[a] for v in crossed),
                "role_b_components_in_shared_families": sum(v[b] for v in crossed),
            }
    patterns = Counter("+".join(role for role in ROLE_NAMES if role in counts)
                       for counts in fam_roles.values())
    return {
        "role_counts": per_role,
        "families_total_among_role_components": len(fam_roles),
        "families_crossing_two_or_more_roles": len(mixed),
        "components_in_cross_role_families": sum(sum(c.values()) for c in mixed),
        "cross_role_pattern_counts": dict(sorted(patterns.items())),
        "pairwise_overlap": paired,
        "publication_family_disjoint": not mixed,
        "readiness": ("NO_METADATA_CROSS_ROLE_SOURCE_LINKS_DETECTED"
                      if not mixed else "PROVISIONAL_SPLIT_SOURCE_LINKS_CROSS_ROLES"),
    }


def file_hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def generate(family_path=FAMILY_PATH, split_path=SPLIT_PATH) -> dict:
    source = json.loads(Path(family_path).read_text(encoding="utf-8-sig"))
    split = json.loads(Path(split_path).read_text(encoding="utf-8-sig"))
    if source.get("status") != "QUARANTINED_LABEL_BLIND_PUBLICATION_LINKED_PROTEIN_COMPONENTS_NO_SPLIT_OR_APPROVAL":
        raise ValueError("Unexpected source-linkage preflight status")
    if source.get("scientific_gate") != "BLOCKED_UNAPPROVED_NOT_TRAINING_READY":
        raise ValueError("Source-linkage scientific gate changed")
    mapping = source["linkage"]["source_family_by_protein_component"]
    roles = {role: split[key] for role, key in zip(ROLE_NAMES, ROLE_KEYS)}
    checks = assess_roles(roles, mapping)
    result = {
        "scope": "Provisional component/source-metadata split-leakage preflight only",
        "scientific_gate": "BLOCKED_UNAPPROVED_NOT_TRAINING_READY",
        "data_access": "Only source-family preflight metadata JSON and split role-manifest JSON; no affinity labels, protein/chemical sequences, prediction values or test outcomes",
        "original_source_linkage_sha256": file_hash(Path(family_path)),
        "original_provisional_split_sha256": file_hash(Path(split_path)),
        **checks,
        "caveat": "DOI/PMID/patent links do not establish assay identity. Missing source IDs can hide overlap. Protein homology and construct identity require independent review. No role changes or approvals performed.",
    }
    return result


def main():
    if OUT.exists():
        raise FileExistsError(f"Refusing to overwrite immutable preflight: {OUT}")
    result = generate()
    OUT.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
