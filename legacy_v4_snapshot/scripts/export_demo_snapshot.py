"""Create sanitized, standalone GitHub handoff staging tree.

NEVER modifies the original PerceiverCPI clone, experimental data or sources.
Uses an explicit file inclusion policy. Never follows symlinks. Excludes
credentials, virtual environments, private caches, giant source archives,
all published test results that are not actual measurements, and artifacts
larger than 49 MiB except already declared experiment checkpoints (smaller).
"""
from pathlib import Path
import fnmatch
import hashlib
import json
import os
import re
import shutil

SRC=Path(r"C:\Users\J0124\dynacpi_v4")
DEST=Path(r"C:\Users\J0124\Demo-smth")
if not SRC.is_dir() or not (SRC/"project.md").exists():
    raise RuntimeError("Original DynaCPI v4 workspace missing")
if DEST.exists() and any(DEST.iterdir()):
    raise RuntimeError("Refusing to overwrite existing nonempty destination "+str(DEST))
DEST.mkdir(parents=True,exist_ok=True)

root_files=[
 "LICENSE.txt","project.md","environment.yml","run.sh","train.py","predict.py",
 "prepare_data_for_setting.py","hyperparameter_optimization.py"
]
allowed_source_ext={".py",".yml",".yaml",".sh",".md",".txt",".json",".csv",".tsv",".ipynb"}
included=[]
excluded=[]
def digest(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda:f.read(4*1024*1024),b""):
            h.update(chunk)
    return h.hexdigest()
def copy_rel(rel,reason):
    f=SRC/rel
    if not f.is_file():
        raise FileNotFoundError(f)
    if f.is_symlink():
        raise RuntimeError("Refusing symlink "+str(f))
    target=DEST/rel
    target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(f,target)
    included.append({"path":rel.as_posix(),"size":f.stat().st_size,
                     "sha256":digest(f),"group":reason})
def add_file(text,reason):
    copy_rel(Path(text),reason)

for root in root_files:
    add_file(root,"upstream-perceivercpi-foundation")
# Preserve upstream README under its real original name to avoid overriding
# the handoff entrypoint README.md; also keep full MIT attribution.
copy_rel(Path("README.md"),"upstream-original-readme")
(DEST/"README.md").rename(DEST/"UPSTREAM_README.md")
included[-1]["path"]="UPSTREAM_README.md"
for folder in ("chemprop","deeppupose","toy_dataset"):
    d=SRC/folder
    if not d.is_dir():continue
    for file in sorted(d.rglob("*")):
        if not file.is_file() or file.is_symlink():continue
        rel=file.relative_to(SRC)
        if "__pycache__" in rel.parts or file.suffix.lower() in (".pyc",".pyo",".dll",".exe"):
            continue
        if file.suffix.lower() not in allowed_source_ext:
            excluded.append({"path":rel.as_posix(),"reason":"upstream_non_source_or_binary"})
            continue
        if file.stat().st_size>12*1024*1024:
            excluded.append({"path":rel.as_posix(),"reason":"upstream_asset_exceeds_12MiB"})
            continue
        copy_rel(rel,"upstream-source")
for folder in ("dynacpi","scripts","tests_v4"):
    d=SRC/folder
    if not d.is_dir():raise FileNotFoundError(d)
    for file in sorted(d.rglob("*")):
        if not file.is_file() or file.is_symlink():continue
        rel=file.relative_to(SRC)
        if "__pycache__" in rel.parts or file.suffix.lower() in (".pyc",".pyo"):
            continue
        if file.suffix.lower() not in (".py",".md",".txt",".json",".yml",".yaml",".sh"):
            excluded.append({"path":rel.as_posix(),"reason":"generated_non_source"})
            continue
        if rel.name=="create_github_demo_handoff_repo.py":
            # One-off account operation, not part of the research code.
            excluded.append({"path":rel.as_posix(),"reason":"one_time_repo_admin_script"})
            continue
        copy_rel(rel,"v4-source-tests-scripts")

for file in sorted((SRC/"research").rglob("*")):
    if not file.is_file() or file.is_symlink():continue
    rel=file.relative_to(SRC)
    if "__pycache__" in rel.parts:continue
    if file.suffix.lower() in (".md",".json",".log"):
        copy_rel(rel,"research-evidence-and-logs")
    elif file.suffix.lower() in (".pt",".pth"):
        # Exact 3 pilot checkpoints used in the code and comparisons, < 20 MiB each
        expected={
          "research/pilot_sequence/checkpoint.pt",
          "research/pilot_perceiver/checkpoint.pt",
          "research/pilot_geometry_9train_4val/best_checkpoint.pt"}
        if rel.as_posix() in expected and file.stat().st_size<25*1024*1024:
            copy_rel(rel,"trained-engineering-checkpoint")
        else:excluded.append({"path":rel.as_posix(),"reason":"unrecognized_binary_checkpoint"})
    else:excluded.append({"path":rel.as_posix(),"reason":"third_party_or_nonessential_research_asset"})

# Copy needed derived data and deterministic engineering stages, not the
# massive pretrained protein or raw BindingDB archives.
for folder in ("data/engineering_pilot","data/staging","data/references",
               "data/reference","data/structure_fixtures","data/bindingdb_kd15_preliminary"):
    d=SRC/folder
    if not d.is_dir():continue
    for file in sorted(d.rglob("*")):
        if not file.is_file() or file.is_symlink():continue
        rel=file.relative_to(SRC)
        if file.suffix.lower() not in (".json",".jsonl",".csv",".npz",".pt",".pdb",".fasta",".fa",".tsv",".txt",".md",".npy"):
            excluded.append({"path":rel.as_posix(),"reason":"unlisted_engineering_data_type"})
            continue
        if file.stat().st_size>=49*1024*1024:
            excluded.append({"path":rel.as_posix(),"reason":"engineer_asset_exceeds_49MiB"})
            continue
        copy_rel(rel,"engineering-rehydration-fixture")
audit=SRC/"data/audit"
for file in sorted(audit.iterdir()):
    if not file.is_file() or file.is_symlink():continue
    rel=file.relative_to(SRC)
    size=file.stat().st_size
    if file.name in {"bindingdb_full_kd_source_provenance.jsonl",
                     "bindingdb_full_kd_candidates.jsonl",
                     "bindingdb_full_kd_mmseqs_hits.tsv"}:
        excluded.append({"path":rel.as_posix(),"reason":"large_reproducible_source_intermediate"})
        continue
    if file.suffix.lower() not in (".json",".jsonl",".csv",".fasta",".md"):
        excluded.append({"path":rel.as_posix(),"reason":"unnecessary_audit_binary"})
        continue
    if size>=49*1024*1024:
        excluded.append({"path":rel.as_posix(),"reason":"audit_asset_exceeds_49MiB"})
        continue
    copy_rel(rel,"research-audit-and-episode-candidate")
for file in sorted((SRC/"data/bindingdb_202611").glob("*.json")):
    copy_rel(file.relative_to(SRC),"bindingdb-archive-hash-manifest")
for file in sorted((SRC/"data/esm2_t33_650M_layer33_w1000_o128_fp16").glob("*.json")):
    copy_rel(file.relative_to(SRC),"esm2-cache-settings-and-progress")
for huge in (SRC/"data/bindingdb_202611",SRC/"data/esm2_t33_650M_layer33_w1000_o128_fp16"):
    if huge.is_dir():
        for file in huge.iterdir():
            if file.is_file() and file.suffix.lower() not in (".json",):
                excluded.append({"path":file.relative_to(SRC).as_posix(),
                                 "size":file.stat().st_size,
                                 "reason":"raw_archive_or_heavy_embedding_cache_not_for_git"})
# Never include hidden secrets, personal configuration or local git state.
ignore=StringRaw=r"""# Python and research local state
.git/
.venv/
venv/
__pycache__/
*.pyc
*.pyo
*.log.tmp
.gpu.lock
*.partial
*.tmp
.env
.env.*
*.secret
*credentials*
*.pem
*.key
# Upstream heavyweight datasets and ESM-2 caches (regenerate from sources)
data/bindingdb_202611/*.zip
data/bindingdb_202611/*.tmp
data/esm2_t33_650M_layer33_w1000_o128_fp16/*.pt
data/audit/bindingdb_full_kd_source_provenance.jsonl
data/audit/bindingdb_full_kd_candidates.jsonl
data/audit/bindingdb_full_kd_mmseqs_hits.tsv
# Local run output
__editable__*
.coverage
.pytest_cache/
.vscode/
.ipynb_checkpoints/
"""
(DEST/".gitignore").write_text(ignore,encoding="utf-8")
(DEST/".gitattributes").write_text("*.pt binary\n*.npz binary\n*.npy binary\n*.pdb text\n",encoding="utf-8")
manifest={
 "snapshot_date":"2026-11-05",
 "source":"original isolated DynaCPI v4 workspace on built-diff12",
 "destination":str(DEST),
 "notes":[
     "Snapshot deliberately excludes Git metadata, credentials, venv, raw archive and 848MiB cached ESM protein tensors.",
     "Includes three small training pilot checkpoints, source code, prepared small DAVIS engineering stages, candidate pre-split BindingDB task episodes and research reports.",
     "Pre-split BindingDB candidate JSONL files are not scientifically source-approved and must NEVER be treated as a final benchmark.",
     "Raw source intermediate files are excluded; corresponding scripts and integrity manifests permit regeneration from original datasets.",
     "No DynaCPI improvement or publishable generalization established."],
 "included_count":len(included),
 "included_total_bytes":sum(x["size"] for x in included),
 "files":included,
 "excluded":excluded}
d=DEST/"provenance"
d.mkdir(exist_ok=True)
(d/"snapshot_file_manifest.json").write_text(json.dumps(manifest,indent=2,ensure_ascii=False),encoding="utf-8")
print(json.dumps({"included_files":len(included),
                  "included_total_MiB":round(sum(x["size"] for x in included)/2**20,2),
                  "excluded_files":len(excluded),
                  "destination":str(DEST),
                  "largest_included":sorted([{"path":x["path"],"MiB":round(x["size"]/2**20,2)} for x in included],
                                            key=lambda a:-a["MiB"])[:12]},indent=2))
