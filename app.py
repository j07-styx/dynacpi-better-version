"""DynaCPI v4 backend-only REST API for separate frontends.

Engineering scope: four cached DAVIS pilot targets; source-family-independent
BindingDB scientific cohort remains blocked. Training is CLI-only, not HTTP.
The baseline is the conservative default. Experimental geometry checkpoints
must be requested explicitly.
"""
import hashlib
import json
import math
import msvcrt
import os
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from flask import Flask, jsonify, request
from rdkit import Chem

ROOT = Path(r"C:\Users\J0124\dynacpi_v4")
WORKSPACE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(WORKSPACE))
from dynacpi.frozen_adapter import FrozenMeans, adapt_episode
from dynacpi.episode_data import PredictionEpisode, QueryCompound, SupportObservation

APP = Flask(__name__)
APP.config["MAX_CONTENT_LENGTH"] = 8192
BASE_CKPT = ROOT / "research/pilot_sequence/checkpoint.pt"
GRAPH_MANIFEST = ROOT / "data/engineering_pilot/limited_alpha_3d_graph_manifest.json"
DAVIS_ROWS = ROOT / "data/engineering_pilot/rows.json"
TRAIN_DIR = ROOT / "research/dynacpi_v4_trainonly_geometry_20261109"
REPAIRED_DIR = WORKSPACE / "checkpoints"
ALLOWED_K = {0, 1, 2, 5, 10, 20}
LOCK = threading.RLock()
CACHE = {"active_model": None, "predictor": None, "targets": None, "torch": None}
DEFAULT_ORIGINS = {
    "http://localhost:3000", "http://127.0.0.1:3000",
    "http://localhost:5173", "http://127.0.0.1:5173",
    "http://localhost:4173", "http://127.0.0.1:4173",
    "http://localhost:8080", "http://127.0.0.1:8080",
}
ORIGINS = set(os.environ.get("DYNACPI_ALLOWED_ORIGINS", "").split(",")) - {""}
ORIGINS = ORIGINS or DEFAULT_ORIGINS


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


@contextmanager
def gpu_lease_timeout(timeout=15):
    path = ROOT / ".gpu.lock"
    with path.open("a+b") as f:
        if not path.stat().st_size:
            f.write(b"1")
            f.flush()
        start = time.monotonic()
        while True:
            f.seek(0)
            try:
                msvcrt.locking(f.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if time.monotonic() - start > timeout:
                    raise RuntimeError("GPU busy: another research job holds the shared lease.")
                time.sleep(0.2)
        try:
            yield
        finally:
            f.seek(0)
            msvcrt.locking(f.fileno(), msvcrt.LK_UNLCK, 1)


@APP.after_request
def cors(response):
    origin = request.headers.get("Origin", "")
    if origin in ORIGINS:
        response.headers["Access-Control-Allow-Origin"] = origin
        response.headers["Vary"] = "Origin"
        response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
        response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


def available_models():
    models = [
        {"id": "sequence_baseline", "kind": "sequence-only frozen foundation",
         "source": "research/pilot_sequence/checkpoint.pt",
         "mode": "sequence", "is_default": True, "status": "available"}
    ]
    archived = ROOT / "research/pilot_geometry_seed_sweep_dev/seed_1107/best_checkpoint.pt"
    if archived.is_file():
        models.append({"id": "legacy_geometry_1107", "kind": "frozen sequence plus previously dev-selected GVP",
                       "source": "research/pilot_geometry_seed_sweep_dev/seed_1107/best_checkpoint.pt",
                       "mode": "geometry", "is_default": False,
                       "status": "development-selected; no independent validation"})
    for seed in (1901, 1902, 1903):
        p = TRAIN_DIR / ("real_seed" + str(seed))
        manifest, weights = p / "COMPLETE.json", p / "best_checkpoint.pt"
        if manifest.is_file() and weights.is_file():
            info = json.loads(manifest.read_text(encoding="utf-8"))
            if info.get("status") == "COMPLETE_DEVELOPMENT_ONLY_NOT_SCIENTIFIC_VALIDATION":
                models.append({
                    "id": "trainonly_geometry_" + str(seed),
                    "kind": "GVP geometry trained with train-only checkpoint selection",
                    "source": str(weights.relative_to(ROOT)).replace("\\", "/"),
                    "mode": "geometry", "is_default": False,
                    "status": "development-only; negative historical validation finding",
                    "selected_epoch": info["selected_epoch"],
                    "historical_dev_macro_rmse": info["historically_inspected_dev_diagnostic"]["macro_geometry_rmse"]
                })
    for seed in (2831, 2832, 2833):
        path = REPAIRED_DIR / ("seed_" + str(seed))
        manifest, weights = path / "COMPLETE.json", path / "checkpoint.pt"
        if manifest.is_file() and weights.is_file():
            m = json.loads(manifest.read_text(encoding="utf-8"))
            if m.get("status") == "ENGINEERING_TRAIN_ONLY_CANDIDATE_NOT_INDEPENDENTLY_VALIDATED":
                models.append({
                    "id": "repaired_sequence_" + str(seed),
                    "kind": "two-bypass repaired compound and compound-protein head",
                    "source": str(weights), "mode": "sequence", "is_default": False,
                    "status": "24-protein train-only engineering checkpoint; DEV previously inspected",
                    "epoch": m["epoch"],
                    "historical_dev_macro_rmse": m["historical_dev_diagnostic"]["target_macro_rmse"],
                })
    from profile_serving import candidates
    models.extend(candidates())
    return models


def canonical_smiles(raw):
    if not isinstance(raw, str) or not raw.strip() or len(raw) > 450:
        raise ValueError("smiles must be a nonempty string up to 450 characters")
    mol = Chem.MolFromSmiles(raw.strip())
    if mol is None or mol.GetNumAtoms() > 250:
        raise ValueError("invalid SMILES or >250-atom demo limit")
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def targets():
    if CACHE["targets"] is not None:
        return CACHE["targets"]
    records = json.loads(GRAPH_MANIFEST.read_text(encoding="utf-8"))["records"]
    selected = {r["source_name"]: r for r in records if
                r.get("status") == "GRAPH_MAPPED_ENGINEERING_ONLY" and r.get("pilot_split") == "val"}
    if set(selected) != {"ADCK4", "FRK", "MEK5", "OSR1"}:
        raise RuntimeError("Unexpected DAVIS development target manifest; refusing to serve.")
    rows = json.loads(DAVIS_ROWS.read_text(encoding="utf-8"))
    smiles = json.loads((ROOT / "data/staging/compounds.json").read_text(encoding="utf-8"))
    prepared = {}
    for name, item in selected.items():
        pid = item["sequence_sha256"]
        pool = []
        for row in rows:
            if row["sequence"] == pid and row["split"] == "val":
                s = canonical_smiles(smiles[row["compound"]])
                cid = hashlib.sha256(s.encode()).hexdigest()
                pool.append({"compound_id": cid, "smiles": s, "y": float(row["y"])})
        pool.sort(key=lambda r: hashlib.sha256(("fixed_backend_support_v1|"+pid+"|"+r["compound_id"]).encode()).hexdigest())
        if len(pool) != 54 or len({r["compound_id"] for r in pool}) != 54:
            raise RuntimeError("DAVIS development input identity mismatch.")
        prepared[name] = {"id": pid, "record": item, "pool": pool, "esm": None, "graph": None}
    CACHE["targets"] = prepared
    return prepared


def load_predictor(model_id):
    if CACHE["active_model"] == model_id and CACHE["predictor"] is not None:
        return CACHE["predictor"], CACHE["torch"]
    if model_id.startswith(("profile_", "anchored_sequence_")):
        from profile_serving import load_candidate
        predictor, torch = load_candidate(model_id)
        CACHE["active_model"] = model_id
        CACHE["predictor"] = predictor
        CACHE["torch"] = torch
        return predictor, torch
    import torch
    from dynacpi.sequence_model import SequenceFoundation
    from dynacpi.geometry_branch import GeometryBranch
    from dynacpi.frozen_pair_predictor import FrozenPairPredictor
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the existing MPN molecular graph implementation")
    if model_id.startswith("repaired_sequence_"):
        from repaired_model import RepairedSequenceFoundation
        options = {m["id"]: m for m in available_models()}
        if model_id not in options:
            raise ValueError("Unknown repaired model ID")
        weight_path = Path(options[model_id]["source"])
        report = json.loads((weight_path.parent / "COMPLETE.json").read_text(encoding="utf-8"))
        if sha(weight_path) != report["checkpoint_sha256"]:
            raise RuntimeError("Repaired checkpoint sha256 mismatch.")
        sc = torch.load(weight_path, map_location="cpu", weights_only=False)
        if sc.get("kind") != "repaired_sequence_v4_2" or sc.get("architecture") != "two_path_v1":
            raise RuntimeError("Repaired model architecture mismatch.")
        if sc.get("row_sha256") != sha(DAVIS_ROWS):
            raise RuntimeError("Training dataset digest mismatch.")
        if sc.get("repair_source_sha256") != sha(WORKSPACE / "repaired_model.py"):
            raise RuntimeError("Repaired model source digest mismatch.")
        seq = RepairedSequenceFoundation().cuda().eval()
        seq.load_state_dict(sc["model_state"], strict=True)
    else:
        sc = torch.load(BASE_CKPT, map_location="cpu", weights_only=False)
        if sc.get("kind") != "sequence" or sc.get("row_sha256") != sha(DAVIS_ROWS):
            raise RuntimeError("Frozen sequence checkpoint and dataset hashes do not match.")
        seq = SequenceFoundation().cuda().eval()
        seq.load_state_dict(sc["state"], strict=True)
    geo = GeometryBranch().cuda().eval()
    if model_id != "sequence_baseline" and not model_id.startswith("repaired_sequence_"):
        models = {m["id"]: m for m in available_models()}
        if model_id not in models or models[model_id]["mode"] != "geometry":
            raise ValueError("Model unavailable.")
        weight_path = ROOT / models[model_id]["source"]
        if model_id.startswith("trainonly_geometry_"):
            report = json.loads((weight_path.parent / "COMPLETE.json").read_text(encoding="utf-8"))
            if sha(weight_path) != report["checkpoint_sha256"]:
                raise RuntimeError("Trained GVP checkpoint digest mismatch.")
        checkpoint = torch.load(weight_path, map_location="cpu", weights_only=False)
        if checkpoint["sequence_checkpoint_sha256"] != sha(BASE_CKPT):
            raise RuntimeError("Frozen foundation/geometry checkpoints have different parent hashes.")
        if checkpoint.get("graph_manifest_sha256", checkpoint.get("manifest_sha256")) != sha(GRAPH_MANIFEST):
            raise RuntimeError("3D graph manifest digest changed.")
        if checkpoint.get("davis_rows_sha256", checkpoint.get("row_sha256")) != sha(DAVIS_ROWS):
            raise RuntimeError("3D model dataset digest changed.")
        geo.load_state_dict(checkpoint["state"], strict=True)
    predictor = FrozenPairPredictor(seq, geo, device="cuda",
                                    label_mean=float(sc["label_mean"]), label_std=float(sc["label_std"]))
    CACHE["active_model"] = model_id
    CACHE["predictor"] = predictor
    CACHE["torch"] = torch
    return predictor, torch


def infer(payload):
    model_id = payload.get("model_id", "sequence_baseline")
    models = {m["id"]: m for m in available_models()}
    if model_id not in models:
        raise ValueError("Invalid model_id: see GET /api/v1/models")
    target_name = payload.get("target")
    if target_name not in targets():
        raise ValueError("Unsupported protein: see GET /api/v1/targets")
    support_k = payload.get("support_k", 0)
    if type(support_k) is not int or support_k not in ALLOWED_K:
        raise ValueError("support_k must be one of 0, 1, 2, 5, 10, 20")
    smi = canonical_smiles(payload.get("smiles"))
    qid = hashlib.sha256(smi.encode()).hexdigest()
    item = targets()[target_name]
    supports = [r for r in item["pool"] if r["compound_id"] != qid][:support_k]
    if len(supports) != support_k:
        raise RuntimeError("Insufficient nonoverlapping support items.")
    predictor, torch = load_predictor(model_id)
    if item["esm"] is None:
        pid = item["id"]
        obj = torch.load(ROOT / "data/esm2_t33_650M_layer33_w1000_o128_fp16" / (pid + ".pt"),
                         map_location="cpu", weights_only=True)
        if obj["sequence_hash"] != pid:
            raise RuntimeError("ESM input sequence hash mismatch.")
        gp = ROOT / "data/engineering_pilot/structure_graphs" / pid / "graph.pt"
        if sha(gp) != item["record"]["graph_sha256"]:
            raise RuntimeError("3D protein graph digest mismatch.")
        item["esm"] = obj["embeddings"]
        item["graph"] = torch.load(gp, map_location="cpu", weights_only=True)
    all_smiles = [smi] + [r["smiles"] for r in supports]
    has_geometry = models[model_id]["mode"] == "geometry"
    values = predictor.predict(all_smiles, item["esm"], item["graph"] if has_geometry else None)
    episode = PredictionEpisode(
        task_id="ENGINEERING_DAVIS_BACKEND_" + item["id"],
        sequence_sha256=item["id"], protein_component_id=item["id"], assay_key=item["id"],
        support=tuple(SupportObservation(r["compound_id"], r["smiles"], r["y"], item["id"]) for r in supports),
        queries=(QueryCompound(qid, smi),))
    output = adapt_episode(episode, values)[0]
    numbers = {
        "sequence_pkd": output.sequence_pKd,
        "geometry_pkd": output.geometry_pKd,
        "A_zero_shot_mixture_pkd": output.A,
        "B_support_calibrated_pkd": output.B,
        "C_differential_only_pkd": output.C,
        "D_full_adapted_pkd": output.D,
        "calibration_offset_pkd": output.offset,
        "differential_correction_pkd": output.differential,
        "similarity_weight_sum": output.W,
        "geometry_disagreement_variance": output.V
    }
    if not all(math.isfinite(float(x)) for x in numbers.values()):
        raise RuntimeError("Model emitted a nonfinite prediction.")
    response = {"model_id": model_id, "model_status": models[model_id]["status"],
            "protein": target_name, "canonical_smiles": smi, "support_k": support_k,
            "geometry_available": has_geometry, "predictions": numbers,
            "units": "pKd (model-estimated, not experimentally verified)",
            "scope": "legacy_davis_development_only",
            "warnings": ["Not an independent generalization test",
                         "DAVIS development support labels are used only for adaptation, not returned",
                         "Structure-to-experimental construct provenance is not independently verified"]}
    if hasattr(predictor, "last_details"):
        response["components"] = predictor.last_details[qid]
        response["target_conditioned"] = models[model_id].get("target_conditioned", True)
    return response


@APP.route("/api/v1/health", methods=["GET"])
def health():
    return jsonify({"ok": True, "project": "DynaCPI", "backend_version": "v4.3-anchored-chemical-profile-20261109",
                    "scope": "engineering-only", "current_model": CACHE["active_model"]})


@APP.route("/api/v1/models", methods=["GET"])
def models():
    return jsonify({"models": available_models(), "default_model_id": "sequence_baseline"})


@APP.route("/api/v1/targets", methods=["GET"])
def target_index():
    return jsonify({"targets": [{"id": name, "protein_hash": item["id"], "support_limit": 20}
                                for name, item in targets().items()],
                    "scope": "four DAVIS engineering validation proteins"})


@APP.route("/api/v1/training", methods=["GET"])
def training_index():
    summary = TRAIN_DIR / "SUMMARY.json"
    if not summary.is_file():
        return jsonify({"status": "no_completed_new_trainonly_run"}), 404
    report = json.loads(summary.read_text(encoding="utf-8"))
    return jsonify({"status": report["status"], "arms": report["arms"],
                    "baseline_dev_rmse": report["baseline_dev"],
                    "scientific_approval": False,
                    "note": "Four previously inspected DAVIS dev proteins, not independent test."})


@APP.route("/api/v1/training/repaired", methods=["GET"])
def repaired_training_index():
    report_path = REPAIRED_DIR / "SUMMARY.json"
    if not report_path.is_file():
        return jsonify({"status": "new_repaired_training_not_yet_complete"}), 404
    summary = json.loads(report_path.read_text(encoding="utf-8"))
    return jsonify({
        "status": summary["status"],
        "candidate_count": len(summary.get("completed", [])),
        "seeds": summary["seeds"],
        "trained_pairs": summary["training_pairs"],
        "trained_proteins": summary["train_proteins"],
        "dev_diagnostic": summary.get("completed", []),
        "independent_scientific_validation": False,
        "note": "DEV cohort has been inspected in previous experiments; not an independent test.",
    })


@APP.route("/api/v1/training/profile", methods=["GET"])
def profile_training_index():
    path = WORKSPACE / "interaction_study_20261109/study_results.json"
    if not path.is_file():
        return jsonify({"status": "profile_study_not_complete"}), 404
    report = json.loads(path.read_text(encoding="utf-8"))
    families = ["global_mean", "chemical_profile", "additive_profile", "interaction_profile",
                "repaired_neural", "anchored_neural"]
    metrics = {}
    for family in families:
        metrics[family] = {}
        for regime in ["cold_protein", "cold_compound", "double_cold"]:
            values = [v[regime]["macro_rmse"] for f in report["folds"]
                      for name, v in f["models"].items()
                      if name == family or name.startswith(family + "_seed")]
            metrics[family][regime] = sum(values) / len(values)
    return jsonify({"status": report["status"], "outer_folds": 3,
                    "neural_seeds": report["neural_seeds"], "macro_rmse": metrics,
                    "chemical_similarity_threshold": 0.5,
                    "independent_scientific_validation": False,
                    "note": "Historical training-data CV; protein identities, not families, are held out."})


@APP.route("/api/v1/predict", methods=["POST", "OPTIONS"])
def predict():
    if request.method == "OPTIONS":
        return ("", 204)
    body = request.get_json(silent=True)
    if not isinstance(body, dict):
        return jsonify({"error": "Expected JSON object"}), 400
    try:
        with LOCK:
            if str(body.get("model_id", "")).startswith("profile_"):
                return jsonify(infer(body))
            with gpu_lease_timeout():
                return jsonify(infer(body))
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        APP.logger.exception("Prediction failed")
        return jsonify({"error": str(exc)}), 503


if __name__ == "__main__":
    print("DynaCPI v4.3 backend API: http://127.0.0.1:8781/api/v1/health", flush=True)
    APP.run(host="127.0.0.1", port=int(os.environ.get("DYNACPI_BACKEND_PORT", "8781")),
            debug=False, use_reloader=False, threaded=True)
