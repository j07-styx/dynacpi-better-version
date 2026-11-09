"""Equal-support few-shot PerceiverCPI comparator; frozen predictions only.

A meaningful DynaCPI adaptation gain must beat a PerceiverCPI predictor given
THE SAME labeled support compounds and chemical similarity features.
This module consumes an in-memory PredictionEpisode; it never loads assays,
query labels or source roles, runs an optimizer, or grants cohort approval.

Controls, all frozen PerceiverCPI + identical K support observations:
 P0: original model (K effectively zero)
 G: global shrinkage offset
 L: query-local Tanimoto^4 weighted shrinkage offset
 KR: query-local kernel ridge residual adaptation

No tuning of regularization or caps on reserved held-out outcomes.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping, Tuple
import hashlib
import math

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from dynacpi.episode_data import PredictionEpisode, CandidateEpisodeStore


@dataclass(frozen=True)
class PerceiverSupportSettings:
    lambda_global: float = 5.0
    lambda_local: float = 5.0
    lambda_kernel: float = 1.0
    residual_cap: float = 1.0
    tanimoto_power: int = 4


@dataclass(frozen=True)
class PerceiverSupportPrediction:
    compound_id: str
    p0: float
    global_offset: float
    local_offset: float
    kernel_ridge: float
    support_used: int


def _fingerprint(smiles, generator):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid query/support SMILES")
    canonical=Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    fp=generator.GetFingerprint(mol)
    if fp.GetNumOnBits()==0:
        raise ValueError("Empty Morgan fingerprint")
    return canonical, fp


def perceiver_equal_support_controls(
    episode: PredictionEpisode,
    frozen_perceiver_pKd: Mapping[str, float],
    settings: PerceiverSupportSettings = PerceiverSupportSettings(),
) -> Tuple[PerceiverSupportPrediction, ...]:
    CandidateEpisodeStore.validate_prediction_input(episode)
    if settings.tanimoto_power != 4:
        raise ValueError("Predeclared comparison uses stereochemical Morgan Tanimoto^4")
    if not all(math.isfinite(z) and z>0 for z in (
        settings.lambda_global,settings.lambda_local,
        settings.lambda_kernel,settings.residual_cap,
    )):
        raise ValueError("Positive finite regularization/cap required")
    support=list(episode.support)
    queries=list(episode.queries)
    all_ids=[p.compound_id for p in (*support,*queries)]
    if set(frozen_perceiver_pKd)!=set(all_ids):
        raise ValueError("Frozen Perceiver cache must cover exactly support and query compounds")
    pred={}
    for cid in all_ids:
        value=float(frozen_perceiver_pKd[cid])
        if not math.isfinite(value):
            raise ValueError("Nonfinite Perceiver prediction")
        pred[cid]=value
    for obs in support:
        if not math.isfinite(float(obs.pKd)):
            raise ValueError("Nonfinite support measurement")
    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    fps={}
    for obj in (*support,*queries):
        canonical,fp=_fingerprint(obj.smiles,gen)
        if hashlib.sha256(canonical.encode("utf-8")).hexdigest()!=obj.compound_id:
            raise ValueError("Exact stereo-aware compound identity mismatch")
        fps[obj.compound_id]=fp
    if not queries:
        raise ValueError("Episode has no queries")
    if not support:
        return tuple(PerceiverSupportPrediction(
            q.compound_id,pred[q.compound_id],pred[q.compound_id],
            pred[q.compound_id],pred[q.compound_id],0) for q in queries)
    support_ids=[p.compound_id for p in support]
    residual=np.asarray([float(p.pKd)-pred[p.compound_id] for p in support])
    n=len(support)
    global_offset=float(np.clip(residual.sum()/(n+settings.lambda_global),
                                -settings.residual_cap,settings.residual_cap))
    K=np.empty((n,n),dtype=np.float64)
    for i in range(n):
        for j in range(i,n):
            sim=float(DataStructs.TanimotoSimilarity(fps[support_ids[i]],fps[support_ids[j]]))
            K[i,j]=K[j,i]=sim**4
    alpha=np.linalg.solve(K+settings.lambda_kernel*np.eye(n),residual)
    out=[]
    for query in queries:
        qid=query.compound_id
        weights=np.asarray([
            float(DataStructs.TanimotoSimilarity(fps[qid],fps[sid]))**4
            for sid in support_ids],dtype=np.float64)
        local_bias=float(np.clip(np.dot(weights,residual)/
                                  (weights.sum()+settings.lambda_local),
                                  -settings.residual_cap,settings.residual_cap))
        kr_bias=float(np.clip(np.dot(weights,alpha),
                              -settings.residual_cap,settings.residual_cap))
        base=pred[qid]
        out.append(PerceiverSupportPrediction(
            qid, base,base+global_offset,base+local_bias,base+kr_bias,n))
    return tuple(out)
