"""DynaCPI v4: frozen-prediction inference and predeclared factorial contrasts.

The input PredictionEpisode exposes only assay-compatible support labels, no
query labels. All neural predictors are assumed frozen/eval and their cached
sequence/geometry means supplied independently. This module never trains.

A: m0, B: m0+b, C: m0+delta, D: m0+b+delta
Kernel: stereo-aware RDKit Morgan radius=2, 2048-bit, Tanimoto^4.
Exact absent-geometry fallback: g==s for all query and support records.
"""
from dataclasses import dataclass
from typing import Dict,Mapping,Tuple
import hashlib
import math
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from dynacpi.adaptation import adapt
from dynacpi.episode_data import PredictionEpisode,CandidateEpisodeStore

@dataclass(frozen=True)
class FrozenMeans:
    sequence_pKd: float
    geometry_pKd: float
    geometry_available: bool

@dataclass(frozen=True)
class AdapterSettings:
    w0: float=0.5
    lambda_b: float=5.0
    lambda_theta: float=0.1
    offset_cap: float=2.0
    differential_cap: float=1.0
    kernel_exponent: int=4

@dataclass(frozen=True)
class AdaptedQuery:
    compound_id: str
    smiles: str
    sequence_pKd: float
    geometry_pKd: float
    geometry_available: bool
    A: float
    B: float
    C: float
    D: float
    offset: float
    theta: float
    differential: float
    W: float
    V: float
    support_used: int

def _fp(smi,generator):
    mol=Chem.MolFromSmiles(smi)
    if mol is None:raise ValueError("Invalid query/support chemical structure")
    canonical=Chem.MolToSmiles(mol,canonical=True,isomericSmiles=True)
    fp=generator.GetFingerprint(mol)
    if fp.GetNumOnBits()==0:raise ValueError("Empty Morgan fingerprint")
    return canonical,fp

def adapt_episode(episode:PredictionEpisode,means:Mapping[str,FrozenMeans],
                  settings:AdapterSettings=AdapterSettings())->Tuple[AdaptedQuery,...]:
    CandidateEpisodeStore.validate_prediction_input(episode)
    if settings.kernel_exponent!=4:
        raise ValueError("v4 protocol fixes Tanimoto exponent at 4")
    if not 0<=settings.w0<=1 or min(settings.lambda_b,settings.lambda_theta,
                                      settings.offset_cap,settings.differential_cap)<=0:
        raise ValueError("Invalid adapter settings")
    fullset={q.compound_id for q in episode.queries}|{s.compound_id for s in episode.support}
    if set(means)!=fullset:
        raise ValueError("Frozen prediction cache must match exact episode query/support IDs")
    all_means=[]
    geometry_modes=set()
    for cid in fullset:
        m=means[cid]
        ss=float(m.sequence_pKd);gg=float(m.geometry_pKd)
        if not math.isfinite(ss) or not math.isfinite(gg):
            raise ValueError("Nonfinite frozen predictions")
        if not m.geometry_available and gg!=ss:
            raise ValueError("Missing geometry must yield exact sequence-only prediction")
        geometry_modes.add(bool(m.geometry_available))
        all_means.append(m)
    if len(geometry_modes)!=1:
        raise ValueError("One task/construct must have a consistent geometry availability state")

    gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True)
    fs={}
    for obj in (*episode.support,*episode.queries):
        canonical,fp=_fp(obj.smiles,gen)
        if hashlib.sha256(canonical.encode()).hexdigest()!=obj.compound_id:
            raise ValueError("Compound identity mismatch: canonical stereo SMILES hash")
        fs[obj.compound_id]=fp

    support=list(episode.support)
    sid=[s.compound_id for s in support]
    supp_s=np.array([means[cid].sequence_pKd for cid in sid],dtype=np.float64)
    supp_g=np.array([means[cid].geometry_pKd for cid in sid],dtype=np.float64)
    support_y=np.array([s.pKd for s in support],dtype=np.float64)
    if not np.isfinite(support_y).all():
        raise ValueError("Nonfinite support affinity")
    results=[]
    for query in episode.queries:
        mid=query.compound_id
        m=means[mid]
        k=np.asarray([float(DataStructs.TanimotoSimilarity(fs[mid],fs[cid]))**4
                      for cid in sid],dtype=np.float64)
        result=adapt(m.sequence_pKd,m.geometry_pKd,supp_s,supp_g,support_y,k,
                     w0=settings.w0,lb=settings.lambda_b,lt=settings.lambda_theta,
                     B=settings.offset_cap,C=settings.differential_cap)
        a=float(result["base"])
        b=float(result["offset"])
        delta=float(result["differential"])
        d=float(result["mean"])
        assert abs(a+b+delta-d)<1e-10
        results.append(AdaptedQuery(
            compound_id=mid,smiles=query.smiles,
            sequence_pKd=float(m.sequence_pKd),
            geometry_pKd=float(m.geometry_pKd),
            geometry_available=bool(m.geometry_available),
            A=a,B=a+b,C=a+delta,D=d,offset=b,theta=float(result["theta"]),
            differential=delta,W=float(result["W"]),V=float(result["V"]),
            support_used=len(support)))
    return tuple(results)
