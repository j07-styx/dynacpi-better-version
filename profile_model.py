"""Chemical prior plus a regularized protein-dependent affinity residual.

Small-data comparator, not a 3D or novel deep-learning architecture. All
centering, length scales and coefficients are fitted from training rows only.
Protein features are mean-pooled frozen ESM embeddings, L2 normalized. Compound
features are binary Morgan fingerprints. No protein IDs are trainable features.
"""
from pathlib import Path
import numpy as np


def tanimoto(x,z):
    x=np.asarray(x,dtype=np.float64);z=np.asarray(z,dtype=np.float64)
    dot=x@z.T
    den=x.sum(1)[:,None]+z.sum(1)[None,:]-dot
    return np.divide(dot,den,out=np.zeros_like(dot),where=den>0)


def normalize(x):
    x=np.asarray(x,dtype=np.float64)
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)


class AffinityProfile:
    VERSION='chemical_prior_centered_protein_residual_v1'
    def __init__(self,chemical_ridge=.1,residual_ridge=1.0,mode='interaction'):
        if mode not in ('chemical','additive','interaction'):raise ValueError('Invalid profile mode')
        if chemical_ridge<=0 or residual_ridge<=0:raise ValueError('Ridge must be positive')
        self.chemical_ridge=float(chemical_ridge);self.residual_ridge=float(residual_ridge);self.mode=mode

    def fit(self,protein_features,fingerprints,labels):
        self.proteins=normalize(protein_features)
        self.fingerprints=np.asarray(fingerprints,dtype=np.float64)
        y=np.asarray(labels,dtype=np.float64)
        if y.shape!=(len(self.proteins),len(self.fingerprints)) or not np.isfinite(y).all():
            raise ValueError('Expected finite protein-by-compound affinity matrix')
        kc=tanimoto(self.fingerprints,self.fingerprints)
        prior=y.mean(0);self.intercept=float(prior.mean())
        self.chemical_coef=np.linalg.solve(kc+self.chemical_ridge*np.eye(len(kc)),prior-self.intercept)
        dist=np.maximum(0,2-2*(self.proteins@self.proteins.T))
        values=dist[np.triu_indices(len(dist),1)]
        self.protein_scale=float(np.median(values[values>1e-12])) if np.any(values>1e-12) else 1.0
        kp=np.exp(-dist/self.protein_scale)
        self.protein_column_mean=kp.mean(0);self.protein_grand_mean=float(kp.mean())
        centered=kp-kp.mean(1,keepdims=True)-kp.mean(0,keepdims=True)+kp.mean()
        residual=y-y.mean(0,keepdims=True)
        self.protein_coef=np.linalg.solve(centered+self.residual_ridge*np.eye(len(kp)),residual.mean(1))
        dp,up=np.linalg.eigh(centered);dc,uc=np.linalg.eigh(kc)
        dp=np.maximum(dp,0);dc=np.maximum(dc,0)
        spectral=(up.T@residual@uc)/(dp[:,None]*dc[None,:]+self.residual_ridge)
        self.interaction_coef=up@spectral@uc.T
        return self

    def chemical(self,fingerprints):
        return self.intercept+tanimoto(fingerprints,self.fingerprints)@self.chemical_coef

    def protein_kernel(self,features):
        q=normalize(features)
        kp=np.exp(-np.maximum(0,2-2*(q@self.proteins.T))/self.protein_scale)
        return kp-kp.mean(1,keepdims=True)-self.protein_column_mean[None,:]+self.protein_grand_mean

    def predict(self,protein_features,fingerprints,mode=None):
        mode=mode or self.mode
        prior=self.chemical(fingerprints)
        out=np.broadcast_to(prior,(len(protein_features),len(prior))).copy()
        if mode=='chemical':return out
        kp=self.protein_kernel(protein_features)
        if mode=='additive':return out+(kp@self.protein_coef)[:,None]
        if mode=='interaction':return out+kp@self.interaction_coef@tanimoto(fingerprints,self.fingerprints).T
        raise ValueError('Invalid mode')

    def save(self,path):
        path=Path(path)
        if path.exists():raise RuntimeError('Refusing to replace an existing model')
        np.savez_compressed(path,version=self.VERSION,mode=self.mode,
            chemical_ridge=self.chemical_ridge,residual_ridge=self.residual_ridge,
            proteins=self.proteins,fingerprints=self.fingerprints,intercept=self.intercept,
            chemical_coef=self.chemical_coef,protein_scale=self.protein_scale,
            protein_column_mean=self.protein_column_mean,protein_grand_mean=self.protein_grand_mean,
            protein_coef=self.protein_coef,interaction_coef=self.interaction_coef)

    @classmethod
    def load(cls,path):
        with np.load(path,allow_pickle=False) as z:
            if str(z['version'])!=cls.VERSION:raise ValueError('Unsupported profile checkpoint')
            m=cls(float(z['chemical_ridge']),float(z['residual_ridge']),str(z['mode']))
            for name in ['proteins','fingerprints','chemical_coef','protein_column_mean','protein_coef','interaction_coef']:
                setattr(m,name,z[name].copy())
            for name in ['intercept','protein_scale','protein_grand_mean']:setattr(m,name,float(z[name]))
        return m
