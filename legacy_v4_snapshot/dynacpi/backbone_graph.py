"""Construct-aware GVP features for a single aligned protein chain."""
import numpy as np
import torch
from scipy.spatial import cKDTree

def unit(x):
    norm=np.linalg.norm(x)
    return (x/norm,True) if np.isfinite(norm) and norm>1e-8 else (np.zeros(3),False)

def torsion(a,b,c,d):
    axis,ok=unit(c-b)
    if not ok:return 0.,False
    u=a-b;u=u-u.dot(axis)*axis
    v=d-c;v=v-v.dot(axis)*axis
    u,ok1=unit(u);v,ok2=unit(v)
    return np.arctan2(np.cross(axis,u).dot(v),u.dot(v)),bool(ok1 and ok2)

def make_graph(coords,polymer_positions,chain_ids,plddt=None,pae=None):
    xyz=np.asarray(coords,dtype=np.float64)
    if xyz.ndim!=3 or xyz.shape[1:]!=(4,3) or not len(xyz):raise ValueError("Expected [L,4,3] N,CA,C,O")
    length=len(xyz)
    if len(polymer_positions)!=length or len(chain_ids)!=length:raise ValueError("Mapping length mismatch")
    valid=np.isfinite(xyz).all(axis=(1,2));indices=np.flatnonzero(valid)
    if len(indices)/length<.8 or not len(indices):return None
    if plddt is not None and len(plddt)!=length:raise ValueError("Confidence length mismatch")
    if pae is not None and np.shape(pae)!=(length,length):raise ValueError("PAE mapping mismatch")
    contiguous=np.zeros(length,dtype=bool)
    for j in range(length-1):
        if valid[j] and valid[j+1] and chain_ids[j]==chain_ids[j+1] and polymer_positions[j+1]==polymer_positions[j]+1:
            contiguous[j]=1<=np.linalg.norm(xyz[j,2]-xyz[j+1,0])<=2
    ns=np.zeros((len(indices),14));nv=np.zeros((len(indices),3,3))
    for i,j in enumerate(indices):
        n,ca,c,_=xyz[j];back=j>0 and contiguous[j-1];forward=contiguous[j]
        angles=[torsion(xyz[j-1,2],n,ca,c) if back else (0,False),
                torsion(n,ca,c,xyz[j+1,0]) if forward else (0,False),
                torsion(ca,c,xyz[j+1,0],xyz[j+1,1]) if forward else (0,False)]
        for h,(angle,ok) in enumerate(angles):
            if ok:ns[i,h]=np.cos(angle);ns[i,3+h]=np.sin(angle);ns[i,6+h]=1
        if forward:nv[i,0],ns[i,9]=unit(xyz[j+1,1]-ca)
        if back:nv[i,1],ns[i,10]=unit(xyz[j-1,1]-ca)
        cv,cok=unit(c-ca);nn,nok=unit(n-ca)
        b,bok=unit(cv+nn);p,pok=unit(np.cross(cv,nn))
        if cok and nok and bok and pok:nv[i,2]=-b*np.sqrt(1/3)-p*np.sqrt(2/3);ns[i,11]=1
        if plddt is not None and np.isfinite(plddt[j]):ns[i,12]=np.clip(plddt[j]/100,0,1);ns[i,13]=1
    compact={j:i for i,j in enumerate(indices)}
    ca=xyz[indices,1];tree=cKDTree(ca);edges=[]
    for dest,j in enumerate(indices):
        cand=[h for h in tree.query_ball_point(ca[dest],20) if h!=dest]
        cand.sort(key=lambda h:(np.linalg.norm(ca[h]-ca[dest]),indices[h]))
        sources=set(cand[:30])
        if j>0 and contiguous[j-1]:sources.add(compact[j-1])
        if contiguous[j]:sources.add(compact[j+1])
        edges.extend((h,dest) for h in sorted(sources))
    es=np.zeros((len(edges),37));ev=np.zeros((len(edges),1,3));freq=np.exp(np.arange(0,16,2)*(-np.log(10000)/16))
    for k,(a,b) in enumerate(edges):
        i,j=indices[a],indices[b];diff=ca[a]-ca[b];dist=np.linalg.norm(diff);ev[k,0]=unit(diff)[0]
        es[k,:16]=np.exp(-((dist-np.linspace(0,20,16))/(20/16))**2)
        angle=(i-j)*freq;es[k,16:32]=np.r_[np.cos(angle),np.sin(angle)]
        if pae is not None:
            for h,val in enumerate((pae[i,j],pae[j,i])):
                if np.isfinite(val):es[k,32+h]=np.clip(val/30,0,1);es[k,34+h]=1
        es[k,36]=abs(i-j)==1 and contiguous[min(i,j)]
    return {"indices":torch.tensor(indices),"node_s":torch.tensor(ns,dtype=torch.float32),"node_v":torch.tensor(nv,dtype=torch.float32),
            "edge_s":torch.tensor(es,dtype=torch.float32),"edge_v":torch.tensor(ev,dtype=torch.float32),
            "edges":torch.tensor(edges,dtype=torch.long).reshape(-1,2).T.contiguous()}
