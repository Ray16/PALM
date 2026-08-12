"""De-risk the low-rank model BEFORE building the optimizer.
(1) Does Nystrom BB^T recover Tanimoto?  (2) Does low-rank cross-leakage
p0.p1 track the real ECFP scaled_lpi across random splits?"""
import sys, numpy as np, torch
sys.path.insert(0,"/nfs/lambda_stor_01/homes/rzhu")
import logging; logging.disable(logging.CRITICAL)
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from PALM.benchmark.benchmark_moleculenet1d import load_smiles
from PALM.benchmark.leakage import scaled_lpi

dev="cuda" if torch.cuda.is_available() else "cpu"

def ecfp(sm):
    X=np.zeros((len(sm),1024),np.float32)
    for i,s in enumerate(sm):
        m=Chem.MolFromSmiles(s)
        if m is not None: DataStructs.ConvertToNumpyArray(AllChem.GetMorganFingerprintAsBitVect(m,2,nBits=1024),X[i])
    return X

def tani(A,B):  # (a,d),(b,d)->(a,b)
    inter=A@B.T; ca=A.sum(1); cb=B.sum(1)
    union=ca[:,None]+cb[None,:]-inter
    return torch.where(union>0, inter/union, torch.zeros_like(inter))

def nystrom(X, r, seed=0):
    n=X.shape[0]; r=min(r,n)
    g=torch.Generator(device=dev).manual_seed(seed)
    idx=torch.randperm(n, generator=g, device=dev)[:r]
    Xt=torch.as_tensor(X,dtype=torch.float32,device=dev)
    L=Xt[idx]
    C=tani(Xt,L)               # (n,r)
    W=tani(L,L)                # (r,r)
    ev,evec=torch.linalg.eigh(W)
    ev=torch.clamp(ev,min=1e-6)
    Wih=evec @ torch.diag(1.0/torch.sqrt(ev)) @ evec.T
    return C @ Wih             # (n,r)

for ds in ["esol","bace","lipophilicity"]:
    sm=load_smiles(ds); n=len(sm); X=ecfp(sm)
    Xt=torch.as_tensor(X,dtype=torch.float32,device=dev)
    for r in [128,256,512]:
        B=nystrom(X,r,seed=0)
        # (1) recovery on 4000 random pairs
        g=torch.Generator(device=dev).manual_seed(1)
        ii=torch.randint(0,n,(4000,),generator=g,device=dev)
        jj=torch.randint(0,n,(4000,),generator=g,device=dev)
        mask=ii!=jj; ii,jj=ii[mask],jj[mask]
        true=(tani(Xt[ii],Xt[jj]).diag() if False else (Xt[ii]*Xt[jj]).sum(1))  # placeholder
        # exact tanimoto per pair
        inter=(Xt[ii]*Xt[jj]).sum(1); ca=Xt[ii].sum(1); cb=Xt[jj].sum(1)
        true=inter/(ca+cb-inter+1e-9)
        approx=(B[ii]*B[jj]).sum(1)
        rmse=float(torch.sqrt(((true-approx)**2).mean()))
        corr=float(torch.corrcoef(torch.stack([true,approx]))[0,1])
        # (2) objective tracking across 25 random splits
        lr_vals,real_vals=[],[]
        for s in range(25):
            gg=np.random.default_rng(s); lab=np.array([0]*int(0.8*n)+[1]*(n-int(0.8*n))); gg.shuffle(lab)
            labt=torch.as_tensor(lab,device=dev)
            p0=B[labt==0].sum(0); p1=B[labt==1].sum(0)
            lr_vals.append(float(p0@p1))
            real_vals.append(scaled_lpi(sm,{sm[i]:("train" if lab[i]==0 else "test") for i in range(n)})[0])
        oc=float(np.corrcoef(lr_vals,real_vals)[0,1])
        print(f"{ds:<14} r={r:>4}  pair: RMSE={rmse:.3f} corr={corr:.3f}   split-obj corr(lowrank,real)={oc:.3f}")
    print()
