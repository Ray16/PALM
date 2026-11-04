"""Independent exact packed-bit Tanimoto; pair/query sampling is explicit."""
from __future__ import annotations
import ctypes as ct
from pathlib import Path
import subprocess

import numpy as np

HERE=Path(__file__).resolve().parent


def load():
    library=HERE/'cache/packed_metrics.so'
    library.parent.mkdir(exist_ok=True)
    source=HERE/'packed_metrics.cpp'
    if not library.exists():
        subprocess.run(['g++','-O3','-march=native','-fopenmp','-shared','-fPIC',str(source),'-o',str(library)],check=True)
    module=ct.CDLL(str(library))
    u64=np.ctypeslib.ndpointer(dtype=np.uint64,flags='C_CONTIGUOUS')
    u16=np.ctypeslib.ndpointer(dtype=np.uint16,flags='C_CONTIGUOUS')
    u8=np.ctypeslib.ndpointer(dtype=np.uint8,flags='C_CONTIGUOUS')
    i64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    f64=np.ctypeslib.ndpointer(dtype=np.float64,flags='C_CONTIGUOUS')
    module.bitcounts.argtypes=[u64,ct.c_int64,ct.c_int,u16,ct.c_int]
    module.pair_tanimoto.argtypes=[u64,u16,ct.c_int,i64,i64,ct.c_int64,f64,ct.c_int]
    module.closest_training.argtypes=[u64,u16,u8,ct.c_int64,ct.c_int,i64,ct.c_int64,f64,ct.c_int]
    for name in ['bitcounts','pair_tanimoto','closest_training']:getattr(module,name).restype=None
    return module


class PackedMetrics:
    def __init__(self,packed,threads=16):
        self.bytes=np.asarray(packed,dtype=np.uint8,order='C')
        if self.bytes.ndim!=2 or self.bytes.shape[1]%8:raise ValueError('packed rows must have a multiple of eight bytes')
        self.x=self.bytes.view(np.uint64);self.n,self.words=self.x.shape
        self.threads=threads;self.library=load();self.counts=np.zeros(self.n,dtype=np.uint16)
        self.library.bitcounts(self.x,self.n,self.words,self.counts,threads)

    def pairs(self,a,b):
        a=np.ascontiguousarray(a,dtype=np.int64);b=np.ascontiguousarray(b,dtype=np.int64)
        if a.shape!=b.shape or a.ndim!=1 or min(a.min(),b.min())<0 or max(a.max(),b.max())>=self.n:
            raise ValueError('invalid pair indices')
        out=np.empty(len(a),dtype=np.float64)
        self.library.pair_tanimoto(self.x,self.counts,self.words,a,b,len(a),out,self.threads)
        return out

    def nearest(self,labels,queries):
        labels=np.ascontiguousarray(labels,dtype=np.uint8);queries=np.ascontiguousarray(queries,dtype=np.int64)
        if labels.shape!=(self.n,) or not (labels==0).any() or (queries<0).any() or (queries>=self.n).any():
            raise ValueError('invalid nearest-training inputs')
        out=np.empty(len(queries),dtype=np.float64)
        self.library.closest_training(self.x,self.counts,labels,self.n,self.words,queries,len(queries),out,self.threads)
        return out


def pair_sample(n,count=262144,seed=81273):
    rng=np.random.default_rng(seed);a=rng.integers(0,n,count)
    b=rng.integers(0,n-1,count);b+=b>=a
    return a,b


def global_score(sim,a,b,labels,bootstrap_seed=91):
    labels=np.asarray(labels);counts=np.bincount(labels)
    expectation=(len(labels)**2-float(counts@counts))/(len(labels)*(len(labels)-1))
    crossing=sim*(labels[a]!=labels[b]);mass=float(sim.mean())
    if mass<=0 or expectation<=0:
        return dict(global_relative=None,global_ci95=None,pair_mean_similarity=mass,
                    crossing_pair_mean=float(crossing.mean()),pair_count=len(sim),
                    global_status='undefined_nonpositive_mass_or_no_crossing_pairs')
    ratio=float(crossing.mean()/mass/expectation)
    blocks=64;usable=len(sim)//blocks*blocks
    numerator=crossing[:usable].reshape(blocks,-1).mean(1)
    denominator=sim[:usable].reshape(blocks,-1).mean(1)
    sample=np.random.default_rng(bootstrap_seed).integers(0,blocks,(1000,blocks))
    boot=numerator[sample].mean(1)/denominator[sample].mean(1)/expectation
    return dict(global_relative=ratio,global_ci95=np.quantile(boot,[.025,.975]).tolist(),
                pair_mean_similarity=mass,crossing_pair_mean=float(crossing.mean()),
                pair_count=len(sim),bootstrap_blocks=blocks,
                interval_scope='Approximate 95% Monte Carlo interval, bootstrap of independent uniform-pair block means; not dataset or seed generalization uncertainty')


def score(engine,labels,fold_names,a,b,sim,seed=0,query_count=128):
    result=global_score(sim,a,b,labels);result['counts']=np.bincount(labels,minlength=len(fold_names)).tolist()
    rows=[];artifacts={}
    for c,name in enumerate(fold_names[1:],1):
        eligible=np.flatnonzero(labels==c)
        queries=np.random.default_rng(200003+100*c+seed).choice(eligible,min(query_count,len(eligible)),replace=False)
        nearest=engine.nearest(labels,queries)
        boot=np.random.default_rng(300007+c+seed).choice(nearest,(1000,len(nearest)),replace=True).mean(1)
        rows.append(dict(fold=name,queries=len(queries),training_entities=int((labels==0).sum()),
                         mean_nearest=float(nearest.mean()),nearest_ci95=np.quantile(boot,[.025,.975]).tolist(),
                         fraction_gt06=float((nearest>.6).mean()),fraction_eq1=float((nearest==1).mean()),
                         interval_scope='Approximate 95% query-sampling bootstrap interval; each query is searched against every training entity'))
        artifacts[name]=dict(queries=queries,nearest=nearest)
    result['nearest_by_fold']=rows
    result['mean_heldout_nearest']=float(np.average([r['mean_nearest'] for r in rows],weights=result['counts'][1:]))
    return result,artifacts
