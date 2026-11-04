import numpy as np
import pytest
from rdkit import DataStructs

from metrics import PackedMetrics,pair_sample,global_score


def test_packed_tanimoto_and_full_training_search_match_rdkit():
    rng=np.random.default_rng(9);bits=(rng.random((37,128))<.2).astype(np.uint8)
    bits[:,0]=1;bits[36]=bits[0]
    packed=np.packbits(bits,axis=1,bitorder='little');engine=PackedMetrics(packed,threads=2)
    fps=[]
    for row in bits:
        fp=DataStructs.ExplicitBitVect(128);fp.SetBitsFromList(np.flatnonzero(row).tolist());fps.append(fp)
    S=np.array([DataStructs.BulkTanimotoSimilarity(fp,fps) for fp in fps])
    a,b=pair_sample(len(bits),4096)
    assert not np.any(a==b)
    assert np.allclose(engine.pairs(a,b),S[a,b],rtol=0,atol=1e-15)
    labels=np.repeat(np.arange(3),[25,6,6]).astype(np.uint8)
    queries=np.flatnonzero(labels!=0)
    assert np.allclose(engine.nearest(labels,queries),S[queries][:,labels==0].max(1),rtol=0,atol=1e-15)
    assert engine.nearest(labels,np.array([36]))[0]==1


def test_global_pair_normalization_matches_direct_full_kernel():
    bits=np.zeros((5,128),dtype=np.uint8);bits[:,0]=1;bits[np.arange(5),np.arange(1,6)]=1
    engine=PackedMetrics(np.packbits(bits,axis=1,bitorder='little'),threads=2)
    a,b=np.nonzero(~np.eye(5,dtype=bool));a=np.tile(a,64);b=np.tile(b,64)
    labels=np.array([0,0,0,1,2],dtype=np.uint8)
    result=global_score(engine.pairs(a,b),a,b,labels)
    assert result['global_relative']==pytest.approx(1.)
    zero=global_score(np.zeros(len(a)),a,b,labels)
    assert zero['global_relative'] is None and zero['global_ci95'] is None
