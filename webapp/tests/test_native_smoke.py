"""Small native checks of interaction adapters and the clustered DataSAIL path."""
import numpy as np
import pytest
from PALM.webapp import methods,worker
from PALM.benchmarks.datasail_pipeline.config import SplitContext,SplittingConfig


@pytest.mark.parametrize('adapter',[methods._split_hypergraph,methods._split_lowrank])
def test_interaction_adapter_preserves_unique_pairs(adapter):
    rng=np.random.default_rng(14)
    first={f'a{i}':rng.normal(size=8) for i in range(10)}
    second={f'b{i}':rng.normal(size=8) for i in range(12)}
    pairs=[(a,b) for a in first for b in second]
    ctx=SplitContext(False,first,second,pairs,{},None,None,SplittingConfig(techniques=['lowrank']))
    result=adapter(ctx)
    assert set(result)==set(pairs) and set(result.values())=={'train','test'}
    assert abs(sum(s=='test' for s in result.values())/len(pairs)-.2)<=.05


def test_clustered_datasail_accepts_custom_matrix_callable():
    from datasail.sail import datasail
    from datasail.solver.utils import solve
    names=[str(i) for i in range(20)];groups=np.array([0]*16+[1]*4)
    distance=(groups[:,None]!=groups[None,:]).astype(float)
    defaults=solve.__defaults__
    try:
        worker._configure_datasail_threads()
        result=datasail(techniques=['C1e'],splits=[8,2],names=['train','test'],
                       e_type='O',e_data={n:np.array([groups[i],1.]) for i,n in enumerate(names)},
                       e_clusters=2,e_dist=lambda:(names,distance),max_sec=10,epsilon=.01,solver='SCIP')[0]['C1e'][0]
        assert set(result)==set(names) and sum(s=='train' for s in result.values())==16
    finally:
        solve.__defaults__=defaults
