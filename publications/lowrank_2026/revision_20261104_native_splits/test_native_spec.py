from pathlib import Path
import sys
import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parent.parent/'package'))
from lowrank import split
from lowrank.splitters.common.balanced_assignment import capacity_corridor,target_sizes
from native_spec import NativeSplitSpec


@pytest.mark.parametrize('epsilon',[0.,.01])
def test_four_native_roles_keep_capacity_and_complete_coverage(epsilon,monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda,'is_available',lambda:False)
    X=(np.random.default_rng(8).random((200,32))<.3).astype(np.float32)
    ratios=[90,2,4,4];names=['train','valid','test-dev','test-challenge']
    result=split('lowrank',dict(enumerate(X)),NativeSplitSpec(ratios,names,seed=0),
                 rank=16,metric='tanimoto',allowed_size_imbalance=epsilon,score_exact_leakage=False,
                 n_restarts=1,refine_restarts=1)
    assert set(result.assignment)==set(range(len(X)))
    counts=np.array([sum(v==name for v in result.assignment.values()) for name in names])
    caps,floors=capacity_corridor(len(X),ratios,epsilon)
    assert ((counts>=floors)&(counts<=caps)).all()
    if epsilon==0:assert np.array_equal(counts,target_sizes(len(X),ratios))


def test_adapter_rejects_invalid_four_role_specs():
    with pytest.raises(ValueError):NativeSplitSpec([1,1,1,float('nan')],['a','b','c','d'])
    with pytest.raises(ValueError):NativeSplitSpec([1,1,1,1],['a','b','c','c'])
    with pytest.raises(ValueError):NativeSplitSpec([1,1,1,1],['a','b','c','d'],epsilon=2)
