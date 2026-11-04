"""Batched shortlist search checked against independent dense-kernel costs."""
import numpy as np
import pytest
import torch
from scipy import sparse
from lowrank.factors import FactorOps
from lowrank.optimize import swap_polish


@pytest.mark.parametrize('csr', [False, True])
@pytest.mark.parametrize('seed', range(6))
def test_batched_search_preserves_scalar_choices_and_dense_cost(csr, seed):
    # Integer factors make gains exact, including ties, in float32 arithmetic.
    rng = np.random.default_rng(seed)
    B = rng.integers(0, 3, (60, 11)).astype(np.float32)
    K = B.astype(float) @ B.astype(float).T
    ops = FactorOps(sparse.csr_matrix(B) if csr else B)
    initial = rng.permutation(np.repeat(np.arange(3), [40, 10, 10]))
    result = []
    for batch in [None, ops.block]:
        lab = ops.labels(initial, 3)
        final, moves = swap_polish(ops.column, lab, ops.mass(lab, 3),
                                  ops.diagonal(), 3, max_swaps=30, sim_block=batch)
        final = final.cpu().numpy()
        assert np.array_equal(np.bincount(final), [40, 10, 10])
        assert K[final[:, None] != final].sum() <= K[initial[:, None] != initial].sum()
        result.append((final, moves))
    assert np.array_equal(result[0][0], result[1][0])
    assert result[0][1] == result[1][1]


@pytest.mark.parametrize('csr', [False, True])
def test_selected_blocks_match_full_kernel_across_tiles(csr):
    B = np.random.default_rng(11).normal(size=(35, 9)).astype(np.float32)
    ops = FactorOps(sparse.csr_matrix(B) if csr else B)
    rows = torch.tensor([32, 5, 8, 0, 2, 1, 4], device=ops.device)
    cols = torch.tensor([7, 0, 9], device=ops.device)
    got = ops.block(rows, cols, block_size=2).cpu().numpy()
    expected = (B @ B.T)[np.ix_(rows.cpu().numpy(), cols.cpu().numpy())]
    np.testing.assert_allclose(got, expected, rtol=2e-6, atol=2e-6)
