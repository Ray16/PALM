"""Verify count matching and weighted metrics against explicit tiny examples."""
import itertools
import unittest
import numpy as np

from compare_original import assign_exact, metrics, weighted_cut


class ComparisonTests(unittest.TestCase):
    def test_weighted_cut_equals_expanded_configuration_cut(self):
        B = np.array([[1., 0.], [.2, .8], [.5, .5]])
        w = np.array([2, 3, 1])
        lab = np.array([0, 1, 0])
        expanded = np.repeat(B, w, axis=0)
        z = np.repeat(lab, w)
        expected = (expanded @ expanded.T)[z[:, None] != z[None, :]].sum() / 2
        self.assertAlmostEqual(weighted_cut(B, w, lab), expected)

    def test_exact_assignment_matches_small_exhaustive_knapsack(self):
        w = np.array([2, 3, 4, 5])
        cost = np.array([1., -2., 3., 0.])
        lab, status = assign_exact(cost, w, 7)
        candidates = [np.array(v) for v in itertools.product((0, 1), repeat=4) if w @ np.array(v) == 7]
        self.assertEqual(w @ lab, 7)
        self.assertAlmostEqual(cost @ lab, min(cost @ c for c in candidates))

    def test_weighted_metrics_match_explicit_original_overlap(self):
        K = np.array([[1., .2], [.2, 1.]])
        masses = np.array([[2, 1], [1, 2]])
        expanded = np.repeat(np.arange(2), masses.sum(1))
        labels = np.array([0, 0, 1, 0, 1, 1])
        S = K[expanded[:, None], expanded[None, :]]
        np.fill_diagonal(S, 0)
        expected = S[labels[:, None] != labels[None, :]].sum() / S.sum()
        r = metrics(K, masses)
        self.assertAlmostEqual(r['global_share'], expected)
        self.assertEqual(r['overlapping_groups'], 2)
        self.assertEqual(r['val_nearest_train_similarity_frame_weighted'], 1.)


if __name__ == '__main__':
    unittest.main()
