"""Scientific invariants for the metadata-only split and transfer boundary."""
import io
import unittest
from unittest.mock import Mock

from build_split import family_name, infer_host_formula, reduced_formula, fraction_matrix, UnionFind
from range_io import RangeReader


class MetadataTests(unittest.TestCase):
    def test_formula_is_supercell_invariant(self):
        self.assertEqual(reduced_formula('C24H16O16Zr4'), reduced_formula('C6H4O4Zr'))
        self.assertNotEqual(reduced_formula('C6H4O4Zr'), reduced_formula('C6H6O4Zr'))

    def test_reject_unparsed_formula(self):
        with self.assertRaises(ValueError):
            reduced_formula('C6H4O4Zr?')

    def test_adsorbate_placement_index_is_not_occupancy(self):
        p = 'ODAC25__mof_plus_adsorbate__'
        a = infer_host_formula(p + 'RACZIG_w_CO2_3', 'C13H8O10Zn2')
        b = infer_host_formula(p + 'RACZIG_w_CO2_2H2O_8', 'C13H12O12Zn2')
        self.assertEqual(a, reduced_formula('C12H8O8Zn2'))
        self.assertEqual(a, b)
        self.assertIsNone(infer_host_formula('ODAC25__gcmc__RACZIG_cut10_mof_w_ads_3', 'C13H8O10Zn2'))

    def test_related_name_conventions_stay_together(self):
        names = ['ODAC25__mof__RACZIG_0.08_0_w_CO2_3.traj',
                 'ODAC25__mof_plus_adsorbate__RACZIG01_0.12_1_eeen_19_w_H2O_1',
                 'ODAC25__gcmc__RACZIG_cut10.0_mol-1.00_p5000_t300_mof_w_ads_1']
        self.assertEqual({family_name(n)[1] for n in names}, {'RACZIG'})
        self.assertEqual(family_name('ODAC25__mof_plus_adsorbate__Gen180_H2O_single')[1], 'Gen180')

    def test_fractions_cover_rare_elements_without_imputation(self):
        X, names = fraction_matrix(['C2H2Np2'])
        self.assertAlmostEqual(float(X.sum()), 1, places=6)
        self.assertAlmostEqual(float(X[0, names.index('Np')]), 1 / 3, places=6)

    def test_alias_union_is_transitive(self):
        uf = UnionFind()
        uf.union('B', 'C')
        uf.union('A', 'B')
        self.assertEqual(uf.find('A'), uf.find('C'))

    def test_refuses_whole_file_response_before_reading(self):
        r = RangeReader('https://example.org/large.parquet', 1000000000)
        response = Mock(status_code=200)
        r.session = Mock()
        r.session.get.return_value = response
        with self.assertRaisesRegex(RuntimeError, 'non-range'):
            r.read(10)
        response.raw.read.assert_not_called()
        response.close.assert_called_once()

    def test_counts_exact_range_bytes_and_enforces_budget(self):
        r = RangeReader('https://example.org/large.parquet', 1000, max_bytes=4)
        r.session = Mock()
        r.session.get.return_value = Mock(status_code=206,
            headers={'Content-Range': 'bytes 990-993/1000'}, raw=io.BytesIO(b'abcd'))
        r.seek(-10, 2)
        self.assertEqual(r.read(4), b'abcd')
        self.assertEqual(r.bytes_read, 4)
        with self.assertRaisesRegex(RuntimeError, 'budget'):
            r.read(1)


if __name__ == '__main__':
    unittest.main()
