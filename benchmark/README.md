# Hypergraph backend vs DataSAIL — benchmark

Head-to-head on the prepared 1D molecule datasets (`data/DataSAIL_data/1D/moleculenet`).
All methods are scored with **the same leakage metric**: the **scaled L(π)** used
in the DataSAIL paper/addendum (ECFP/Tanimoto, lower = less train/test leakage).
We compute it with `leakage.scaled_lpi`, a GPU, chunked reimplementation of
DataSAIL's `eval_split` that is numerically equal to it (`--validate`) but scales
to 100k+ entities, so every method can be scored on every dataset. Splits are 80/20.

Run:
```bash
python -m PALM.benchmark.benchmark_moleculenet1d            # full run -> moleculenet1d_results.csv
python -m PALM.benchmark.benchmark_moleculenet1d --validate # check scaled_lpi == eval_split
python -m PALM.benchmark.make_chart           # moleculenet1d_results.csv -> benchmark_chart.png
```

## Results (scaled L(π), lower is better)

Numbers are read from `moleculenet1d_results.csv`; the chart is `benchmark_chart.png`.

| dataset | n | **hypergraph** | DataSAIL (fresh, v1.3.0) | random | paper S1 | hg time | DataSAIL time |
|---|---:|---:|---:|---:|---:|---:|---:|
| freesolv | 642 | 0.1509 | **0.1424** | 0.3146 | 0.1410 | 3.4s | 8.2s |
| esol | 1,128 | **0.1653** | 0.1668 | 0.3154 | 0.1808 | 0.1s | 16.2s |
| clintox | 1,484 | **0.2150** | 0.2294 | 0.3146 | 0.2303 | 0.2s | 22.6s |
| sider | 1,427 | 0.2378 | **0.2360** | 0.3175 | 0.2345 | 0.1s | 6.6s |
| bace | 1,513 | 0.2628 | **0.2387** | 0.3182 | 0.3036 | 0.1s | 306s |
| bbbp | 2,050 | 0.2338 | **0.2319** | 0.3150 | 0.2866 | 0.1s | 71s |
| lipophilicity | 4,200 | **0.2546** | 0.2718 | 0.3186 | 0.3027 | 0.3s | 323s |
| tox21 | 7,831 | 0.2239 | **0.2230** | 0.3174 | 0.2224 | 0.6s | 65s |
| qm8 | 21,766 | **0.1818** | 0.2077 | 0.3199 | 0.2918 | **1.8s** | **742s** |
| hiv | 41,127 | **0.2498** | timeout (600s) | 0.3196 | 0.3071 | **3.8s** | ✗ |
| muv | 93,087 | **0.2484** | timeout (600s) | 0.3204 | 0.3143 | **10.2s** | ✗ |

## Key findings

1. **Comparable quality, often better on larger/more-diverse data.** Hypergraph
   wins clearly on clintox, lipophilicity, and qm8, edges esol, and is within
   ~0.002 of DataSAIL on sider/bbbp/tox21 (a statistical tie). DataSAIL wins on
   freesolv and bace (small, congeneric). Both beat random (~0.31–0.32) and the
   hypergraph beats the paper S1 number on 8 of 11 datasets.
2. **Orders of magnitude faster.** Hypergraph: 0.1–10.2 s across *all* sizes.
   DataSAIL: seconds at <2k, **1–12 min at 1.5k–22k**, and **times out (600 s)
   at 41k and 93k** — its O(n²) clustering + ILP is the bottleneck.
3. **Scales where DataSAIL can't.** Hypergraph split MUV's 93k molecules in
   10.2 s; DataSAIL did not finish on HIV (41k) or MUV (93k).

## Why DataSAIL sometimes wins (bace, freesolv)

- **Objective vs metric:** the hypergraph minimizes a *sparse k-NN cut*, but
  L(π) scores the *full pairwise* similarity. On dense/congeneric sets (bace)
  the k-NN proxy under-counts diffuse similarity; larger k closes most of the
  gap (bace improves as k grows from 5→60).
- **Exact ILP vs heuristic:** DataSAIL solves near-optimally on small instances
  (bace took 306 s); Mt-KaHyPar is a fast multilevel heuristic. This advantage
  vanishes — and reverses — as n grows.
- **Construction vs scoring space:** the hypergraph is built on 2048-bit Morgan;
  `scaled_lpi` scores on DataSAIL's 1024-bit ECFP — a small space mismatch.

## Notes

- **Metric parity.** `scaled_lpi` is validated against DataSAIL's `eval_split`
  (`python -m PALM.benchmark.benchmark_moleculenet1d --validate`); they agree to <1e-3 on the
  small datasets where eval_split is feasible. eval_split itself is O(n²) on CPU
  and raises/timeouts past ~20–40k, which is why it is not used directly here.
- **Determinism.** Mt-KaHyPar is initialized once per process
  (`hypergraph._get_initializer`) and seeded (`set_seed`), so results are
  reproducible run-to-run. (An earlier harness re-initialized it per dataset,
  which inflated the bace number to ~0.305; that is fixed.)
