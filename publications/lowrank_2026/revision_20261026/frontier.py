"""Native workflow frontiers and observational matched-DataSAIL-cluster controls."""

from __future__ import annotations

import importlib.metadata
import time

import numpy as np


def frontier_case(job, output, artifacts):
    from study import public_data, common_score, record_array, check_assignment
    from lowrank import split, SplitSpec
    from lowrank.splitters.common.balanced_assignment import capacity_corridor

    data = public_data(job["dataset"])
    X = data["X"]
    names = ["train", "val", "test"]
    seed = job["seed"]
    caps, floors = capacity_corridor(len(X), [8, 1, 1], 0.01)
    rows = []
    if job["method"] == "graph":
        from graph_adapter import graph_cut

        labels, diag = graph_cut(
            X, "tanimoto", epsilon=0.01, seed=seed, k=job["degree"], threads=2
        )
        native = np.array(diag.pop("native_labels"))
        for method, lab in [("native_graph", native), ("repaired_graph", labels)]:
            record_array(output, job, method, lab, artifacts)
            rows.append(
                dict(
                    method=method,
                    scores=common_score(X, lab, data),
                    counts=np.bincount(lab, minlength=3).tolist(),
                )
            )
        counts = np.bincount(labels, minlength=3)
        if not ((counts >= floors) & (counts <= caps)).all():
            raise AssertionError("graph count corridor failed")
        return dict(rows=rows, diagnostics=diag, seconds=diag["split_s"])
    if job["method"] in ("plain", "combined"):
        params = dict(rank=256, score_exact_leakage=False, allowed_size_imbalance=0.01)
        if job["method"] == "combined":
            params.update(kernel_power=3, must_link=0.4)
        start = time.perf_counter()
        result = split(
            "lowrank",
            dict(enumerate(X)),
            SplitSpec([8, 1, 1], names, seed=seed, epsilon=0.01),
            **params,
        )
        seconds = time.perf_counter() - start
        labels = np.array([names.index(result.assignment[i]) for i in range(len(X))])
        counts = np.bincount(labels, minlength=3)
        if not ((counts >= floors) & (counts <= caps)).all():
            raise AssertionError("low-rank count corridor failed")
        record_array(output, job, job["method"], labels, artifacts)
        return dict(
            rows=[
                dict(
                    method=job["method"],
                    scores=common_score(X, labels, data),
                    counts=counts.tolist(),
                )
            ],
            seconds=seconds,
            diagnostics=result.diagnostics,
            params=params,
        )

    from scipy.spatial.distance import pdist, squareform
    from datasail.sail import datasail
    import datasail.routine as routine

    # Observe the exact post-overflow dataset passed into DataSAIL's unchanged
    # solver. The wrapper neither changes arguments nor modifies its result.
    captured = {}
    original_solver = routine.run_solver
    original_cluster = routine.cluster

    def observe_cluster(*args, **kwargs):
        start = time.perf_counter()
        value = original_cluster(*args, **kwargs)
        captured["cluster_seconds"] = time.perf_counter() - start
        return value

    def observe_solver(*args, **kwargs):
        ds = kwargs["e_dataset"]
        captured["mapping"] = {
            str(i): str(ds.cluster_map[ds.id_map[str(i)]]) for i in range(len(X))
        }
        start = time.perf_counter()
        result = original_solver(*args, **kwargs)
        captured["solver_seconds"] = time.perf_counter() - start
        return result

    cluster_count = {
        "coarse": 9,
        "medium": max(9, len(X) // 50),
        "fine": max(9, len(X) // 20),
    }[job["cluster_policy"]]
    ids = [str(i) for i in range(len(X))]
    routine.cluster, routine.run_solver = observe_cluster, observe_solver
    start = time.perf_counter()
    try:
        t = time.perf_counter()
        distance = np.nan_to_num(squareform(pdist(X, metric="jaccard")), nan=1.0)
        distance_seconds = time.perf_counter() - t
        e_splits, _, _ = datasail(
            techniques=["C1e"],
            splits=[8, 1, 1],
            names=names,
            runs=1,
            solver="SCIP",
            max_sec=job["solver_seconds"],
            threads=2,
            e_type="O",
            e_data={name: name for name in ids},
            e_dist=(ids, distance),
            e_clusters=cluster_count,
            delta=0.1,
            epsilon=0.01,
            verbose="I",
        )
        seconds = time.perf_counter() - start
    finally:
        routine.cluster, routine.run_solver = original_cluster, original_solver
    raw = e_splits["C1e"][0]
    if set(raw) != set(ids) or not set(raw.values()) <= set(names):
        raise RuntimeError("DataSAIL did not produce a full valid assignment")
    native = np.array([names.index(raw[i]) for i in ids])
    counts = np.bincount(native, minlength=3)
    if min(counts) == 0:
        raise RuntimeError("DataSAIL returned an empty fold")
    record_array(output, job, "datasail", native, artifacts)
    rows.append(
        dict(
            method="datasail",
            seconds=seconds,
            counts=counts.tolist(),
            scores=common_score(X, native, data),
        )
    )
    group_labels = [captured["mapping"][i] for i in ids]
    _, gid = np.unique(group_labels, return_inverse=True)
    check_assignment(native, counts, gid)
    record_array(output, job, "native_groups", gid, artifacts)
    for method, group_map in [
        ("matched_counts", None),
        ("matched_native_groups", {i: int(gid[i]) for i in range(len(X))}),
    ]:
        params = dict(rank=256, score_exact_leakage=False)
        if group_map is not None:
            params.update(groups=group_map, group_policy="strict")
        started = time.perf_counter()
        result = split(
            "lowrank",
            dict(enumerate(X)),
            SplitSpec(counts.tolist(), names, seed=seed, epsilon=0),
            **params,
        )
        elapsed = time.perf_counter() - started
        labels = np.array([names.index(result.assignment[i]) for i in range(len(X))])
        check_assignment(labels, counts, gid if group_map is not None else None)
        record_array(output, job, method, labels, artifacts)
        rows.append(
            dict(
                method=method,
                seconds=elapsed,
                counts=counts.tolist(),
                scores=common_score(X, labels, data),
                diagnostics=result.diagnostics,
            )
        )
    captured.pop("mapping")
    return dict(
        rows=rows,
        seconds=seconds,
        distance_seconds=distance_seconds,
        cluster_count=cluster_count,
        observed_phases=captured,
        actual_groups=int(gid.max() + 1),
        datasail_version=importlib.metadata.version("datasail"),
        protocol="DataSAIL native pipeline includes distances, clustering and solver; imports/feature loading excluded. Low-rank controls exactly match native counts; second control also preserves every observed native solver group. Mean/linked cluster similarity and entity dense cut can still differ as objectives. Solver status remains in retained native log; assignment is not an optimality certificate.",
    )
