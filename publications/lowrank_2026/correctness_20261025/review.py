"""Release audit: exact source identity, saved assignments and independent scores."""

from __future__ import annotations
import argparse
import ast
import hashlib
import json
from pathlib import Path
import sys
import subprocess
import benchmark


def executable_ast(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(
            node, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)
        ):
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body.pop(0)
    return ast.dump(tree, include_attributes=False)


def main(args):
    import numpy as np
    import torch

    sys.path.insert(0, str(args.package))
    from lowrank import split, SplitSpec
    from lowrank.grouped import must_link_groups
    from lowrank.splitters.common.balanced_assignment import target_sizes
    from lowrank.splitters.common.pairwise_similarity import pairwise_similarity

    manifest = json.loads((args.run / "manifest.json").read_text())
    current = benchmark.source_identity(args.package)
    expected = manifest["sources"]["new"]
    exact_source = current == expected
    doc_changes = []
    if not exact_source:
        alignment = json.loads(
            (Path(__file__).parent / "verification/doc_alignment.json").read_text()
        )
        if set(current["files"]) != set(expected["files"]):
            raise RuntimeError("release source module set changed")
        for name, digest in expected["files"].items():
            if current["files"][name] == digest:
                continue
            repository_path = "publications/lowrank_2026/package/" + name
            before = subprocess.check_output(
                ["git", "show", alignment["numerical_commit"] + ":" + repository_path],
                cwd=args.package,
            )
            if hashlib.sha256(before).hexdigest() != digest or executable_ast(
                before
            ) != executable_ast((args.package / name).read_bytes()):
                raise RuntimeError("executable source changed: " + name)
            doc_changes.append(name)
    if (
        benchmark.sha(Path(__file__).with_name("benchmark.py"))
        != manifest["runner_sha256"]
    ):
        raise RuntimeError("benchmark runner drift")
    previous = Path(manifest["implementations"]["previous"])
    if benchmark.source_identity(previous) != manifest["sources"]["previous"]:
        raise RuntimeError("previous-version source drift")
    rows = []
    by_dataset = {}
    strict = 0
    for job in manifest["jobs"]:
        path = args.run / "rows" / (job["name"] + ".json")
        r = json.loads(path.read_text())
        if r["status"] != "ok":
            raise RuntimeError("unavailable final case: " + job["name"])
        if r["manifest_sha256"] != benchmark.sha(args.run / "manifest.json"):
            raise RuntimeError("row manifest drift")
        if r["source_sha256"] != manifest["sources"][job["implementation"]]["sha256"]:
            raise RuntimeError("row source drift")
        array = path.with_suffix(".npy")
        if benchmark.sha(array) != r["assignment_sha256"]:
            raise RuntimeError("assignment drift")
        if job["dataset"] not in by_dataset:
            meta = json.loads(
                (args.run / "inputs" / f"{job['dataset']}.json").read_text()
            )
            cache = args.run / "cache" / f"{job['dataset']}.npz"
            if benchmark.sha(cache) != meta["cache_sha256"]:
                raise RuntimeError("input drift")
            data = np.load(cache)
            X, ids = data["X"], data["ids"]
            if len(np.unique(ids)) != len(ids) or not np.all((X == 0) | (X == 1)):
                raise RuntimeError("input identity/kernel contract")
            ref = torch.as_tensor(X)
            mass = 0.0
            for start in range(0, len(X), 256):
                S = pairwise_similarity(ref[start : start + 256], ref, "tanimoto")
                i = torch.arange(len(S))
                S[i, i + start] = 0
                mass += float(S.sum(dtype=torch.float64)) / 2
            by_dataset[job["dataset"]] = dict(
                X=X, ids=ids, y=data["y"], maccs=data["maccs"], mass=mass
            )
        d = by_dataset[job["dataset"]]
        labels = np.load(array)
        if labels.shape != (len(d["ids"]),) or not np.array_equal(
            np.bincount(labels, minlength=3), target_sizes(len(labels), [8, 1, 1])
        ):
            raise RuntimeError("saved assignment violates retention/counts")
        if job["setting"] == "new_default_strict":
            if "gid" not in d:
                cap = int(0.5 * target_sizes(len(labels), [8, 1, 1]).min())
                d["gid"] = must_link_groups(d["X"], "tanimoto", 0.4, cap, k=32)
            gid = d["gid"]
            low = np.full(int(gid.max()) + 1, 3)
            high = np.full(len(low), -1)
            np.minimum.at(low, gid, labels)
            np.maximum.at(high, gid, labels)
            if not np.array_equal(low, high):
                raise RuntimeError("strict group was cut")
            strict += 1
        rows.append(r)
    # This separate NumPy implementation checks all 42 cases on two full inputs.
    maximum_error = 0.0
    dense_cases = 0
    for dataset in ["moleculenet_freesolv", "moleculenet_clintox"]:
        d = by_dataset[dataset]
        X = d["X"].astype(float)
        dot = X @ X.T
        union = X.sum(1)[:, None] + X.sum(1)[None, :] - dot
        S = np.divide(dot, union, out=np.zeros_like(dot), where=union > 0)
        np.fill_diagonal(S, 0)
        W = np.array(benchmark.WEIGHTS)
        for row in [r for r in rows if r["dataset"] == dataset]:
            labels = np.load(args.run / "rows" / (row["name"] + ".npy"))
            exact = S[labels[:, None] != labels[None, :]].sum() / S.sum()
            weighted = (S * W[labels[:, None], labels[None, :]]).sum() / S.sum()
            near = S[labels == 2][:, labels == 0].max(1)
            ramp = np.maximum(0, (near - 0.4) / 0.6).mean()
            error = max(
                abs(exact - row["scores"]["global_ratio"]),
                abs(weighted - row["scores"]["weighted_global_ratio"]),
                abs(ramp - row["scores"]["local_ramp_04"]),
            )
            if error > 1e-6:
                raise RuntimeError("common scorer disagrees with dense NumPy reference")
            maximum_error = max(maximum_error, float(error))
            dense_cases += 1
    fidelity = []
    for row in [r for r in rows if r["setting"] == "new_default_cut"]:
        mass = by_dataset[row["dataset"]]["mass"]
        actual = (
            abs(
                row["diagnostics"]["factor_leakage"]
                - row["scores"]["global_ratio"] * mass
            )
            / mass
        )
        factor = row["diagnostics"]["factorization"]
        summary = factor["audit"]["relative_pair_mae"]
        if factor["audit_tolerance_met"] and actual > factor["rank_tolerance"] + 1e-6:
            raise RuntimeError("audited cut tolerance failed on exact benchmark score")
        fidelity.append(
            dict(
                dataset=row["dataset"],
                seed=row["seed"],
                actual_cut_error=float(actual),
                reported_interval=summary["ci95"],
                audit_tolerance_met=factor["audit_tolerance_met"],
            )
        )
    # A representative end-to-end replay complements exact source/array checks.
    replay = []
    for dataset in ["moleculenet_freesolv", "moleculenet_clintox"]:
        d = by_dataset[dataset]
        data = {int(i): x for i, x in zip(d["ids"], d["X"])}
        for setting, (implementation, params) in benchmark.SETTINGS.items():
            if implementation != "new":
                continue
            result = split(
                "lowrank",
                data,
                SplitSpec([8, 1, 1], ["train", "val", "test"], seed=0),
                score_exact_leakage=False,
                **params,
            )
            names = ["train", "val", "test"]
            labels = np.array(
                [names.index(result.assignment[int(i)]) for i in d["ids"]],
                dtype=np.uint8,
            )
            expected = np.load(args.run / "rows" / f"{dataset}__s0__{setting}.npy")
            if not np.array_equal(labels, expected):
                raise RuntimeError("end-to-end replay changed assignment")
            replay.append(dict(dataset=dataset, setting=setting, assignment_equal=True))
    predictions = json.loads((args.predictions / "manifest.json").read_text())
    transferred = 0
    for job in predictions["jobs"]:
        row = json.loads(
            (args.predictions / "rows" / (job["name"] + ".json")).read_text()
        )
        name = f"{job['dataset']}__s{job['seed']}__{job['setting']}"
        final = json.loads((args.run / "rows" / (name + ".json")).read_text())
        original = json.loads(
            (args.prediction_inputs / "rows" / (name + ".json")).read_text()
        )
        if (
            final["assignment_sha256"] != row["assignment_sha256"]
            or final["feature_sha256"] != original["feature_sha256"]
        ):
            raise RuntimeError("predictive-check inputs differ from final release grid")
        if benchmark.sha(args.run / "cache" / f"{job['dataset']}.npz") != benchmark.sha(
            args.prediction_inputs / "cache" / f"{job['dataset']}.npz"
        ):
            raise RuntimeError("predictive-check feature/target cache drift")
        transferred += 1
    record = dict(
        final_cases=len(rows),
        retained_rows_and_exact_counts=len(rows),
        strict_group_artifacts=strict,
        dense_reference_cases=dense_cases,
        maximum_dense_score_error=maximum_error,
        source_exact_match=exact_source,
        non_doc_executable_source_match=True,
        documentation_only_changes=doc_changes,
        replays=replay,
        cut_fidelity=fidelity,
        predictive_inputs_identical_to_final_cases=transferred,
        predictive_note="Models were fit on the preceding grid; every feature/target cache and assignment used by all 90 fits is byte-identical to the corresponding final-source case.",
    )
    benchmark.atomic(args.run / "pipeline_review.json", record)
    print(
        json.dumps(
            {k: v for k, v in record.items() if k not in ["replays", "cut_fidelity"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=Path, required=True)
    p.add_argument("--package", type=Path, required=True)
    p.add_argument("--predictions", type=Path, required=True)
    p.add_argument("--prediction-inputs", type=Path, required=True)
    main(p.parse_args())
