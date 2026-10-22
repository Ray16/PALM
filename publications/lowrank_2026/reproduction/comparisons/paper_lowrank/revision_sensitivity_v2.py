"""Graph-degree sensitivity and a longer-search optimization control.

Graph k=16/64/128 complements prespecified k=32 on six full development sets.
Both unmodified native labels and count-repaired labels are scored. FreeSolv
also compares four versus thirty restarts, with five fixed factor/split seeds.
These searches measure empirical improvement, not a global optimality gap.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mlst_study as study
from revision_baselines import graph_cut
from revision_benchmark import score
from revision_external_control import source_files
from splitters import SplitSpec, split
from splitters.common.pairwise_similarity import pairwise_similarity

DEST = study.OUT / 'revision_20261021/sensitivity_v2'
DATASETS = ['moleculenet_' + n for n in ('freesolv', 'esol', 'bace', 'bbbp', 'clintox', 'lipophilicity')]


def run(dataset):
    directory = DEST / dataset
    directory.mkdir(parents=True, exist_ok=True)
    sources = source_files()
    for name in ('revision_sensitivity_v2.py', 'revision_baselines.py', 'revision_benchmark.py'):
        path = Path(__file__).with_name(name)
        sources[str(path.relative_to(study.ROOT))] = hashlib.sha256(path.read_bytes()).hexdigest()
    final = directory / 'result.json'
    if final.exists():
        if json.loads(final.read_text())['source_files'] != sources:
            raise RuntimeError('sensitivity result uses other sources')
        return
    b, ids, X, y, metric = study.prepare(dataset)
    study.atomic_json(directory/'ids.json', ids)
    exact = pairwise_similarity(torch.as_tensor(X), torch.as_tensor(X), metric).double().numpy()
    np.fill_diagonal(exact, 0.)
    mass = float(exact.sum() / 2)
    rows = []
    for k in (16, 64, 128):
        for seed in (0, 1, 2):
            stem = f'graph__k{k}__s{seed}'
            saved = directory / (stem + '.json')
            if saved.exists():
                row = json.loads(saved.read_text())
                if row.get('source_files') != sources:
                    raise RuntimeError('partial graph result uses other sources')
            else:
                start = time.perf_counter()
                codes, diag = graph_cut(X, metric, epsilon=.01, seed=seed, k=k, threads=2)
                seconds = time.perf_counter() - start
                native = study.NAMES[np.asarray(diag['native_labels'])]
                labels = study.NAMES[codes]
                np.save(directory / (stem+'.npy'), labels)
                np.save(directory / (stem+'__native.npy'), native)
                row = dict(study='graph_degree', k=k, seed=seed, split_s=seconds, source_files=sources,
                           diagnostics=diag, native=score(X,native,metric,seed),
                           repaired=score(X,labels,metric,seed),
                           quantized_retained_mass=diag['sum_integer_edge_weights']/diag['weight_scale']/mass,
                           retained_mass_error_bound=diag['graph_edges']*diag['max_weight_quantization_error']/mass)
                study.atomic_json(saved, study.clean(row))
            rows.append(row)
            print(dataset, 'graph', k, seed, flush=True)
    search = []
    if dataset == 'moleculenet_freesolv':
        for method in ('plain','combined'):
            for seed in range(5):
                for restarts in (4,30):
                    stem = f'{method}__s{seed}__restarts{restarts}'
                    saved = directory/(stem+'.json')
                    if saved.exists():
                        row = json.loads(saved.read_text())
                        if row.get('source_files') != sources:
                            raise RuntimeError('partial search result uses other sources')
                    else:
                        kwargs = dict(n_restarts=restarts, allowed_size_imbalance=0., score_exact_leakage=False)
                        if method=='combined':kwargs.update(kernel_power=3,must_link=.4)
                        start = time.perf_counter()
                        result = split('lowrank',b.feature_data,
                                       SplitSpec([8,1,1],study.NAMES.tolist(),seed=seed,epsilon=0.),**kwargs)
                        seconds = time.perf_counter()-start
                        labels = np.array([result.assignment[i] for i in ids])
                        np.save(directory/(stem+'.npy'), labels)
                        cross = exact[labels[:,None]!=labels[None,:]]
                        power = 3 if method=='combined' else 1
                        row = dict(study='long_search',method=method,seed=seed,restarts=restarts,split_s=seconds,source_files=sources,
                                   exact_cut=float(cross.sum()/2),exact_powered_cut=float((cross**power).sum()/2),
                                   diagnostics=result.diagnostics,**score(X,labels,metric,seed))
                        study.atomic_json(saved,study.clean(row))
                    search.append(row)
                    print(dataset,method,seed,restarts,flush=True)
    current = source_files()
    if any(sources[k]!=v for k,v in current.items()):
        raise RuntimeError('core sources changed during sensitivity run')
    result = dict(dataset=dataset,status='ok',n=len(ids),source_files=sources,graph_rows=rows,search_rows=search,
                  feature_sha256=hashlib.sha256(X.tobytes()).hexdigest(),ordinary_off_diagonal_mass=mass,
                  scope='six full development datasets; graph degree and native/repair effects; FreeSolv search saturation only')
    study.atomic_json(final,study.clean(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--datasets',nargs='+',default=DATASETS)
    args=parser.parse_args()
    if torch.cuda.is_available():raise RuntimeError('CPU-only sensitivity protocol')
    for dataset in args.datasets:run(dataset)
