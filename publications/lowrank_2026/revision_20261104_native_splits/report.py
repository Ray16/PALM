"""Replay saved native-split metrics and generate concise quality/cost tables."""
from __future__ import annotations
import argparse
from collections import defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path

import numpy as np

import run as producer
from prepare import atomic,sha,DATASETS

HERE=Path(__file__).resolve().parent
PAPER=HERE.parent/'paper'
NAMES=dict(pcqm4mv2='PCQM4Mv2',moses='MOSES',molpcba='ogbg-molpcba')
METHODS={'native':'Native','random':'Random','exact_assignment':'Exact assignment','corridor_plain':r'1\% corridor'}


def verified(allow_partial=False):
    root=HERE/'run';manifest=json.loads((root/'manifest.json').read_text())
    if producer.sources()!=manifest['sources']:raise RuntimeError('native benchmark producer drift')
    if len(manifest['jobs'])!=30:raise RuntimeError('native grid size')
    rows=[];failures=[];missing=[]
    for j in manifest['jobs']:
        path=root/'rows'/(j['name']+'.json');failure=root/'failures'/(j['name']+'.json')
        if not path.exists():
            if failure.exists():
                f=json.loads(failure.read_text())
                if f['status']!='end_to_end_budget_exhausted':raise RuntimeError('unresolved numerical failure '+j['name'])
                failures.append(f)
            else:missing.append(j['name'])
            continue
        r=json.loads(path.read_text())
        if r['job']!=j or r['status']!='ok' or r['manifest_sha256']!=sha(root/'manifest.json'):
            raise RuntimeError('invalid native observation')
        if r['host']!=manifest['host'] or r['cpu_affinity']!=manifest['cpus']:raise RuntimeError('resource mismatch')
        prep=HERE/'preparation'/(j['dataset']+'.json')
        if sha(prep)!=r['source_input_manifest_sha256']:raise RuntimeError('input preparation identity')
        prepared=json.loads(prep.read_text())
        for name,digest in r['artifacts'].items():
            if sha(root/name)!=digest:raise RuntimeError('native artifact drift '+name)
        assignment=np.load(root/'assignments'/j['name']/'labels.npz')['labels']
        k=len(r['fold_names']);counts=np.bincount(assignment,minlength=k)
        if len(assignment)!=r['n'] or counts.tolist()!=r['realized_counts'] or np.any(assignment>=k):raise RuntimeError('counts/coverage')
        from lowrank.splitters.common.balanced_assignment import capacity_corridor
        caps,floors=capacity_corridor(r['n'],r['native_counts'],r['size_tolerance'])
        if not ((counts>=floors)&(counts<=caps)).all():raise RuntimeError('native capacity check')
        if r['native_counts']!=prepared['retained_counts'] or r['n']!=prepared['retained_n']:
            raise RuntimeError('native retained population changed')
        if j['method']=='native':
            buffer=io.BytesIO();np.save(buffer,assignment)
            if hashlib.sha256(buffer.getvalue()).hexdigest()!=prepared['artifacts']['native.npy']:
                raise RuntimeError('published native assignment not preserved')
            if r['split_seconds'] is not None or r['peak_process_rss_gib'] is not None:
                raise RuntimeError('native creation cost was imputed')
        if j['method']=='random':
            native_path=root/'assignments'/f"{j['dataset']}__native__s0"/'labels.npz'
            native=np.load(native_path)['labels']
            expected=native[np.random.default_rng(j['seed']).permutation(len(native))]
            if not np.array_equal(assignment,expected):raise RuntimeError('count-matched random replay')
        block=np.load(root/'assignments'/j['name']/'pair_blocks.npz')['means']
        expectation=(r['n']**2-float(counts@counts))/(r['n']*(r['n']-1))
        score=block[1].mean()/block[0].mean()/expectation
        if abs(score-r['metrics']['global_relative'])>1e-12:raise RuntimeError('global aggregation replay')
        selections=np.random.default_rng(91).integers(0,64,(1000,64))
        bootstrap=block[1][selections].mean(1)/block[0][selections].mean(1)/expectation
        if not np.allclose(np.quantile(bootstrap,[.025,.975]),r['metrics']['global_ci95'],rtol=0,atol=1e-12):
            raise RuntimeError('global interval replay')
        audit=np.load(root/'assignments'/j['name']/'nearest_audit.npz')
        for c,fold in enumerate(r['metrics']['nearest_by_fold'],1):
            queries=audit[f'fold{c}_queries'];nearest=audit[f'fold{c}_nearest']
            eligible=np.flatnonzero(assignment==c)
            expected=np.random.default_rng(200003+100*c+j['seed']).choice(eligible,min(128,len(eligible)),replace=False)
            if not np.array_equal(queries,expected) or np.any(assignment[queries]!=c):raise RuntimeError('uniform query replay')
            if np.any(nearest<0) or np.any(nearest>1) or abs(nearest.mean()-fold['mean_nearest'])>1e-12:
                raise RuntimeError('nearest aggregation replay')
            if abs(np.mean(nearest>.6)-fold['fraction_gt06'])>1e-12 or abs(np.mean(nearest==1)-fold['fraction_eq1'])>1e-12:
                raise RuntimeError('nearest threshold replay')
            boot=np.random.default_rng(300007+c+j['seed']).choice(nearest,(1000,len(nearest)),replace=True).mean(1)
            if not np.allclose(np.quantile(boot,[.025,.975]),fold['nearest_ci95'],rtol=0,atol=1e-12):
                raise RuntimeError('query interval replay')
        aggregate=np.average([f['mean_nearest'] for f in r['metrics']['nearest_by_fold']],weights=counts[1:])
        if abs(aggregate-r['metrics']['mean_heldout_nearest'])>1e-12:raise RuntimeError('held-out weighted mean replay')
        rows.append(r)
    if missing and not allow_partial:raise RuntimeError('native cases still running/missing: '+str(missing))
    return manifest,rows,failures,missing


def csv_write(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        w=csv.DictWriter(f,keys,lineterminator='\n');w.writeheader();w.writerows(rows)


def tex_table(path,headers,rows,caption,label=None,long=False):
    kind='longtable' if long else 'tabular'
    lines=[] if long else [r'\begin{table}[htbp]\centering\small']
    lines+=[r'\begin{'+kind+'}{l'+'r'*(len(headers)-1)+'}']
    if long:lines+=[r'\caption{'+caption+r'}\\']
    lines+=[r'\toprule',' & '.join(headers)+r'\\\midrule']
    if long:lines+=[r'\endfirsthead',r'\toprule',' & '.join(headers)+r'\\\midrule',r'\endhead',r'\midrule\endfoot',r'\bottomrule\endlastfoot']
    lines+=[' & '.join(map(str,r))+r'\\' for r in rows]
    if long:lines+=[r'\end{longtable}']
    else:lines+=[r'\bottomrule\end{tabular}',r'\caption{'+caption+'}',r'\label{'+label+'}' if label else '',r'\end{table}']
    path.write_text('\n'.join(lines)+'\n')


def mean(values):return float(np.mean(values)) if values else None
def fmt(value,n=3):return '--' if value is None else f'{value:.{n}f}'


def main(args):
    manifest,rows,failures,missing=verified(args.allow_partial)
    out=HERE/'analysis';out.mkdir(exist_ok=True)
    csv_write(out/'cases.csv',[dict(**r['job'],n=r['n'],counts=r['realized_counts'],
        global_relative=r['metrics']['global_relative'],global_ci95=r['metrics']['global_ci95'],
        mean_heldout_nearest=r['metrics']['mean_heldout_nearest'],split_seconds=r['split_seconds'],
        peak_rss_gib=r['peak_process_rss_gib'],audit_seconds=r['audit_seconds']) for r in rows])
    per_fold=[]
    for r in rows:
        for f in r['metrics']['nearest_by_fold']:per_fold.append(dict(**r['job'],**f))
    csv_write(out/'heldout_folds.csv',per_fold)
    summary=dict(expected=len(manifest['jobs']),completed=len(rows),budget_outcomes=len(failures),missing=missing,
                 metered_metadata_bytes=json.loads((HERE/'transfer_ledger.json').read_text())['received_bytes'])
    atomic(out/'summary.json',summary);print(json.dumps(summary,indent=2))
    if not args.write_paper:return
    if missing:raise RuntimeError('cannot write manuscript from incomplete native results')
    main_rows=[];detail=[];by=defaultdict(list)
    for r in rows:by[(r['job']['dataset'],r['job']['method'])].append(r)
    for dataset in DATASETS:
        for method in ['native','exact_assignment','corridor_plain']:
            rr=by[(dataset,method)];meta=json.loads((HERE/'preparation'/f'{dataset}.json').read_text())
            main_rows.append([NAMES[dataset],f"{meta['retained_n']:,}",METHODS[method],
                fmt(mean([r['metrics']['global_relative'] for r in rr])),
                fmt(mean([r['metrics']['mean_heldout_nearest'] for r in rr])),
                fmt(mean([r['split_seconds'] for r in rr if r['split_seconds'] is not None]),1),
                fmt(mean([r['peak_process_rss_gib'] for r in rr if r['peak_process_rss_gib'] is not None]),1)])
        for method in ['native','random','exact_assignment','corridor_plain']:
            rr=by[(dataset,method)]
            for fold in json.loads((HERE/'preparation'/f'{dataset}.json').read_text())['fold_names'][1:]:
                ff=[f for r in rr for f in r['metrics']['nearest_by_fold'] if f['fold']==fold]
                detail.append([NAMES[dataset],METHODS[method],fold.replace('_',r'\_'),
                               len(rr),fmt(mean([f['mean_nearest'] for f in ff])),
                               fmt(mean([f['fraction_gt06'] for f in ff])),fmt(mean([f['fraction_eq1'] for f in ff]))])
    tex_table(PAPER/'native_split_table.tex',['Dataset','$n$','Assignment','Global','Nearest','Seconds','GiB'],main_rows,
              'Released partitions and low-rank controls on the same parsed records. Global is overlap '
              'relative to count-matched random. Nearest weights role means from 128 sampled queries '
              'against the complete training fold. Exact assignment preserves native counts and omits '
              'local refinement; the corridor uses 1\\% tolerance and the standard size cutoff. '
              'Low-rank rows average three seeds. Seconds cover the public call; GiB is peak process '
              'RSS before auditing. Native construction costs are unavailable. Parsing losses, per-role '
              'scores and sampling intervals are in the supplement.',
              label='tab:native-molecular')
    tex_table(PAPER/'native_split_roles.tex',['Dataset','Assignment','Role','Runs','Nearest',r'$>0.6$',r'$=1$'],detail,
              'Per-held-out-role exact nearest-training exposure on 128 uniform sampled queries per '
              'role and seed. Every query is compared with every returned training record. Fractions '
              'at one measure fingerprint identity, not identical geometries. Rows average completed '
              'seeds; full query-bootstrap intervals and per-seed observations accompany the evidence.',long=True)
    identity=[]
    for dataset in DATASETS:
        p=json.loads((HERE/'preparation'/f'{dataset}.json').read_text())
        identity.append([NAMES[dataset],f"{p['metadata']['original_n']:,}",f"{p['retained_n']:,}",p['parse_failures'],
                         f"{p['identity_audit']['unique_canonical_keys']:,}",f"{p['identity_audit']['canonical_groups_crossing_native_folds']:,}"])
    tex_table(PAPER/'native_split_identity.tex',['Dataset','Original','Retained','Failed','Unique keys','Crossing keys'],identity,
              'Metadata identity audit with the recorded RDKit version. Keys hash isomeric canonical '
              'SMILES; no chemical deduplication is silently applied. Crossing keys occur in more '
              'than one original native role. Parsing failures are fixed before all methods are '
              'compared, with original-row maps retained. Counts are molecular records, not atomistic frames.',long=True)
    outcome_rows=[];completed={r['job']['name']:r for r in rows};failed={r['job']['name']:r for r in failures}
    for j in manifest['jobs']:
        r=completed.get(j['name'])
        outcome_rows.append([NAMES[j['dataset']],METHODS[j['method']],j['seed'],
                             'Complete' if r else 'Budget',fmt(r['split_seconds'],1) if r else '--'])
    tex_table(PAPER/'native_split_outcomes.tex',['Dataset','Assignment','Seed','Outcome','Call (s)'],outcome_rows,
              'Complete native-split outcome inventory. Budget denotes the 1,200-second whole-case '
              'limit, including loading, warm-up, optimization and auditing; it is not an imputed '
              'split-call runtime. Native fixed-assignment construction times are unknown. '
              'Exact assignment explicitly disables refinement; the corridor setting uses the unchanged '
              'size-dependent refinement rule.',long=True)
    fidelity=[]
    for dataset in DATASETS:
        audit=[]
        for seed in range(3):
            path=HERE/'verification'/f'factor_fidelity_{dataset}__s{seed}.json'
            if not path.exists():raise RuntimeError('missing returned-cut native fidelity audit')
            record=json.loads(path.read_text());audit.extend(record['results'])
        for method in ['exact_assignment','corridor_plain']:
            rr=[r for r in audit if f'__{method}__' in r['case']]
            fidelity.append([NAMES[dataset],METHODS[method],
                             fmt(100*max(r['mass_relative_error'] for r in rr),2),
                             fmt(100*max(r['cut_relative_error'] for r in rr),2),
                             fmt(100*max(r['cut_error_ci95'][1] for r in rr),2)])
    tex_table(PAPER/'native_split_fidelity.tex',['Dataset','Setting','Mass error (\\%)','Cut error (\\%)','Upper CI (\\%)'],fidelity,
              'Post-hoc returned-cut fidelity at 256 landmarks. The factor is reconstructed only '
              'at uniform sampled pairs using saved landmarks and the pinned double-construction, '
              'float32-storage formula. Each cell is the largest estimate or interval endpoint over '
              'three seeds. The intervals are approximate Monte Carlo block-bootstrap intervals for '
              'the tested returned partition; they certify neither uniform fidelity nor optimizer optimality.',long=True)
    macros=dict(NativeMetadataMB=f"{summary['metered_metadata_bytes']/1e6:.1f}",NativeCompletedCases=str(len(rows)),
                NativeBudgetOutcomes=str(len(failures)),NativeLargestRecords=f"{manifest['inputs']['pcqm4mv2']['n']:,}")
    for method,prefix in [('native','Native'),('exact_assignment','Exact'),('corridor_plain','Corridor')]:
        rr=by[('pcqm4mv2',method)]
        macros['NativePcqm'+prefix+'Global']=fmt(mean([r['metrics']['global_relative'] for r in rr]))
        macros['NativePcqm'+prefix+'Nearest']=fmt(mean([r['metrics']['mean_heldout_nearest'] for r in rr]))
        if method!='native':
            macros['NativePcqm'+prefix+'Seconds']=fmt(mean([r['split_seconds'] for r in rr]),1)
            macros['NativePcqm'+prefix+'Memory']=fmt(mean([r['peak_process_rss_gib'] for r in rr]),1)
    (PAPER/'native_split_numbers.tex').write_text('% Generated from native-split observations.\n'+''.join(
        '\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in macros.items()))
    atomic(out/'provenance.json',dict(report_sha256=sha(__file__),manifest_sha256=sha(HERE/'run/manifest.json'),
           generated_files={p.name:sha(p) for p in PAPER.glob('native_split_*.tex') if p.name not in ['native_split_results.tex','native_split_supplement.tex']},
           observed_row_hashes={r['job']['name']:sha(HERE/'run/rows'/(r['job']['name']+'.json')) for r in rows}))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--allow-partial',action='store_true');p.add_argument('--write-paper',action='store_true')
    main(p.parse_args())
