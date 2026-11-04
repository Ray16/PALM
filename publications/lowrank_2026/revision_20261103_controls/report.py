"""Validate the complete control inventory and generate publication tables."""
from __future__ import annotations

from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

import numpy as np

HERE=Path(__file__).resolve().parent
PUB=HERE.parent
PAPER=PUB/'paper'
sys.path.insert(0,str(HERE))
import suite

NAMES={"moleculenet_"+s:n for s,n in [('freesolv','FreeSolv'),('esol','ESOL'),
       ('bace','BACE'),('bbbp','BBBP'),('lipophilicity','Lipophilicity'),('tox21','Tox21')]}


def mean(rows,key):
    groups=defaultdict(list)
    for row in rows:
        v=row.get(key)
        if v is not None:groups[row['dataset']].append(v)
    return float(np.mean([np.mean(x) for x in groups.values()])) if groups else None


def number(v,digits=3):
    return '--' if v is None else f'{v:.{digits}f}'


def table(name,headers,rows,caption,label=None,long=False):
    kind='longtable' if long else 'tabular'
    text=[] if long else [r'\begin{table}[htbp]\centering\small']
    text+=[r'\begin{'+kind+'}{l'+'r'*(len(headers)-1)+'}']
    if long:text+=[r'\caption{'+caption+r'}\\']
    text += [r'\toprule',' & '.join(headers)+r'\\\midrule']
    if long:
        text += [r'\endfirsthead',r'\toprule',' & '.join(headers)+r'\\\midrule',
                 r'\endhead',r'\midrule\endfoot',r'\bottomrule\endlastfoot']
    text+=[' & '.join(map(str,row))+r'\\' for row in rows]
    text += ([r'\end{longtable}'] if long else [r'\bottomrule\end{tabular}',r'\caption{'+caption+'}',
              r'\label{'+label+'}' if label else '',r'\end{table}'])
    (PAPER/name).write_text('\n'.join(text)+'\n')


def verified():
    root=HERE/'run';m=json.loads((root/'manifest.json').read_text())
    if m['sources']!=suite.sources():raise RuntimeError('new-control producer source drift')
    if len(m['jobs'])!=432:raise RuntimeError('unexpected grid')
    rows=[]
    for job in m['jobs']:
        p=root/'rows'/(job['name']+'.json')
        if not p.exists():raise RuntimeError('missing case: '+job['name'])
        r=json.loads(p.read_text())
        if r['job']!=job or r['status']!='ok':raise RuntimeError('invalid case')
        if r['manifest_sha256']!=suite.sha(root/'manifest.json'):raise RuntimeError('manifest identity')
        if r['host']!=m['host'] or ','.join(map(str,r['cpu_affinity'])) not in m['cpus']:
            raise RuntimeError('wrong host/CPU allocation')
        for path,digest in r['artifacts'].items():
            if suite.sha(root/path)!=digest:raise RuntimeError('artifact drift: '+path)
        if job['study'] in ('iterations','clock'):
            labels=np.load(root/'assignments'/job['name']/'labels.npy')
            initial=np.load(root/'assignments'/job['name']/'initial.npy')
            if not np.array_equal(initial,suite.init_labels(r['n'],job['seed'])):
                raise RuntimeError('initialization mismatch')
            suite.check(labels,r['n'],job['epsilon'])
            if np.bincount(labels,minlength=3).tolist()!=r['result']['metrics']['counts']:
                raise RuntimeError('saved counts mismatch')
        rows.append(r)
    failures=list((root/'failures').glob('*.json'))
    if failures:raise RuntimeError('unresolved production failures')
    return m,rows


def flat_rows(rows):
    out=[]
    for r in rows:
        job=r['job'];res=r['result']
        if job['study'] in ('iterations','clock'):
            out.append(dict(**job,**res['metrics'],n=r['n'],seconds=res['seconds'],
                       rss=res['native_peak_rss_gib'],storage_bytes=res['diagnostics']['storage_bytes'],
                       construction_seconds=res['construction_seconds'],overrun_seconds=res['overrun_seconds']))
        else:
            for v in res['variants']:
                out.append(dict(**job,variant=v['variant'],n=r['n'],**v['metrics'],
                           seconds=v.get('seconds',v.get('optimization_seconds')),
                           construction_seconds=v.get('construction_seconds'),rss=v.get('native_peak_rss_gib'),
                           matched_status=res.get('matched_status'),
                           group_count=res.get('group_count'),largest_group=res.get('largest_group')))
    return out


def fidelity():
    original=PUB/'revision_20261026/fidelity/run'
    m=json.loads((original/'manifest.json').read_text());items=[]
    for p in sorted((original/'rows').glob('*.json')):
        r=json.loads(p.read_text())
        if r['manifest_sha256']!=suite.sha(original/'manifest.json'):
            raise RuntimeError('fidelity manifest identity')
        for v in r['rows']:
            absolute=abs(v['true_cut']-v['approximate_cut'])
            items.append(dict(dataset=r['dataset'],seed=r['seed'],rank=v['rank'],
                         power=v['power'],partition=v['partition'],partition_kind=v['partition_kind'],
                         true_cut=v['true_cut'],approximate_cut=v['approximate_cut'],absolute_error=absolute,
                         mass_relative_error=absolute/v['total_mass'],
                         cut_relative_error=absolute/v['true_cut'] if v['true_cut']>0 else None))
    if len(items)!=900:raise RuntimeError('fidelity inventory')
    return items


def write_csv(path,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=keys,lineterminator='\n');writer.writeheader();writer.writerows(rows)


def generate():
    m,raw=verified();rows=flat_rows(raw)
    analysis=HERE/'analysis';analysis.mkdir(exist_ok=True)
    write_csv(analysis/'all_observations.csv',rows)
    fid=fidelity();write_csv(analysis/'full_fidelity_cut_errors.csv',fid)
    iterations=[r for r in rows if r['study']=='iterations']
    dense={(r['dataset'],r['seed'],r['power'],r['epsilon']):r for r in iterations if r['arm']=='dense'}
    factors=[r for r in iterations if r['arm']=='factor']
    pairs=[]
    for r in factors:
        ref=dense[(r['dataset'],r['seed'],r['power'],r['epsilon'])]
        pairs.append(dict(**r,dense_global=ref['global_relative'],dense_cut=ref['true_powered_cut'],
                          true_cut_ratio=r['true_powered_cut']/ref['true_powered_cut'],
                          dense_seconds=ref['seconds'],time_ratio=r['seconds']/ref['seconds'],
                          rss_ratio=r['rss']/ref['rss'],storage_ratio=r['storage_bytes']/ref['storage_bytes']))
    write_csv(analysis/'dense_factor_pairs.csv',pairs)
    selected=[r for r in pairs if r['power']==1 and r['rank']==256 and r['epsilon']==.01]
    table_rows=[]
    for ds in NAMES:
        rr=[r for r in selected if r['dataset']==ds]
        table_rows.append([NAMES[ds],rr[0]['n'],number(mean(rr,'dense_global')),number(mean(rr,'global_relative')),
                           number(100*mean(rr,'cut_relative_error'),2),number(mean(rr,'dense_seconds'),2),number(mean(rr,'seconds'),2)])
    table('controls_dense_table.tex',['Dataset','$n$','Dense','Factor','Error (\\%)','Dense (s)','Factor (s)'],table_rows,
          'Direct dense and rank-256 factor optimization with identical initial assignments, '
          '25 assignment iterations at most, the same refinement limit, and a 1\\% size corridor. '
          'Dense and Factor are original-kernel global overlap relative to count-matched random. '
          'Error is factor cut error divided by the true cut at its returned assignment. '
          'Times include representation construction and optimization on two CPU cores, '
          'excluding loading, warm-up and independent scoring. Values average three seeds. '
          'These are heuristic solutions, not dense optima.',label='tab:dense-controls')
    detail=[]
    for power in [1,3]:
        for eps in [0.,.01]:
            for rank in [64,256,512]:
                rr=[r for r in pairs if r['power']==power and r['epsilon']==eps and r['rank']==rank]
                detail.append([power,number(eps,2),rank,number(mean(rr,'true_cut_ratio')),
                               number(100*max(r['cut_relative_error'] for r in rr),2),
                               number(mean(rr,'time_ratio')),number(mean(rr,'storage_ratio'))])
    table('controls_rank_table.tex',['Power',r'$\epsilon$','Rank','Cut / dense','Max error (\\%)','Time / dense','Storage / dense'],detail,
          'Iteration-controlled dense comparisons over six datasets and three seeds. Ratios are '
          'averaged within datasets before averaging datasets. Cut / dense compares original '
          'powered-kernel cuts of the separately optimized partitions; a value below one can '
          'reflect heuristic search trajectories and does not prove approximation superiority. '
          'Max error is the largest cut-relative representation error across returned partitions. '
          'Storage counts the float32 kernel or factor array only, not process RSS.',long=True)
    memory=[]
    for ds in NAMES:
        for arm in ['dense','factor']:
            rr=[r for r in iterations if r['dataset']==ds and r['power']==1 and r['epsilon']==.01
                and r['arm']==arm and r['rank']==(0 if arm=='dense' else 256)]
            memory.append([NAMES[ds],arm,number(mean(rr,'rss')),number(mean(rr,'storage_bytes')/2**20,2),
                           number(mean(rr,'construction_seconds'),2),number(mean(rr,'seconds'),2)])
    table('controls_memory_table.tex',['Dataset','Arm','RSS (GiB)','Array (MiB)','Build (s)','Total (s)'],memory,
          'Fresh-process profiles for the main dense control. RSS is recorded before any independent '
          'scoring, includes imports, inputs and identical warm-up, and is not an incremental '
          'representation-memory measurement. Array bytes are reported separately. Mean of three seeds.',long=True)
    clock=[r for r in rows if r['study']=='clock'];clock_pairs=[]
    for ds in NAMES:
        for budget in [1.,5.]:
            rr=[r for r in clock if r['dataset']==ds and r['budget_seconds']==budget]
            d=[r for r in rr if r['arm']=='dense'];f=[r for r in rr if r['arm']=='factor']
            clock_pairs.append([NAMES[ds],int(budget),number(mean(d,'global_relative')),number(mean(f,'global_relative')),
                                number(max(r['overrun_seconds'] for r in d),3),number(max(r['overrun_seconds'] for r in f),3)])
    table('controls_clock_table.tex',['Dataset','Budget (s)','Dense','Factor','Dense overrun','Factor overrun'],clock_pairs,
          'Equal-clock controls including representation construction. Both arms repeatedly call '
          'the current one-step assignment and refinement routines, using the same restart-seed '
          'sequence. Clocks are checked between operations: the last operation may overrun, and '
          'the table reports maximum overrun in seconds. If construction exhausts the budget, '
          'the unchanged feasible initial partition is returned. Global scores average three seeds. '
          'These cooperative budgets are not hard real-time deadlines.',long=True)
    abl=[r for r in rows if r['study']=='ablation'];variants=list(dict.fromkeys(r['variant'] for r in abl))
    abrows=[]
    names={'initial':'Feasible initialization','assignment_only':'Assignment only',
           'refinement_only':'Refinement only','one_restart':'Assignment + refinement',
           'four_restarts':'Four restarts / two refined','uniform_landmarks':'Uniform landmarks / four starts',
           'power_only':'Cubed factor / four starts','group_initial':'Fixed groups / feasible packing',
           'groups_plain':'Fixed groups / ordinary factor','groups_power':'Fixed groups / cubed factor'}
    for v in variants:
        rr=[r for r in abl if r['variant']==v]
        abrows.append([names[v],number(mean(rr,'global_relative')),number(mean(rr,'mean_nearest')),
                       number(mean(rr,'seconds'),2)])
    table('controls_ablation_table.tex',['Setting','Global','Nearest','Optimize (s)'],abrows,
          'Current-version component controls at rank 256 and a 1\\% size corridor. Each cell averages '
          'three seeds within each of six datasets, then datasets equally. The first seven rows '
          'are ungrouped; the last three share the same strictly preserved capped groups and feasible '
          'group initialization. Grouping changes the feasible set. Optimization time excludes '
          'shared factor and group construction, which are retained per case. No response labels '
          'enter these controls. Restart budgets and group starts are specified in Methods.',long=True)
    cc=[r for r in rows if r['study']=='ccpart'];ccrows=[]
    for threshold in [.4,.6,.8]:
        for variant,label in [('ccpart','CCPart'),('matched_counts','Low rank, counts'),('matched_groups','Low rank, groups')]:
            rr=[r for r in cc if r['threshold']==threshold and r['variant']==variant]
            ccrows.append([number(threshold,1),label,number(mean(rr,'global_relative')),
                           number(mean(rr,'mean_nearest')),number(mean(rr,'seconds'),2)])
    table('controls_ccpart_table.tex',['Threshold','Assignment','Global','Nearest','Seconds'],ccrows,
          'CCPart and low-rank controls at its native counts, with and without its connected-component '
          'groups. Global is overlap relative to count-matched random; Nearest is mean maximum '
          'test-to-training Tanimoto similarity. Scores average input-order repetitions within datasets and datasets equally. '
          'Native runtime includes exhaustive threshold-similarity construction and component packing. '
          'Matched controls include factor construction and optimization; native groups are reused, '
          'so their clocks describe an optimization intervention, not a standalone grouping pipeline. '
          'Realized counts, retention, missing folds and per-dataset results are in the supplement.',label='tab:ccpart-controls')
    counts=[]
    for ds in NAMES:
        for threshold in [.4,.6,.8]:
            rr=[r for r in cc if r['dataset']==ds and r['threshold']==threshold and r['variant']=='ccpart']
            def ranges(k):
                vv=[r['counts'][k] for r in rr];return str(min(vv)) if min(vv)==max(vv) else f'{min(vv)}--{max(vv)}'
            counts.append([NAMES[ds],number(threshold,1),rr[0]['n'],ranges(0),ranges(1),ranges(2),
                           min(r['group_count'] for r in rr),max(r['largest_group'] for r in rr)])
    table('controls_ccpart_counts.tex',['Dataset',r'$\tau$','$n$','Train','Val','Test','Groups','Largest'],counts,
          'CCPart retains every entity. Count ranges span three input permutations. Group count and '
          'largest component show the native threshold contract. Whole components are retained; '
          'fold sizes are not silently repaired to the low-rank corridor.',long=True)
    native_cost=[]
    for ds in NAMES:
        for threshold in [.4,.6,.8]:
            rr=[r for r in cc if r['dataset']==ds and r['threshold']==threshold and r['variant']=='ccpart']
            native_cost.append([NAMES[ds],number(threshold,1),number(mean(rr,'global_relative')),
                                number(mean(rr,'mean_nearest')),number(mean(rr,'seconds'),2),number(mean(rr,'rss'))])
    table('controls_ccpart_native.tex',['Dataset',r'$\tau$','Global','Nearest','Seconds','RSS (GiB)'],native_cost,
          'Native CCPart quality and complete split-call cost, including exhaustive threshold similarities, '
          'connected components and packing. Memory is fresh-process peak RSS before scoring or matched '
          'interventions. Each row averages three input-order repetitions; all entities are retained.',long=True)
    fidrows=[]
    for power in [1,3]:
        for rank in [32,64,128,256,512]:
            rr=[r for r in fid if r['power']==power and r['rank']==rank]
            returned=[r for r in rr if r['partition_kind']=='returned']
            fidrows.append([power,rank,number(100*max(r['mass_relative_error'] for r in rr),2),
                            number(100*max(r['cut_relative_error'] for r in rr),2),
                            number(100*max(r['cut_relative_error'] for r in returned),2)])
    table('controls_cut_error_table.tex',['Power','Rank','Mass error (\\%)','Cut error (\\%)','Returned cut (\\%)'],fidrows,
          'Reanalysis of all 900 saved full-kernel fidelity comparisons: six datasets, three seeds, '
          'five ranks, two powers and five tested partitions. Entries are maxima. Mass and cut '
          'denominators use the same power as the factor. Returned cut restricts to the optimized '
          'partition; the other maxima include four count-preserving perturbations. '
          'All true cuts in this suite are positive; zero-cut relative errors would be undefined.',long=True)
    summary=dict(expected=len(m['jobs']),completed=len(raw),failures=0,
                 per_study={s:sum(r['job']['study']==s for r in raw) for s in ['iterations','clock','ablation','ccpart']},
                 matched_dense_rank256=dict(true_cut_ratio=mean(selected,'true_cut_ratio'),
                                           time_ratio=mean(selected,'time_ratio'),rss_ratio=mean(selected,'rss_ratio')),
                 worst_clock_overrun=max(r['overrun_seconds'] for r in clock),
                 ccpart_empty_folds=sum(r['matched_status']=='empty_native_fold' for r in cc if r['variant']=='ccpart'),
                 fidelity_comparisons=len(fid))
    suite.atomic(analysis/'summary.json',summary)
    macros=dict(ControlsCases=str(len(raw)),ControlsCutRatio=number(mean(selected,'true_cut_ratio'),4),
                ControlsTimeRatio=number(mean(selected,'time_ratio')),
                ControlsPlainCutError=number(100*max(r['cut_relative_error'] for r in fid if r['power']==1 and r['rank']==256),2),
                ControlsPowerCutError=number(100*max(r['cut_relative_error'] for r in fid if r['power']==3 and r['rank']==256),2),
                ControlsClockOverrun=number(summary['worst_clock_overrun'],2))
    (PAPER/'controls_numbers.tex').write_text('% Generated from frozen control observations.\n'+''.join(
        '\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in macros.items()))
    generated={p.name:suite.sha(p) for p in sorted(PAPER.glob('controls_*.tex')) if p.name not in ('controls_results.tex','controls_methods.tex','controls_supplement.tex')}
    suite.atomic(analysis/'provenance.json',dict(manifest_sha256=suite.sha(HERE/'run/manifest.json'),
                  report_sha256=suite.sha(__file__),generated_files=generated,
                  all_observations_sha256=suite.sha(analysis/'all_observations.csv'),
                  summary_sha256=suite.sha(analysis/'summary.json')))
    print(json.dumps(summary,indent=2))


def scaling_tables():
    import dense_scaling
    root=HERE/'scaling_run';m=json.loads((root/'manifest.json').read_text())
    if m['sources']!=dense_scaling.identities():raise RuntimeError('scaling source drift')
    if len(m['jobs'])!=36:raise RuntimeError('scaling inventory')
    rows=[]
    for j in m['jobs']:
        p=root/'rows'/(j['name']+'.json');r=json.loads(p.read_text())
        if r['job']!=j or r['status']!='ok' or r['manifest_sha256']!=suite.sha(root/'manifest.json'):
            raise RuntimeError('invalid scaling case')
        for name,digest in r['artifacts'].items():
            if suite.sha(root/name)!=digest:raise RuntimeError('scaling artifact drift')
        if r['host']!=m['host'] or ','.join(map(str,r['cpu_affinity'])) not in m['cpus']:
            raise RuntimeError('scaling resources')
        labels=np.load(root/'assignments'/j['name']/'labels.npy')
        initial=np.load(root/'assignments'/j['name']/'initial.npy')
        suite.check(labels,j['n'],.01)
        if not np.array_equal(initial,suite.init_labels(j['n'],j['seed'])):raise RuntimeError('scaling initial assignment')
        result=r['result'];rows.append(dict(**j,**result['metrics'],seconds=result['seconds'],
                     rss=result['native_peak_rss_gib'],array_bytes=result['diagnostics']['storage_bytes']))
    if list((root/'failures').glob('*.json')):raise RuntimeError('scaling failures')
    write_csv(HERE/'analysis/scaling_observations.csv',rows)
    table_rows=[];detail=[]
    for n in [4000,16000,32000]:
        dense=[r for r in rows if r['n']==n and r['arm']=='dense']
        factor=[r for r in rows if r['n']==n and r['rank']==256]
        ratio=np.mean([f['true_cut']/next(d['true_cut'] for d in dense if d['seed']==f['seed']) for f in factor])
        table_rows.append([f'{n:,}',number(np.mean([r['seconds'] for r in dense]),2),
                           number(np.mean([r['seconds'] for r in factor]),2),
                           number(np.mean([r['rss'] for r in dense]),2),
                           number(np.mean([r['rss'] for r in factor]),2),number(ratio,4)])
        for arm,rank in [('dense',0),('factor',64),('factor',256),('factor',512)]:
            rr=[r for r in rows if r['n']==n and r['arm']==arm and r['rank']==rank]
            detail.append([f'{n:,}',arm,rank or '--',number(np.mean([r['seconds'] for r in rr]),2),
                           number(np.mean([r['rss'] for r in rr]),2),number(np.mean([r['global_relative'] for r in rr])),
                           '--' if arm=='dense' else number(100*max(r['cut_relative_error'] for r in rr),2)])
    table('controls_scaling_table.tex',['Entities','Dense (s)','Factor (s)','Dense GiB','Factor GiB','Cut ratio'],table_rows,
          'Dense versus rank-256 factor optimization on nested ChEMBL samples, using identical '
          'initial assignments and search limits on two CPU cores with a 1\\% size corridor. '
          'Time includes representation construction and optimization; peak process RSS precedes '
          'independent scoring. Cut ratio divides the factor solution\'s original-kernel cut by '
          'the dense solution\'s cut. Entries average three seeds. Both solutions are heuristic.',label='tab:dense-scaling')
    table('controls_scaling_details.tex',['Entities','Arm','Rank','Seconds','RSS GiB','Global','Max error (\\%)'],detail,
          'Complete nested ChEMBL dense/factor grid. Global is original-kernel overlap normalized '
          'by count-matched random; error is maximum returned-cut-relative approximation error '
          'over three seeds. Timings and peak RSS are means. Both representations use the same '
          'bounded assignment/refinement protocol, with no response-label information.',long=True)
    largest=[r for r in rows if r['n']==32000]
    macros={}
    for arm,prefix in [('dense','Dense'),('factor','Factor')]:
        rr=[r for r in largest if r['arm']==arm and r['rank']==(0 if arm=='dense' else 256)]
        macros['ControlsLarge'+prefix+'Seconds']=number(np.mean([r['seconds'] for r in rr]),1)
        macros['ControlsLarge'+prefix+'RSS']=number(np.mean([r['rss'] for r in rr]),1)
    p=PAPER/'controls_numbers.tex'
    with p.open('a') as f:
        f.write(''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in macros.items()))
    generated={p.name:suite.sha(p) for p in sorted(PAPER.glob('controls_*.tex')) if p.name not in ('controls_results.tex','controls_methods.tex','controls_supplement.tex')}
    provenance=json.loads((HERE/'analysis/provenance.json').read_text())
    provenance.update(scaling_manifest_sha256=suite.sha(root/'manifest.json'),generated_files=generated,
                      scaling_observations_sha256=suite.sha(HERE/'analysis/scaling_observations.csv'))
    suite.atomic(HERE/'analysis/provenance.json',provenance)


def strict_clock_tables():
    import clock_budget
    root=HERE/'strict_clock_run';m=json.loads((root/'manifest.json').read_text())
    if m['sources']!=clock_budget.sources():raise RuntimeError('strict-clock producer drift')
    if len(m['jobs'])!=72:raise RuntimeError('strict-clock inventory')
    rows=[]
    for j in m['jobs']:
        p=root/'rows'/(j['name']+'.json');r=json.loads(p.read_text())
        if r['status']!='ok' or r['job']!=j or r['manifest_sha256']!=suite.sha(root/'manifest.json'):
            raise RuntimeError('invalid strict-clock case')
        if r['checkpoint']['elapsed']>j['budget_seconds']:raise RuntimeError('late selected checkpoint')
        if r['host']!=m['host'] or ','.join(map(str,r['cpu_affinity'])) not in m['cpus']:
            raise RuntimeError('strict-clock resource identity')
        for name,digest in r['artifacts'].items():
            if suite.sha(root/name)!=digest:raise RuntimeError('strict-clock artifact drift')
        rows.append(dict(**j,**r['metrics'],timed_process_seconds=r['timed_process_seconds'],
                         checkpoint_elapsed=r['checkpoint']['elapsed'],initial=r['checkpoint']['initial'],
                         termination_overrun_seconds=r['termination_overrun_seconds']))
    if list((root/'failures').glob('*.json')):raise RuntimeError('strict-clock failures')
    write_csv(HERE/'analysis/preempted_clock_observations.csv',rows)
    values=[]
    for ds in NAMES:
        for budget in [1.,5.]:
            rr=[r for r in rows if r['dataset']==ds and r['budget_seconds']==budget]
            d=[r for r in rr if r['arm']=='dense'];f=[r for r in rr if r['arm']=='factor']
            values.append([NAMES[ds],int(budget),number(mean(d,'global_relative')),number(mean(f,'global_relative')),
                           number(mean(d,'timed_process_seconds'),3),number(mean(f,'timed_process_seconds'),3)])
    table('controls_preempted_clock_table.tex',['Dataset','Budget (s)','Dense','Factor','Dense (s)','Factor (s)'],values,
          'Equal-time controls with an externally stopped optimizer. Representation construction and '
          'search share the budget; only a feasible checkpoint whose evaluation and serialization '
          'finished before the deadline is scored. Late work cannot improve the selected partition. '
          'Times include scheduling and termination overshoot; scores and times average three seeds. '
          'The initial feasible partition is available before timing starts.',long=True)
    max_overrun=max(r['termination_overrun_seconds'] for r in rows)
    with (PAPER/'controls_numbers.tex').open('a') as f:
        f.write('\\newcommand{\\ControlsPreemptedOverrun}{'+number(max_overrun,3)+'}\n')
    p=HERE/'analysis/provenance.json';provenance=json.loads(p.read_text())
    provenance['strict_clock_manifest_sha256']=suite.sha(root/'manifest.json')
    provenance['generated_files']={p.name:suite.sha(p) for p in sorted(PAPER.glob('controls_*.tex')) if p.name not in ('controls_results.tex','controls_methods.tex','controls_supplement.tex')}
    provenance['preempted_clock_observations_sha256']=suite.sha(HERE/'analysis/preempted_clock_observations.csv')
    suite.atomic(p,provenance)


if __name__=='__main__':
    generate()
    scaling_tables()
    strict_clock_tables()
