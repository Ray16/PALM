"""Validate split-call timing at exact native DataSAIL counts on one host."""
from pathlib import Path
import hashlib
import json
import re
import argparse
import numpy as np
import pandas as pd

PAPER=Path(__file__).resolve().parents[1]
D=PAPER/'revision_20261021'
DATASETS=['moleculenet_'+n for n in ('clintox','bace','bbbp','lipophilicity')]


def main(allow_partial=False):
    issues,rows,outcomes,hashes=[],[],[],{}
    for dataset in DATASETS:
        root=D/'runtime'/dataset;path=root/'result.json'
        if not path.exists():issues.append('missing runtime case '+dataset);continue
        r=json.loads(path.read_text());hashes[dataset]=hashlib.sha256(path.read_bytes()).hexdigest()
        inputs=json.loads((D/'benchmark'/dataset/'inputs.json').read_text())
        if r['status']!='ok' or len(r['rows'])!=7 or r['n']!=len(inputs['ids']) or r['feature_sha256']!=inputs['feature_sha256']:
            issues.append('runtime source/input coverage '+dataset)
        expected={('datasail',0)}|{(m,s) for m in ('plain','combined') for s in (0,1,2)}
        if {(a['method'],a['seed']) for a in r['rows']}!=expected:issues.append('runtime replication '+dataset)
        native=np.load(root/'datasail.npy');counts=[int(np.sum(native==n)) for n in ('train','val','test')]
        for row in r['rows']:
            labels=native if row['method']=='datasail' else np.load(root/f"{row['method']}__s{row['seed']}.npy")
            if len(labels)!=r['n'] or [int(np.sum(labels==n)) for n in ('train','val','test')]!=counts:
                issues.append('runtime exact counts '+dataset)
            if [row['n_train'],row['n_val'],row['n_test']]!=counts or row['split_s']<=0 or not np.isfinite(row['split_s']):
                issues.append('runtime scoring/clock '+dataset)
            rows.append(dict(dataset=dataset,host=r['host'],threads=r['threads'],method=row['method'],seed=row['seed'],split_s=row['split_s'],
                             global_ratio=row['leak_ratio'],local_ratio=row['test_ramp_default_rel'],train_n=counts[0],val_n=counts[1],test_n=counts[2]))
        log=D/'parallel_queue/logs'/f'runtime__{dataset}.attempt1.log'
        text=log.read_text(errors='replace')
        native_status=re.findall(r'Problem status: ([^\n]+)',text)
        if not native_status:native_status=re.findall(r'SCIP status: ([^\n]+)',text)
        outcomes.append(dict(dataset=dataset,cvxpy_status=native_status[-1] if native_status else 'not captured',
            interpretation='solver-reported clustered optimum' if native_status and native_status[-1]=='optimal' else 'feasible assignment without an optimality certificate',
            log_sha256=hashlib.sha256(log.read_bytes()).hexdigest(),timed_scope=r['timed_scope'],initialization=r['timing_initialization']))
    if rows and (len({r['host'] for r in rows})!=1 or {r['threads'] for r in rows}!={2}):issues.append('runtime hardware/thread mismatch')
    gate=dict(passed=not issues,issues=issues,cases=len(hashes),result_sha256=hashes,
              scope='four same-host cases, two threads, split-call-only clocks, exact native counts; one DataSAIL solve and three low-rank seeds')
    (D/'runtime_validation.json').write_text(json.dumps(gate,indent=2)+'\n')
    if issues:
        if allow_partial and all(issue.startswith('missing runtime case ') for issue in issues):
            print('Runtime pending:',issues);return
        raise RuntimeError('; '.join(issues))
    data=pd.DataFrame(rows);data.to_csv(D/'runtime_rows.csv',index=False)
    summary=data.groupby(['dataset','method']).agg(split_s_median=('split_s','median'),split_s_min=('split_s','min'),split_s_max=('split_s','max'),
        global_mean=('global_ratio','mean'),local_mean=('local_ratio','mean'),local_sd=('local_ratio','std'),replications=('seed','count')).reset_index()
    summary.to_csv(D/'runtime_summary.csv',index=False)
    pd.DataFrame(outcomes).to_csv(D/'runtime_solver_outcomes.csv',index=False)
    times=summary.pivot(index='dataset',columns='method',values='split_s_median');speed=times.datasail/times.combined
    commands=dict(MinSpeedup=f'{speed.min():,.0f}',MaxSpeedup=f'{speed.max():,.0f}',
                  DataSailMinSeconds=f'{times.datasail.min():,.0f}',DataSailMaxSeconds=f'{times.datasail.max():,.0f}',
                  CombinedMinSeconds=f'{times.combined.min():.2f}',CombinedMaxSeconds=f'{times.combined.max():.2f}')
    for model in ('combined','datasail'):
        commands['Bace'+model.capitalize()+'Local']=f"{summary[(summary.dataset=='moleculenet_bace')&(summary.method==model)].local_mean.iloc[0]:.3f}"
    (PAPER/'runtime_numbers.tex').write_text('\n'.join('\\newcommand{\\Runtime'+key+'}{'+value+'}' for key,value in commands.items())+'\n')
    table=summary[['dataset','method','split_s_median','global_mean','local_mean']].copy()
    table.dataset=table.dataset.str.replace('moleculenet_','',regex=False)
    table.columns=['Dataset','Method','Median seconds','Global ratio','Local ratio']
    (PAPER/'runtime_table.tex').write_text(r'\begin{table}[htbp]\centering\small'+'\n'+table.to_latex(escape=True,index=False,float_format='%.3f')+
        r'\caption{Same-host split-call time and exact-count quality. DataSAIL distance construction, clustering and integer programming are inside its clock; imports, feature loading and scoring are outside. Low-rank calls exclude a 100-entity warm-up. DataSAIL has one solve per dataset; low-rank entries aggregate three seeds. The solver time limit is 7,200 seconds and is not an end-to-end wall-time bound. Native solver states and all three low-rank observations accompany the table.}\end{table}'+'\n')
    print(summary.to_string(index=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--allow-partial',action='store_true')
    main(parser.parse_args().allow_partial)
