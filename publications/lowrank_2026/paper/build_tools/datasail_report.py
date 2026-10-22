"""Native DataSAIL outcome classification and exact-count quality controls."""
from pathlib import Path
import hashlib
import json
import re
import numpy as np
import pandas as pd

PAPER=Path(__file__).resolve().parents[1]
D=PAPER/'revision_20261021'
SOURCE=PAPER/'historical/datasail_source'


def main():
    status=[]
    for entry in json.loads((SOURCE/'index.json').read_text()):
        files=entry['files']
        for item in files.values():
            if hashlib.sha256((SOURCE/item['path']).read_bytes()).hexdigest()!=item['sha256']:
                raise RuntimeError('native DataSAIL archive digest differs')
        raw=pd.read_csv(SOURCE/files['csv']['path']).iloc[0].to_dict() if 'csv' in files else {}
        log=(SOURCE/files['log']['path']).read_text(errors='replace') if 'log' in files else ''
        native=re.findall(r'Problem status: ([^\n]+)',log)
        solver=native[-1] if native else None
        if 'assignment' in files:
            classification='solver-reported optimum of clustered problem' if solver=='optimal' else 'feasible assignment; no optimality certificate'
        elif raw.get('status')=='skipped_oom':classification='resource-policy exclusion'
        elif 'Splitting failed' in log or 'No assignments could be made' in log:classification='no feasible assignment returned within configured solve'
        else:classification='no recorded attempt under this protocol'
        status.append(dict(dataset=entry['dataset'],epsilon=entry['epsilon'],classification=classification,
                           native_cvxpy_status=solver,legacy_csv_status=raw.get('status'),has_saved_assignment='assignment' in files,
                           old_seconds_for_provenance_only=raw.get('secs')))
    states=pd.DataFrame(status);states.to_csv(D/'datasail_source_statuses.csv',index=False)
    issues,rows,hashes=[],[],{}
    datasets=sorted(states[states.has_saved_assignment].dataset.unique())
    for dataset in datasets:
        root=D/'datasail_matched_v2'/dataset;path=root/'result.json'
        if not path.exists():issues.append('missing exact-count case '+dataset);continue
        record=json.loads(path.read_text());hashes[dataset]=hashlib.sha256(path.read_bytes()).hexdigest()
        ids=json.loads((root/'ids.json').read_text())
        if len(ids)!=record['n'] or len(set(ids))!=record['n'] or len(record['rows'])!=14:
            issues.append('matched coverage '+dataset)
        expected={(m,e,s) for e in (.01,.05) for m in ('datasail','plain','combined') for s in ([0] if m=='datasail' else (0,1,2))}
        if {(r['method'],r['epsilon'],r['seed']) for r in record['rows']}!=expected:issues.append('matched grid '+dataset)
        for row in record['rows']:
            eps=row['epsilon'];method=row['method'];seed=row['seed']
            native=np.load(root/f'datasail__eps{eps}__s0.npy')
            source_entry=next(e for e in json.loads((SOURCE/'index.json').read_text()) if e['dataset']==dataset and e['epsilon']==eps)
            original_ids=np.load(SOURCE/source_entry['files']['ids']['path'],allow_pickle=True).tolist()
            original_labels=np.load(SOURCE/source_entry['files']['assignment']['path'])
            mapping=dict(zip(original_ids,original_labels))
            if set(original_ids)!=set(ids) or not np.array_equal(native,np.array([mapping[i] for i in ids])):
                issues.append('saved native assignment identity '+dataset)
            labels=np.load(root/f'{method}__eps{eps}__s{seed}.npy')
            counts=[int(np.sum(native==name)) for name in ('train','val','test')]
            if (len(labels)!=len(ids) or set(labels)!={'train','val','test'} or
                [int(np.sum(labels==name)) for name in ('train','val','test')]!=counts or
                [row['n_train'],row['n_val'],row['n_test']]!=counts):issues.append('matched counts '+dataset)
            rows.append(dict(dataset=dataset,method=method,epsilon=eps,seed=seed,n=record['n'],
                             global_ratio=row['leak_ratio'],local_ratio=row['test_ramp_default_rel'],
                             train_n=counts[0],val_n=counts[1],test_n=counts[2]))
    gate=dict(passed=not issues,issues=issues,datasets=len(hashes),result_sha256=hashes,
              scope='seven saved DataSAIL assignment sources; both tolerances; exact all-input count matching and common scoring')
    (D/'datasail_matched_validation.json').write_text(json.dumps(gate,indent=2)+'\n')
    if issues:raise RuntimeError('; '.join(issues))
    d=pd.DataFrame(rows);d.to_csv(D/'datasail_matched_rows.csv',index=False)
    summary=d.groupby(['dataset','method','epsilon'])[['global_ratio','local_ratio']].mean().reset_index()
    summary.to_csv(D/'datasail_matched_summary_v2.csv',index=False)
    primary=summary[summary.epsilon==.01].pivot(index='dataset',columns='method',values=['global_ratio','local_ratio'])
    table=pd.DataFrame(index=primary.index)
    for method,label in [('plain','Plain'),('combined','Combined'),('datasail','DataSAIL')]:
        table[label+' G']=primary['global_ratio'][method];table[label+' N']=primary['local_ratio'][method]
    table=table.reset_index().rename(columns={'dataset':'Dataset'});table.Dataset=table.Dataset.str.replace('moleculenet_','',regex=False)
    text=[r'\begin{table}[htbp]\centering\small',table.to_latex(escape=True,index=False,float_format='%.3f'),
          r'\caption{Exact-count controls for saved DataSAIL assignments at nominal tolerance 0.01. Each low-rank partition retains the same entities and has exactly the three native DataSAIL counts. Low-rank scores average three seeds; DataSAIL has one assignment per configuration. G and N denote global and test-only local ratios. These reused assignments supply no new runtime.}\end{table}']
    counts=states.groupby(['epsilon','classification']).size().unstack(0,fill_value=0).reset_index()
    counts.columns=['Recorded outcome','Tolerance 0.01','Tolerance 0.05']
    text += [r'\begin{table}[htbp]\centering\small',counts.to_latex(escape=True,index=False,column_format='p{4in}rr'),
             r"\caption{DataSAIL source coverage, classified from native logs and saved assignments. Solver-reported optima concern its clustered integer program. The legacy CSV label ``infeasible'' is not treated as proof of infeasibility when no such solver certificate was retained. Resource exclusions and missing recorded attempts are separate.}\end{table}"]
    (PAPER/'datasail_tables.tex').write_text('\n'.join(text)+'\n')
    print(states.groupby('classification').size().to_string())


if __name__=='__main__':main()
