"""Validate and report the release-code component ablation without fitting models."""
from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
PAPER=Path(__file__).resolve().parents[1];D=PAPER/'revision_20261021';ROOT=D/'ablation_v2'
DATASETS=['moleculenet_'+x for x in ['bace','bbbp','esol','freesolv','lipophilicity','tox21']]
LABELS=dict(plain='Plain',rank32='Rank 32',rank64='Rank 64',rank128='Rank 128',uniform='Uniform landmarks',one_restart='One restart',no_fm='No move/swap refinement',tolerance05='Size tolerance 0.05',power2='Power 2',power3='Power 3',groups='Groups only',tail='Combined',tau05='Combined threshold 0.5',tau06='Combined threshold 0.6',cap25='Group cap 0.25',cap100='Group cap 1.0',cap_skip='Skip cap-blocked merges',tail_ann='Approximate neighbours')

def main():
    rows=[];issues=[];hashes={};assignments={}
    for dataset in DATASETS:
        directory=ROOT/dataset;final=directory/'result.json'
        if not final.exists():issues.append('missing completed component ablation '+dataset);continue
        record=json.loads(final.read_text());hashes[dataset]=hashlib.sha256(final.read_bytes()).hexdigest()
        inputs=json.loads((D/'benchmark'/dataset/'inputs.json').read_text())
        if record['ids']!=inputs['ids'] or record['feature_sha256']!=inputs['feature_sha256']:issues.append('ablation input identity '+dataset)
        expected={(setting,seed) for setting in LABELS for seed in [0,1,2]}
        if {(r['setting'],r['seed']) for r in record['rows']}!=expected or len(record['rows'])!=54:issues.append('ablation coverage '+dataset)
        for r in record['rows']:
            path=directory/f"{r['setting']}__s{r['seed']}.npy";labels=np.load(path)
            digest=hashlib.sha256(path.read_bytes()).hexdigest();assignments[str(path.relative_to(PAPER))]=digest
            if digest!=r['assignment_sha256'] or r['source_files']!=record['source_files']:issues.append('ablation source/assignment identity '+dataset)
            counts=np.array([np.sum(labels==name) for name in ['train','val','test']]);n=len(inputs['ids']);raw=n*np.array([.8,.1,.1]);target=np.floor(raw).astype(int);target[np.argsort(-(raw-target))[:n-int(target.sum())]]+=1
            eps=r['params']['allowed_size_imbalance'];floors=np.minimum(np.ceil(raw*(1-eps)-1e-9).astype(int),target);caps=np.maximum(np.floor(raw*(1+eps)+1e-9).astype(int),target)
            if len(labels)!=n or set(labels)!={'train','val','test'} or not np.all((counts>=floors)&(counts<=caps)):issues.append('ablation count/retention contract '+dataset)
            if not np.array_equal(counts,[r['n_train'],r['n_val'],r['n_test']]):issues.append('ablation score counts '+dataset)
            if (r['global_query_n']!=min(n,4000) or r['test_query_n']!=min(counts[2],2048) or r['test_reference_n']!=counts[0]):issues.append('ablation common scorer '+dataset)
            if any(r.get(key) is None or not np.isfinite(r[key]) for key in ['leak_ratio','test_ramp_default_rel']):issues.append('undefined ablation primary metric '+dataset)
            diag=r['diagnostics'];grouped=r['params'].get('must_link') is not None
            rows.append(dict(dataset=dataset,setting=r['setting'],seed=r['seed'],global_ratio=r['leak_ratio'],local_ratio=r['test_ramp_default_rel'],n_train=counts[0],n_val=counts[1],n_test=counts[2],broken_groups=diag.get('broken_groups',0) if grouped else np.nan,n_groups=diag.get('n_groups',np.nan),split_s=r['split_s']))
    gate=dict(passed=not issues,issues=issues,rows=len(rows),expected_rows=324,result_sha256=hashes,assignment_sha256=assignments,scope='six full datasets, 18 fixed settings, three seeds; release sources, input identities, assignments, two-sided counts and primary benchmark scoring contract')
    (D/'ablation_validation.json').write_text(json.dumps(gate,indent=2)+'\n')
    if issues:raise RuntimeError('; '.join(issues))
    frame=pd.DataFrame(rows);frame.to_csv(D/'ablation_rows.csv',index=False)
    summary=frame.groupby(['dataset','setting']).mean(numeric_only=True).reset_index();summary.to_csv(D/'ablation_summary.csv',index=False)
    tables=[]
    for dataset in DATASETS:
        g=summary[summary.dataset==dataset].set_index('setting').reindex(LABELS)
        table=g[['global_ratio','local_ratio','broken_groups']].rename(columns={'global_ratio':'Global ratio','local_ratio':'Local ratio','broken_groups':'Dissolved groups'})
        table.index=[LABELS[s] for s in table.index]
        tables += [r'\begin{table}[htbp]\centering\small',table.to_latex(escape=True,index=True,float_format='%.3f',na_rep='--'),r'\caption{'+dataset.replace('moleculenet_','')+r': full release-code component ablation, averaged over three split seeds. Standard settings are rank 256, four restarts, move/swap refinement and exact counts. Combined uses power 3, threshold 0.4 and group cap 0.5. Each row changes the indicated setting; cap and neighbour rows modify Combined. Group dissolution is reported only for grouped settings. Actual fold counts and individual observations accompany the table.}\end{table}']
    (PAPER/'ablation_tables.tex').write_text('\n'.join(tables)+'\n')
    print('Validated',len(rows),'release-code component-ablation partitions')
if __name__=='__main__':main()
