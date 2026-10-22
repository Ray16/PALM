"""Validate retention/count contracts for published Lo-Hi and UMAP controls."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

PAPER=Path(__file__).resolve().parents[1]
D=PAPER/'revision_20261021'
MOLECULES=['moleculenet_'+x for x in ('freesolv','esol','bace','bbbp','sider','clintox','lipophilicity','tox21','qm8','hiv','muv')]


def main():
    tables=[]
    for study in ('lohi','umap'):
        issues,rows,statuses,hashes=[],[],[],{}
        for dataset in MOLECULES:
            path=D/study/dataset/'result.json'
            if not path.exists():
                issues.append('missing '+dataset);continue
            r=json.loads(path.read_text());root=path.parent
            hashes[dataset]=hashlib.sha256(path.read_bytes()).hexdigest()
            if len(r['ids'])!=r['n'] or len(set(r['ids']))!=r['n']:
                issues.append('identity coverage '+dataset)
            status=dict(dataset=dataset,n=r['n'],status=r['status'],reason=r.get('reason',''))
            if r['status']=='unavailable':
                allowed=(r['n']>5000) if study=='lohi' else (r['n']*(r['n']-1)//2*8>16*1024**3)
                if not allowed:issues.append('unverified policy exclusion '+dataset)
                status['classification']='prespecified resource-policy exclusion'
            elif r['status']=='failed':
                audit=json.loads((D/'umap_failure_audit.json').read_text()) if study=='umap' else {}
                if (study!='umap' or dataset!='moleculenet_freesolv' or
                    audit.get('failed_result_sha256')!=hashes[dataset] or
                    audit.get('feature_matrix_finite') is not True or audit.get('zero_similarity_rows')!=[286] or
                    'Input X contains NaN' not in r.get('reason','')):
                    issues.append('unaudited native failure '+dataset)
                status['classification']='audited native method failure; no scored partition'
            elif r['status']=='ok' and study=='lohi':
                full=np.load(root/'lohi_full_assignment.npy');kept=np.load(root/'retained_indices.npy')
                if len(full)!=r['n'] or not np.array_equal(np.flatnonzero(full!='discard'),kept) or len(kept)!=r['retained_n']:
                    issues.append('retention coverage '+dataset)
                labels=full[kept];counts=[int(np.sum(labels==name)) for name in ('train','test')]
                if min(counts)==0 or len(r['rows'])!=10 or r['maximum_cross_similarity']>.400001:
                    issues.append('Lo-Hi threshold/score coverage '+dataset)
                for m in ('random','plain','combined'):
                    for seed in (0,1,2):
                        lab=np.load(root/f'{m}__s{seed}.npy')
                        if len(lab)!=len(kept) or [int(np.sum(lab==name)) for name in ('train','test')]!=counts:
                            issues.append('matched subset/counts '+dataset+'/'+m)
                for row in r['rows']:
                    if [row['n_train'],row['n_test']]!=counts:issues.append('score counts '+dataset)
                    rows.append(dict(row,dataset=dataset,n=r['n'],retained_n=len(kept),retention=len(kept)/r['n']))
                status.update(classification='scored on exactly matched retained subset',retention=len(kept)/r['n'],
                              retained_n=len(kept),train_n=counts[0],test_n=counts[1],
                              maximum_cross_similarity=r['maximum_cross_similarity'],solver_status=r['solver']['status'])
            elif r['status']=='ok' and study=='umap':
                cluster=np.load(root/'clusters.npy');embedding=np.load(root/'embedding.npy')
                if len(cluster)!=r['n'] or set(cluster)!=set(range(7)) or embedding.shape!=(r['n'],2) or not np.isfinite(embedding).all():
                    issues.append('UMAP input/cluster coverage '+dataset)
                if len(r['rows'])!=21 or {(x['method'],x['fold']) for x in r['rows']}!={(m,f) for m in ('umap_cluster','plain','combined') for f in range(7)}:
                    issues.append('UMAP score coverage '+dataset)
                for fold in range(7):
                    expected=np.where(cluster==fold,'test','train')
                    counts=[int(np.sum(expected==name)) for name in ('train','test')]
                    for m in ('umap_cluster','plain','combined'):
                        labels=np.load(root/f'{m}__fold{fold}.npy')
                        if len(labels)!=r['n'] or [int(np.sum(labels==name)) for name in ('train','test')]!=counts or not set(labels)<={'train','test'}:
                            issues.append('UMAP count/retention match '+dataset+'/'+m)
                        if m=='umap_cluster' and not np.array_equal(labels,expected):issues.append('native cluster identity '+dataset)
                for row in r['rows']:
                    if row['n_train']+row['n_test']!=r['n'] or row['n_val']!=0:issues.append('UMAP scoring retention '+dataset)
                    rows.append(dict(dataset=dataset,**row))
                status['classification']='all seven native holdouts and count-matched controls scored'
            else:issues.append('unknown status '+dataset)
            statuses.append(status)
        frame=pd.DataFrame(rows)
        frame.to_csv(D/f'{study}_rows.csv',index=False)
        pd.DataFrame(statuses).to_csv(D/f'{study}_statuses.csv',index=False)
        local='nnl_rel' if study=='lohi' else 'test_ramp_default_rel'
        summary=frame.groupby(['dataset','method'])[['leak_ratio',local]].mean().rename(columns={local:'local_ratio'}).reset_index()
        summary.to_csv(D/f'{study}_summary.csv',index=False)
        gate=dict(passed=not issues,issues=issues,successful_datasets=sum(s['status']=='ok' for s in statuses),
                  policy_exclusions=sum(s['status']=='unavailable' for s in statuses),
                  native_failures=sum(s['status']=='failed' for s in statuses),result_sha256=hashes,
                  scope='retention and exact-count matching; native failures remain unscored; dataset means precede cross-dataset summaries')
        (D/f'{study}_validation.json').write_text(json.dumps(gate,indent=2)+'\n')
        if issues:raise RuntimeError('; '.join(issues))
        native='lohi' if study=='lohi' else 'umap_cluster'
        columns={'Dataset':sorted(summary.dataset.unique())}
        table=pd.DataFrame(columns).set_index('Dataset')
        for m,name in [('plain','Plain'),('combined','Combined'),(native,'Lo-Hi' if study=='lohi' else 'UMAP')]:
            subset=summary[summary.method==m].set_index('dataset')
            table[name+' G']=subset.leak_ratio
            table[name+' N']=subset.local_ratio
        table=table.reset_index();table.Dataset=table.Dataset.str.replace('moleculenet_','',regex=False)
        caption=('Lo-Hi comparisons use each native retained subset and its exact train/test counts. Low-rank settings average three seeds. '
                 'Retained fractions and native solver states are reported separately.' if study=='lohi' else
                 'UMAP comparisons average all seven native cluster holdouts equally within each dataset. Low-rank controls retain all entities and exactly match each holdout count. '
                 'FreeSolv is an audited native embedding/clustering failure; MUV exceeds the prespecified memory policy. Neither is assigned a score.')
        tables += [r'\begin{table}[htbp]\centering\small',table.to_latex(escape=True,index=False,float_format='%.3f'),
                   r'\caption{'+caption+r' G and N denote global and local ratios, evaluated within each study protocol.}\end{table}']
        print(study,gate['successful_datasets'],'scored',gate['native_failures'],'native failures',gate['policy_exclusions'],'policy exclusions')
    (PAPER/'local_baselines_tables.tex').write_text('\n'.join(tables)+'\n')


if __name__=='__main__':main()
