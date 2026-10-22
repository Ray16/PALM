"""Validate the full grid and derive tables/numbers without generating prose."""
from pathlib import Path
import hashlib
import json
import argparse
import numpy as np
import pandas as pd
from scipy.stats import binomtest

PAPER=Path(__file__).resolve().parents[1]
D=PAPER/'revision_20261021'
DATASETS=['moleculenet_'+n for n in ('freesolv','esol','bace','bbbp','sider','clintox','lipophilicity','tox21','qm8','hiv','muv')]+['qmof','materials_project']
METHODS=('random','plain','combined','combined_tau06','knn_graph','scaffold','butina','datasail')
LABELS=dict(random='Random',plain='Plain',combined='Combined',combined_tau06='Higher threshold',knn_graph='k-NN graph',scaffold='Scaffold',butina='Butina',datasail='DataSAIL')


def corridor(n,epsilon):
    # Integer bounds from the declared real-valued corridor, widened to include
    # largest-remainder targets. This check is independent of saved diagnostics.
    raw=n*np.array([.8,.1,.1]);target=np.floor(raw).astype(int)
    target[np.argsort(-(raw-target))[:n-int(target.sum())]]+=1
    return np.minimum(np.ceil(raw*(1-epsilon)-1e-9).astype(int),target),np.maximum(np.floor(raw*(1+epsilon)+1e-9).astype(int),target)


def main(allow_partial=False):
    issues,rows,assignments,missing=[],[],{},{}
    allowed_reasons=('FileNotFoundError: no saved DataSAIL assignment',
                     'ValueError: Butina all-pairs baseline limited',
                     'ValueError: SMILES missing for ', 'ValueError: no molecular SMILES for group baseline')
    for dataset in DATASETS:
        root=D/'benchmark'/dataset
        inputs=json.loads((root/'inputs.json').read_text());ids=inputs['ids']
        if len(ids)!=len(set(ids)):issues.append('duplicate input identities '+dataset)
        for method in METHODS:
            for epsilon in (.01,.05):
                for seed in ([0] if method=='datasail' else (0,1,2)):
                    path=root/f'{method}__eps{epsilon}__s{seed}.json'
                    if not path.exists():missing[dataset]=missing.get(dataset,0)+1;continue
                    row=json.loads(path.read_text())
                    if (row['dataset'],row['method'],row['epsilon'],row['seed'])!=(dataset,method,epsilon,seed):issues.append('row identity '+str(path))
                    if row['source_files']!=inputs['source_files'] or row['n']!=len(ids):issues.append('row source/input identity '+str(path))
                    if row['status']=='unavailable':
                        if not row.get('reason','').startswith(allowed_reasons):issues.append('unclassified method failure '+str(path))
                    elif row['status']=='ok':
                        array=path.with_suffix('.npy');lab=np.load(array)
                        assignments[str(array.relative_to(PAPER))]=hashlib.sha256(array.read_bytes()).hexdigest()
                        counts=np.array([np.sum(lab==name) for name in ('train','val','test')])
                        if len(lab)!=len(ids) or set(lab)!={'train','val','test'} or not np.array_equal(counts,[row['n_train'],row['n_val'],row['n_test']]):issues.append('returned assignment coverage '+str(path))
                        floors,caps=corridor(len(ids),epsilon);fits=bool(np.all(counts>=floors)&np.all(counts<=caps))
                        if fits!=row['corridor_ok']:issues.append('saved corridor diagnostic '+str(path))
                        if method in ('random','plain','combined','combined_tau06','knn_graph') and not fits:issues.append('required capacity corridor '+str(path))
                        if (row['global_query_n']!=min(len(ids),4000) or row['test_query_n']!=min(counts[2],2048) or row['test_reference_n']!=counts[0]):issues.append('common scoring scope '+str(path))
                        if any(row.get(k) is None or not np.isfinite(row[k]) or row[k]<-1e-8 for k in ('leak_ratio','test_ramp_default_rel')):issues.append('undefined primary metric '+str(path))
                        if method=='datasail' and row['split_s'] is not None:issues.append('reused timing attributed to DataSAIL')
                        if method=='knn_graph':
                            diag=row['diagnostics'];native=np.asarray(diag['native_labels'])
                            if (diag['graph_type']!='ordinary undirected pairwise graph' or diag['objective']!='CUT' or
                                diag['preset']!='QUALITY' or len(native)!=len(ids) or diag['native_integer_cut']<0 or
                                diag['sum_integer_edge_weights']>diag['integer_weight_budget']):issues.append('graph comparator contract '+str(path))
                    else:issues.append('unresolved status '+str(path))
                    flat={k:v for k,v in row.items() if k not in ('diagnostics','source_files')}
                    for k in ('broken_groups','n_groups','grouped_points','repair_moves','graph_build_s','partition_s','repair_s'):
                        flat[k]=row.get('diagnostics',{}).get(k)
                    rows.append(flat)
    issues += [f'{dataset}: {n} missing grid records' for dataset,n in missing.items()]
    gate=dict(passed=not issues,issues=issues,rows=len(rows),expected_rows=572,missing_by_dataset=missing,
              scope='thirteen full datasets, eight methods, both tolerances; all retained assignments/counts and common scoring checked')
    (D/'benchmark_validation.json').write_text(json.dumps(gate,indent=2)+'\n')
    if issues:
        if allow_partial and missing and len(issues)==len(missing):
            print('Benchmark pending:',missing);return
        raise RuntimeError('; '.join(issues))
    frame=pd.DataFrame(rows);frame.to_csv(D/'benchmark_rows.csv',index=False)
    (D/'benchmark_assignments_sha256.json').write_text(json.dumps(assignments,indent=2)+'\n')
    good=frame[frame.status=='ok']
    columns=['leak_ratio','test_ramp_default_rel','split_s','n_train','n_val','n_test','broken_groups','repair_moves']
    summary=good.groupby(['dataset','method','epsilon'])[columns].mean().reset_index()
    summary.to_csv(D/'benchmark_summary.csv',index=False)
    stats=[]
    matched=pd.read_csv(D/'datasail_matched_summary_v2.csv').rename(columns={'global_ratio':'leak_ratio','local_ratio':'test_ramp_default_rel'})
    for scope in ('molecular','all_datasets'):
        for epsilon in (.01,.05):
            for first,second in [('plain','knn_graph'),('combined','plain'),('combined','knn_graph'),('plain','datasail'),('combined','datasail')]:
                source=matched if second=='datasail' else summary
                source=source[source.epsilon==epsilon]
                if scope=='molecular':source=source[source.dataset.str.startswith('moleculenet_')]
                for metric in ('leak_ratio','test_ramp_default_rel'):
                    pivot=source.pivot(index='dataset',columns='method',values=metric)
                    a=pivot[[first,second]].dropna();delta=(a[first]-a[second]).to_numpy()
                    win=int(np.sum(delta<-1e-10));loss=int(np.sum(delta>1e-10));tie=len(delta)-win-loss
                    boot=np.random.default_rng(31415).choice(delta,(10000,len(delta))).mean(1)
                    lo,hi=np.quantile(boot,[.025,.975])
                    stats.append(dict(scope=scope,epsilon=epsilon,first=first,second=second,metric=metric,n=len(delta),
                        wins=win,ties=tie,losses=loss,mean_change=delta.mean(),ci_low=lo,ci_high=hi,
                        sign_p=binomtest(win,win+loss,.5).pvalue if win+loss else 1.,
                        count_contract='exact DataSAIL counts' if second=='datasail' else 'common two-sided corridor; actual counts retained'))
    s=pd.DataFrame(stats)
    s['holm_p']=np.nan
    for scope,g in s.groupby('scope'):
        order=g.sort_values('sign_p').index;raw=s.loc[order,'sign_p'].to_numpy()
        s.loc[order,'holm_p']=np.minimum(1,np.maximum.accumulate(raw*(len(raw)-np.arange(len(raw)))))
    s.to_csv(D/'benchmark_paired_statistics.csv',index=False)
    # Full per-dataset tables belong in SI so the main text can carry the argument.
    tables=[]
    medians=good.groupby(['dataset','method','epsilon']).split_s.median()
    for epsilon in (.01,.05):
        tables += [r'\subsection*{Nominal size tolerance '+str(epsilon)+'}']
        for dataset in DATASETS:
            g=summary[(summary.dataset==dataset)&(summary.epsilon==epsilon)].copy().set_index('method').reindex(METHODS)
            g['split_s']=[medians.get((dataset,method,epsilon),np.nan) for method in METHODS]
            table=g[['leak_ratio','test_ramp_default_rel','split_s']].rename(columns={'leak_ratio':'Global ratio','test_ramp_default_rel':'Local ratio','split_s':'Split seconds'})
            table.index=[LABELS[n] for n in table.index]
            name=dataset.replace('moleculenet_','').replace('_',' ')
            tables += [r'\begin{table}[htbp]\centering\small',table.to_latex(escape=True,index=True,float_format='%.3f',na_rep='--'),
                r'\caption{'+name+' at tolerance '+str(epsilon)+r'. Metrics average split seeds; time is the median split-call time in the accompanying row records. DataSAIL assignments are reused and have no new time. Dashes mean unavailable under the stated protocol. Actual counts, group relaxations, raw timing observations and all exclusion reasons are retained in the CSV files.}\end{table}']
    primary=s[(s.scope=='molecular')&(s.epsilon==.01)].copy()
    primary['Comparison']=[LABELS[a]+' / '+LABELS[b] for a,b in zip(primary['first'],primary['second'])]
    primary['Metric']=primary.metric.map({'leak_ratio':'Global','test_ramp_default_rel':'Local'})
    primary['95% interval']=[f'[{lo:.2f}, {hi:.2f}]' for lo,hi in zip(primary.ci_low,primary.ci_high)]
    stats_table=primary[['Comparison','Metric','n','mean_change','95% interval','holm_p']].rename(columns={'mean_change':'Mean change','holm_p':'Holm p'})
    tables += [r'\begin{table}[htbp]\centering\small',stats_table.to_latex(escape=True,index=False,float_format='%.3f'),
               r'\caption{Paired molecular-dataset effects at tolerance 0.01. Changes subtract the second method from the first. Dataset-bootstrap intervals use 10,000 resamples after averaging split seeds. Sign-test p values receive Holm correction across the 20 declared comparisons (five method pairs, two metrics and two tolerances). DataSAIL pairs use exact-count controls. These convenience datasets support descriptive, representation-specific inference.}\end{table}']
    (PAPER/'benchmark_tables.tex').write_text('\n'.join(tables)+'\n')
    commands={}
    for first,second,metric,key in [('plain','knn_graph','leak_ratio','PlainGraphGlobal'),('combined','plain','test_ramp_default_rel','CombinedPlainLocal'),('combined','knn_graph','test_ramp_default_rel','CombinedGraphLocal'),('plain','datasail','leak_ratio','PlainDataSailGlobal'),('combined','datasail','leak_ratio','CombinedDataSailGlobal'),('combined','datasail','test_ramp_default_rel','CombinedDataSailLocal')]:
        cell=s[(s.scope=='molecular')&(s.epsilon==.01)&(s['first']==first)&(s['second']==second)&(s.metric==metric)].iloc[0]
        commands[key+'Wins']=str(int(cell.wins));commands[key+'Ties']=str(int(cell.ties));commands[key+'N']=str(int(cell.n))
        commands[key+'Change']=f'{cell.mean_change:+.3f}'
    (PAPER/'benchmark_numbers.tex').write_text('\n'.join('\\newcommand{\\Bench'+key+'}{'+value+'}' for key,value in commands.items())+'\n')
    print('Validated',len(rows),'grid records;',len(assignments),'scored assignments')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--allow-partial',action='store_true')
    main(parser.parse_args().allow_partial)
