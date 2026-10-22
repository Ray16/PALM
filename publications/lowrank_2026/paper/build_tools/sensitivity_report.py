"""Summarize graph degree, native repair effects and longer-search controls."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

PAPER=Path(__file__).resolve().parents[1]
D=PAPER/'revision_20261021'
DATASETS=['moleculenet_'+x for x in ('freesolv','esol','bace','bbbp','clintox','lipophilicity')]


def main():
    issues,rows,search,hashes=[],[],[],{}
    for dataset in DATASETS:
        root=D/'sensitivity_v2'/dataset;path=root/'result.json'
        if not path.exists():issues.append('missing '+dataset);continue
        r=json.loads(path.read_text());hashes[dataset]=hashlib.sha256(path.read_bytes()).hexdigest()
        if r['status']!='ok' or len(r['graph_rows'])!=9 or {(a['k'],a['seed']) for a in r['graph_rows']}!={(k,s) for k in (16,64,128) for s in (0,1,2)}:
            issues.append('graph coverage '+dataset)
        for row in r['graph_rows']:
            labels=np.load(root/f"graph__k{row['k']}__s{row['seed']}.npy")
            native=np.load(root/f"graph__k{row['k']}__s{row['seed']}__native.npy")
            if len(labels)!=r['n'] or len(native)!=r['n'] or np.any(np.array(row['diagnostics']['repaired_counts'])!=[np.sum(labels==n) for n in ('train','val','test')]):
                issues.append('graph assignment counts '+dataset)
            for stage in ('native','repaired'):
                rows.append(dict(dataset=dataset,k=row['k'],seed=row['seed'],stage=stage,
                    leak_ratio=row[stage]['leak_ratio'],local_ratio=row[stage]['test_ramp_default_rel'],
                    repair_moves=row['diagnostics']['repair_moves'],split_s=row['split_s'],
                    retained_mass=row['quantized_retained_mass'],mass_error_bound=row['retained_mass_error_bound']))
        # Add the prespecified main k=32 results without pretending native scores were saved.
        for seed in (0,1,2):
            file=D/'benchmark'/dataset/f'knn_graph__eps0.01__s{seed}.json'
            mainrow=json.loads(file.read_text());diag=mainrow['diagnostics']
            if mainrow['status']!='ok':issues.append('main k32 unavailable '+dataset)
            rows.append(dict(dataset=dataset,k=32,seed=seed,stage='repaired',leak_ratio=mainrow['leak_ratio'],
                local_ratio=mainrow['test_ramp_default_rel'],repair_moves=diag['repair_moves'],split_s=mainrow['split_s'],
                retained_mass=diag['sum_integer_edge_weights']/diag['weight_scale']/r['ordinary_off_diagonal_mass'],
                mass_error_bound=diag['graph_edges']*diag['max_weight_quantization_error']/r['ordinary_off_diagonal_mass']))
        if dataset=='moleculenet_freesolv':
            if len(r['search_rows'])!=20:issues.append('search coverage')
            for row in r['search_rows']:
                path=root/f"{row['method']}__s{row['seed']}__restarts{row['restarts']}.npy"
                labels=np.load(path)
                if len(labels)!=r['n'] or [np.sum(labels==n) for n in ('train','val','test')]!=[514,64,64]:
                    issues.append('search exact counts')
                search.append(dict(dataset=dataset,**{k:row[k] for k in ('method','seed','restarts','exact_cut','exact_powered_cut','split_s')},local_ratio=row['test_ramp_default_rel'],factor_cut=row['diagnostics']['factor_leakage']))
    gate=dict(passed=not issues,issues=issues,complete_datasets=len(hashes),result_sha256=hashes,
              scope='graph k sensitivity on six full datasets; five-seed FreeSolv search control; no optimality certificate')
    (D/'sensitivity_validation.json').write_text(json.dumps(gate,indent=2)+'\n')
    if issues:raise RuntimeError('; '.join(issues))
    a=pd.DataFrame(rows);a.to_csv(D/'graph_sensitivity_rows.csv',index=False)
    b=a[a.stage=='repaired'].groupby(['dataset','k'])[['leak_ratio','local_ratio','retained_mass','mass_error_bound','split_s','repair_moves']].mean().reset_index()
    b.to_csv(D/'graph_sensitivity_datasets.csv',index=False)
    summary=b.groupby('k')[['leak_ratio','local_ratio','retained_mass','mass_error_bound','split_s','repair_moves']].mean().reset_index()
    summary.to_csv(D/'graph_sensitivity_summary.csv',index=False)
    (PAPER/'sensitivity_numbers.tex').write_text(
        r'\newcommand{\GraphMassDefault}{'+f"{100*summary.set_index('k').loc[32,'retained_mass']:.1f}"+'}\n'+
        r'\newcommand{\GraphMassHigh}{'+f"{100*summary.set_index('k').loc[128,'retained_mass']:.1f}"+'}\n')
    native=a.pivot_table(index=['dataset','k','seed'],columns='stage',values=['leak_ratio','local_ratio']).dropna()
    effects=pd.DataFrame({metric+'_change':native[metric]['repaired']-native[metric]['native'] for metric in ('leak_ratio','local_ratio')}).reset_index()
    effects.to_csv(D/'graph_repair_effects.csv',index=False)
    q=pd.DataFrame(search);q.to_csv(D/'long_search_rows.csv',index=False)
    t=q.pivot(index=['method','seed'],columns='restarts',values=['exact_cut','exact_powered_cut','local_ratio','factor_cut','split_s'])
    changes=pd.DataFrame({metric+'_relative_change':t[metric][30]/t[metric][4]-1 for metric in ('exact_cut','exact_powered_cut','factor_cut','split_s')})
    changes['local_change']=t['local_ratio'][30]-t['local_ratio'][4]
    changes.to_csv(D/'long_search_changes.csv')
    main_table=summary[['k','retained_mass','leak_ratio','local_ratio']].rename(columns={'k':'Neighbours','retained_mass':'Retained mass','leak_ratio':'Global ratio','local_ratio':'Local ratio'})
    search_table=changes.groupby('method')[['exact_cut_relative_change','exact_powered_cut_relative_change','local_change']].mean().reset_index()
    search_table.columns=['Setting','Dense cut change','Powered cut change','Local ratio change']
    text=[r'\begin{table}[htbp]\centering\small',main_table.to_latex(escape=True,index=False,float_format='%.4f'),
          r'\caption{Graph degree sensitivity on six full datasets. Dataset means average three seeds before cross-dataset averaging. Retained mass uses quantized graph weights and the full ordinary-kernel mass, with quantization-error bounds supplied in the CSV. Global and local scores follow the main common protocol. Build, partition and capacity repair are included in split time.}\end{table}',
          r'\begin{table}[htbp]\centering\small',search_table.to_latex(escape=True,index=False,float_format='%.4f'),
          r'\caption{Thirty versus four restarts on full FreeSolv at exact counts, averaged over five fixed seeds. Cut changes are relative to the four-restart value; local changes are absolute. Both use the same factor seed within each pair. More restarts do not certify optimality, and choosing a different pre-refinement basin can change the final polished solution.}\end{table}']
    (PAPER/'sensitivity_tables.tex').write_text('\n'.join(text)+'\n')
    print(summary.to_string(index=False));print(search_table.to_string(index=False))


if __name__=='__main__':main()
