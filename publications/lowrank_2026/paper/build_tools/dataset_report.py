"""Render the fixed dataset catalogue and the separate neighbour engineering table."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

PAPER=Path(__file__).resolve().parents[1]


def main():
    records=json.loads((PAPER/'dataset_statistics.json').read_text())
    table=pd.DataFrame([{'Dataset':r['dataset'].replace('moleculenet_','').replace('_',' '),
        'Entities':r['retained_rows'],'Labelled':r['labelled_n'],'Target':r['target'],'Positive':r['positive_n']} for r in records])
    (PAPER/'dataset_table.tex').write_text(r'\begin{table}[htbp]\centering\small'+'\n'+table.to_latex(escape=True,index=False,column_format='lrrp{2.5in}r',na_rep='--',float_format='%.0f')+
        r'\caption{Retained entities, declared target slices and label coverage. Positive counts apply only to binary tasks; dashes denote continuous targets. Structural splitting retains valid features independently of missing responses. Single-target slices do not constitute an evaluation of all assays in a multitask source. Exact raw-file and retained-identity hashes accompany the catalogue.}\end{table}'+'\n')
    rows=[];issues=[]
    for path in sorted((PAPER/'results/neighbours').glob('*__n*.json')):
        r=json.loads(path.read_text())
        if r['n'] not in (1000,10000,30000,100000) or r['backend'] not in ('gpu_tiled_exact','nndescent') or r['recall_query_n']!=256:
            issues.append('neighbour study coverage '+path.name)
        if r['graph_s']<=0 or not 0<=r['recall_at32']<=1 or not 0<=r['strong_edge_recall_at32']<=1:
            issues.append('neighbour study values '+path.name)
        rows.append(dict(n=r['n'],Backend='Exact tiled' if r['backend']=='gpu_tiled_exact' else 'NN-Descent',Seconds=r['graph_s'],Recall=r['recall_at32'],StrongRecall=r['strong_edge_recall_at32']))
    if len(rows)!=8:issues.append('require eight neighbour construction observations')
    if issues:raise RuntimeError('; '.join(issues))
    table=pd.DataFrame(rows);table.columns=['Entities','Backend','Graph seconds','Top-32 recall','Strong recall']
    (PAPER/'neighbour_table.tex').write_text(r'\begin{table}[htbp]\centering\small'+'\n'+table.to_latex(escape=True,index=False,float_format='%.3f')+
        r'\caption{Neighbour-graph engineering measurements on ChEMBL fingerprints. Exact tiled search uses a Tesla T4; NN-Descent uses two CPU threads. Recall is measured on 256 sampled query entities; strong recall restricts exact top-32 edges to similarity at least 0.4. ANN compilation warm-up is separate. These are graph-construction observations from the preserved source study, not H100 split-call timings or measurements of the revised partition optimizer.}\end{table}'+'\n')


if __name__=='__main__':main()
