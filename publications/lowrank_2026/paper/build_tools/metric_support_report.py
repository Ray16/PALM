"""Recompute every reported metric correlation from the retained observations."""
from pathlib import Path
import json
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PAPER=Path(__file__).resolve().parents[1]
D=PAPER/'revision_20261021/metric_validation'


def main():
    issues=[];raw=pd.read_csv(D/'synthetic_observations.csv');saved=pd.read_csv(D/'synthetic_correlations.csv')
    if not set(raw.method)<={'lowrank','datasail'}:issues.append('unexpected synthetic comparator')
    for regime,frame in [('pooled',raw),*raw.groupby('regime')]:
        common=frame[['family_leak','leak_ratio','nnl_rel']].replace([np.inf,-np.inf],np.nan).dropna()
        for proxy in ('leak_ratio','nnl_rel'):
            expected=saved[(saved.regime==regime)&(saved.proxy==proxy)]
            value=float(spearmanr(common[proxy],common.family_leak).statistic)
            if len(expected)!=1 or expected.n.iloc[0]!=len(common) or not np.isclose(value,expected.rho.iloc[0],atol=1e-12,equal_nan=True):issues.append('synthetic correlation '+regime+'/'+proxy)
    observations=pd.read_csv(D/'per_molecule_observations.csv');correlations=pd.read_csv(D/'per_molecule_correlations.csv')
    for (dataset,reference),frame in observations.groupby(['dataset','reference']):
        for split,group in [('pooled',frame),*frame.groupby('method')]:
            for proxy in ('nn_similarity','nn_ramp','top5_similarity','per_point_global'):
                expected=correlations[(correlations.dataset==dataset)&(correlations.reference==reference)&(correlations.split==split)&(correlations.proxy==proxy)]
                rho=float(spearmanr(group[proxy],group.absolute_error).statistic)
                if len(expected)!=1 or expected.n.iloc[0]!=len(group) or not np.isclose(rho,expected.rho.iloc[0],atol=1e-12,equal_nan=True):issues.append('per-molecule correlation '+dataset+'/'+reference+'/'+split+'/'+proxy)
    gate=dict(passed=not issues,issues=issues,synthetic_rows=len(raw),per_molecule_rows=len(observations),
              observation_sha256={name:hashlib.sha256((D/name).read_bytes()).hexdigest() for name in ('synthetic_observations.csv','per_molecule_observations.csv')},
              scope='all pooled and within-regime/split correlations recomputed from provided observations; original prediction alignment has a separate provenance record')
    (D.parent/'metric_validation.json').write_text(json.dumps(gate,indent=2)+'\n')
    if issues:raise RuntimeError('; '.join(issues))
    synthetic=saved.pivot(index=['regime','n'],columns='proxy',values='rho').reset_index().rename(columns={'regime':'Regime','n':'Observations','leak_ratio':'Global rho','nnl_rel':'Local rho'})
    primary=correlations[(correlations.reference=='train')&(correlations.split!='pooled')&correlations.proxy.isin(['nn_similarity','per_point_global'])]
    per=primary.pivot(index=['dataset','split','n'],columns='proxy',values='rho').reset_index().rename(columns={'dataset':'Dataset','split':'Partition','n':'Test entities','nn_similarity':'NN rho','per_point_global':'Global rho'})
    per.Partition=per.Partition.map({'random':'Random','lowrank':'Plain'})
    per.Dataset=per.Dataset.replace({'esol':'ESOL','freesolv':'FreeSolv','lipophilicity':'Lipophilicity','bbbp':'BBBP'})
    text=[r'\begin{table}[htbp]\centering\small',synthetic.to_latex(escape=True,index=False,float_format='%.3f'),
          r'\caption{Family-leakage correlations on identical finite observations for both metrics. Local similarity is more strongly associated in four of the five constructed regimes; the continuum regime favors the global measure. The pooled contrast also reflects differences among regimes.}\end{table}',
          r'\begin{table}[htbp]\centering\small',per.to_latex(escape=True,index=False,float_format='%.3f'),
          r'\caption{Within-partition correlations with absolute five-initialization D-MPNN ensemble error. NN similarity uses the actual training fold as reference; the global proxy is the per-point overlap score. Each partition has one split realization. Pooled and training-plus-validation analyses accompany these primary within-partition values as CSV files.}\end{table}']
    (PAPER/'metric_validation_tables.tex').write_text('\n'.join(text)+'\n')
    print('Validated all synthetic and per-molecule correlations')


if __name__=='__main__':main()
