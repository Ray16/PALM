"""Validate and summarize random-relative calibration; reconcile legacy gaps."""
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd

PAPER = Path(__file__).resolve().parents[1]
D = PAPER / 'revision_20261021'
DATASETS = ['moleculenet_' + n for n in ('freesolv','esol','bace','bbbp','lipophilicity','sider')]


def main():
    issues, rows, violations, errors, fingerprints = [], [], [], [], {}
    for dataset in DATASETS:
        path = D/'calibration_v2'/dataset/'result.json'
        if not path.exists():
            issues.append('missing calibration: '+dataset)
            continue
        record = json.loads(path.read_text())
        fingerprints[str(path.relative_to(PAPER))] = hashlib.sha256(path.read_bytes()).hexdigest()
        root = path.parent
        ids = json.loads((root/'ids.json').read_text())
        expected = {('calibration_knot',s,'rf',i) for s in (0,1,2) for i in range(5)}
        expected |= {('validation_knot',s,m,i) for s in range(10,15) for m in ('rf','xgb') for i in range(5)}
        expected |= {('target_inversion',s,m,i) for s in range(10,15) for m in ('rf','xgb') for i in range(3)}
        seen = {(r['stage'],r['seed'],r['model'],r['index']) for r in record['rows']}
        if record['status']!='ok' or len(record['rows'])!=95 or seen!=expected:
            issues.append('model coverage: '+dataset)
        if len(ids)!=record['n'] or len(set(ids))!=len(ids):issues.append('identities: '+dataset)
        if len(record['assignment_sha256'])!=63:issues.append('assignment coverage: '+dataset)
        for name,digest in record['assignment_sha256'].items():
            file=root/name
            if not file.exists() or hashlib.sha256(file.read_bytes()).hexdigest()!=digest:
                issues.append('assignment digest: '+dataset+'/'+name)
                continue
            lab=np.load(file)
            if len(lab)!=len(ids) or set(lab)!={'train','val','test'}:
                issues.append('assignment retention: '+dataset+'/'+name)
            seed=int(name.split('__')[0][1:])
            optimized=np.load(root/f's{seed}__optimized.npy')
            if any(np.sum(lab==fold)!=np.sum(optimized==fold) for fold in ('train','val','test')):
                issues.append('assignment counts: '+dataset+'/'+name)
        references={(r['seed'],r['model']):r for r in record['rows']
                    if r['stage'].endswith('_knot') and r['index']==0}
        task=record['task'];key='test_auc' if task=='classification' else 'test_rmse'
        for native in record['rows']:
            row=dict(native,dataset=dataset,units='AUC decrement' if task=='classification' else 'relative RMSE')
            ref=references[row['seed'],row['model']].get(key)
            value=row.get(key)
            recomputed=(ref-value if task=='classification' else value/ref-1) if value is not None and ref is not None else None
            if recomputed is None or row['difficulty'] is None or not np.isclose(recomputed,row['difficulty'],atol=1e-12):
                issues.append('difficulty definition or undefined score: '+dataset)
            if row['stage']=='validation_knot':
                prediction=float(np.interp(row['alpha'],record['alphas'],record['isotonic_mean_difficulties']))
                errors.append(dict(dataset=dataset,model=row['model'],seed=row['seed'],stage='fixed_knots',
                                   units=row['units'],target=prediction,realized=row['difficulty'],abs_error=abs(prediction-row['difficulty'])))
            if row['stage']=='target_inversion':
                target=row['target_difficulty']
                errors.append(dict(dataset=dataset,model=row['model'],seed=row['seed'],stage='interior_targets',
                                   units=row['units'],target=target,realized=row['difficulty'],abs_error=abs(target-row['difficulty'])))
            rows.append(row)
        knots=pd.DataFrame(record['rows'])
        for (stage,model,seed),g in knots[knots.stage!='target_inversion'].groupby(['stage','model','seed']):
            differences=np.diff(g.sort_values('alpha').difficulty.to_numpy())
            violations.append(dict(dataset=dataset,stage=stage,model=model,seed=int(seed),
                                   violations=int(np.sum(differences < -1e-12)),adjacent_pairs=len(differences)))
    gate=dict(passed=not issues,issues=issues,complete_datasets=len(fingerprints),result_sha256=fingerprints,
              scope='six datasets, exact saved counts, 570 fixed-model-seed fits; fixed knots and target inversion are distinct')
    (D/'calibration_validation.json').write_text(json.dumps(gate,indent=2)+'\n')
    if issues:
        raise RuntimeError('; '.join(issues))
    data=pd.DataFrame(rows);data.to_csv(D/'calibration_v2/rows.csv',index=False)
    v=pd.DataFrame(violations)
    v.to_csv(D/'calibration_v2/monotonicity_per_seed.csv',index=False)
    v.groupby(['dataset','stage','model'])[['violations','adjacent_pairs']].sum().to_csv(D/'calibration_v2/monotonicity_summary.csv')
    e=pd.DataFrame(errors);e.to_csv(D/'calibration_v2/validation_errors.csv',index=False)
    summary=[]
    for (dataset,model,stage),g in e.groupby(['dataset','model','stage']):
        seedmeans=g.groupby('seed').abs_error.mean().to_numpy()
        bootstrap=np.random.default_rng(31415).choice(seedmeans,(10000,len(seedmeans))).mean(1)
        lo,hi=np.quantile(bootstrap,[.025,.975])
        summary.append(dict(dataset=dataset,model=model,stage=stage,units=g.units.iloc[0],
                            mae=g.abs_error.mean(),ci_low=lo,ci_high=hi,n=len(g),split_seeds=len(seedmeans)))
    s=pd.DataFrame(summary)
    s.to_csv(D/'calibration_v2/validation_summary.csv',index=False)
    tables=[]
    for stage,caption in [('fixed_knots','Fixed-knot validation on five unseen split seeds (25 points per model and dataset).'),
                           ('interior_targets','Interior-target inversion on the same five validation seeds (15 points per model and dataset). Targets are fitted using RF calibration seeds 0--2.')]:
        g=s[s.stage==stage].pivot(index=['dataset','units'],columns='model',values='mae').reset_index()
        g['dataset']=g.dataset.str.replace('moleculenet_','',regex=False)
        tables += [r'\begin{table}[htbp]\centering\small',
                   g.rename(columns={'dataset':'Dataset','units':'Units','rf':'RF MAE','xgb':'XGB MAE'}).to_latex(escape=True,index=False,float_format='%.3f'),
                   r'\caption{'+caption+' Classification uses AUC(random) minus AUC(split); regression uses RMSE(split)/RMSE(random) minus one. No test label is used to fit the calibration on validation seeds. Each model has initialization seed zero.}'+r'\end{table}']
    mono=v[v.stage=='validation_knot'].groupby(['dataset','model'])[['violations','adjacent_pairs']].sum().reset_index()
    mono['dataset']=mono.dataset.str.replace('moleculenet_','',regex=False)
    tables += [r'\begin{table}[htbp]\centering\small',mono.rename(columns={'dataset':'Dataset','model':'Model','violations':'Decreases','adjacent_pairs':'Adjacent steps'}).to_latex(escape=True,index=False),
               r'\caption{Observed decreases in random-relative difficulty as hardness increases on validation seeds. Four adjacent knot steps are checked for each of five seeds. Isotonic fitting does not remove these violations from the observed scores.}\end{table}']
    (PAPER/'calibration_tables.tex').write_text('\n'.join(tables)+'\n')
    # The two legacy files evaluate different points, rather than the same estimator.
    legacy=[]
    for dataset in DATASETS:
        for source_name,stage,csv_name in [('calibration','validation','calibration_validation.csv'),('inversion','inversion','inversion_validation.csv')]:
            path=PAPER/'results'/f'{source_name}__{dataset}.json'
            native=json.loads(path.read_text())
            selected=native['validation'] if source_name=='calibration' else native['rows']
            saved=pd.read_csv(PAPER/'derived'/csv_name)
            for model in ('rf','xgb'):
                raw=[abs((r['gap'] if source_name=='calibration' else r['realized_gap'])-r['predicted_gap']) for r in selected if r['model']==model]
                cell=saved[(saved.dataset==dataset)&(saved.model==model)].iloc[0]
                if len(raw)!=int(cell['n']) or not np.isclose(np.mean(raw),cell.mae,atol=1e-12):
                    raise RuntimeError('legacy source reconciliation failed: '+dataset+'/'+source_name+'/'+model)
                legacy.append(dict(dataset=dataset,model=model,source=csv_name,raw_source=str(path.relative_to(PAPER)),
                                   mae=float(np.mean(raw)),n=len(raw),study=stage))
    (D/'legacy_calibration_reconciliation.json').write_text(json.dumps(dict(passed=True,rows=legacy,
        conclusion='The old files describe distinct designs: 25 fixed-knot scores versus 15 interior targets. Both use train-minus-test gaps, not random-relative difficulty. The new authoritative estimator is calibration_v2/validation_summary.csv.'),indent=2)+'\n')
    print(s.to_string(index=False))


if __name__=='__main__':main()
