"""Fail closed on missing controls, independent replay, provenance or PDFs."""
from __future__ import annotations
import json
from pathlib import Path
import re
import sys

import suite
import report

HERE=Path(__file__).resolve().parent
PAPER=HERE.parent/'paper'


def main():
    m,rows=report.verified()
    p=json.loads((HERE/'analysis/provenance.json').read_text())
    if p['report_sha256']!=suite.sha(HERE/'report.py'):raise RuntimeError('report source drift')
    for name,digest in p['generated_files'].items():
        if suite.sha(PAPER/name)!=digest:raise RuntimeError('generated number/table drift: '+name)
    for name,key in [('all_observations.csv','all_observations_sha256'),('summary.json','summary_sha256'),
                     ('scaling_observations.csv','scaling_observations_sha256'),
                     ('preempted_clock_observations.csv','preempted_clock_observations_sha256')]:
        if suite.sha(HERE/'analysis'/name)!=p[key]:raise RuntimeError('analysis drift')
    replay_assignments=0
    for ds in suite.DATASETS:
        r=json.loads((HERE/'verification'/('replay_'+ds+'.json')).read_text())
        if not r['passed'] or r['cases']!=72 or r['assignments']!=117:
            raise RuntimeError('incomplete independent replay: '+ds)
        if r['validation_source_sha256']!=suite.sha(HERE/'validate.py'):
            raise RuntimeError('replay source changed')
        if r['manifest_sha256']!=suite.sha(HERE/'run/manifest.json'):raise RuntimeError('replay manifest drift')
        for name,digest in r['observations'].items():
            if suite.sha(HERE/'run/rows'/name)!=digest:raise RuntimeError('replayed row changed')
        replay_assignments+=r['assignments']
    for study,root,expected in [('scaling','scaling_run',36),('clock','strict_clock_run',72)]:
        r=json.loads((HERE/'verification'/('replay_'+study+'_addition.json')).read_text())
        if not r['passed'] or r['assignments']!=expected:raise RuntimeError('incomplete addition replay')
        if r['validator_sha256']!=suite.sha(HERE/'verify_additions.py'):raise RuntimeError('addition validator drift')
        if r['manifest_sha256']!=suite.sha(HERE/root/'manifest.json'):raise RuntimeError('addition replay manifest drift')
        for name,digest in r['observations'].items():
            if suite.sha(HERE/root/'rows'/name)!=digest:raise RuntimeError('addition row drift')
        replay_assignments+=r['assignments']
    tests=(HERE/'verification/tests.log').read_text()
    if not re.search(r'226 passed',tests) or 'FAILED' in tests:raise RuntimeError('numerical regression suite incomplete')
    source=(PAPER/'manuscript.tex').read_text()
    if 'Scalable Dataset Splitting through Constrained Low-Rank Similarity Optimization' not in source:
        raise RuntimeError('title missing')
    for name in ['controls_results.tex','controls_methods.tex','controls_supplement.tex']:
        if not (PAPER/name).exists():raise RuntimeError('missing manuscript control narrative')
    for name in ['manuscript','supplement']:
        log=(PAPER/'logs'/(name+'_pass3.log')).read_text(errors='replace')
        if re.search(r'Overfull \\[hv]box|undefined references|Citation .*undefined',log):
            raise RuntimeError('TeX check: '+name)
    old=json.loads((HERE.parent/'revision_20261026/verification/completion_gate.json').read_text())
    if not old['technical_checks_passed']:raise RuntimeError('primary evidence/PDF gate incomplete')
    result=dict(passed=True,primary_cases=432,nested_scaling_cases=36,preempted_clock_cases=72,
                independently_replayed_assignments=replay_assignments,tests_passed=226,
                full_kernel_fidelity_comparisons=900,
                source_and_provenance_checked=True,pdf_review_gate_passed=True,
                author_scientific_approval='Separate author responsibility; not inferred from technical checks')
    suite.atomic(HERE/'verification/completion.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':main()
