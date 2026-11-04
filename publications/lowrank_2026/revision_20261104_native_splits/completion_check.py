"""Completion evidence for the metadata-only native-split extension."""
from __future__ import annotations
import json
import hashlib
import io
from pathlib import Path

import numpy as np

from prepare import atomic,sha,DATASETS
import report

HERE=Path(__file__).resolve().parent
PAPER=HERE.parent/'paper'


def main():
    m,rows,failures,missing=report.verified()
    ledger=json.loads((HERE/'transfer_ledger.json').read_text())
    if ledger['received_bytes']>200*1024**2:raise RuntimeError('metadata payload budget exceeded')
    if ledger['received_bytes']!=90293593:raise RuntimeError('dataset transfer inventory changed; update reviewed accounting explicitly')
    tests=json.loads((HERE/'verification/tests.json').read_text())
    if not tests['passed'] or tests['tests_passed']!=5:raise RuntimeError('native regression suite')
    for name,digest in tests['sources'].items():
        if sha(HERE/name)!=digest:raise RuntimeError('tested native source drift')
    pairs=queries=0
    maps=json.loads((HERE/'preparation/excluded_row_maps.json').read_text())
    for dataset in DATASETS:
        identity=maps[dataset];mask=np.ones(identity['original_n'],dtype=bool)
        mask[identity['excluded_original_rows']]=False;retained=np.flatnonzero(mask)
        buffer=io.BytesIO();np.save(buffer,retained)
        if len(retained)!=identity['retained_n'] or hashlib.sha256(buffer.getvalue()).hexdigest()!=identity['map_sha256']:
            raise RuntimeError('retained-row identity reconstruction')
        validation=json.loads((HERE/'verification'/f'independent_{dataset}.json').read_text())
        if not validation['passed'] or validation['validator_sha256']!=sha(HERE/'validate.py'):
            raise RuntimeError('independent native validation missing/stale')
        if validation['manifest_sha256']!=sha(HERE/'run/manifest.json'):raise RuntimeError('independent manifest drift')
        if validation['max_pair_discrepancy']>1e-12 or validation['max_query_discrepancy']>1e-12:
            raise RuntimeError('independent original-kernel discrepancy')
        for comparison in validation['comparisons']:
            if sha(HERE/'run/rows'/(comparison['case']+'.json'))!=comparison['row_sha256']:
                raise RuntimeError('independently checked case changed')
        pairs+=validation['independent_pair_checks'];queries+=validation['full_training_query_checks']
        completed_methods={r['job']['method'] for r in rows if r['job']['dataset']==dataset}
        if not {'native','random','exact_assignment','corridor_plain'}<=completed_methods:
            raise RuntimeError('missing completed method for '+dataset)
    if pairs!=3072 or queries!=168:raise RuntimeError('independent native replay scope mismatch')
    reconstruction=json.loads((HERE/'verification/factor_reconstruction_test.json').read_text())
    if not reconstruction['passed'] or reconstruction['auditor_sha256']!=sha(HERE/'factor_audit.py'):
        raise RuntimeError('native factor reconstruction regression')
    fidelity_cases=0
    for dataset in DATASETS:
        for seed in range(3):
            record=json.loads((HERE/'verification'/f'factor_fidelity_{dataset}__s{seed}.json').read_text())
            if record['source_sha256']!=sha(HERE/'factor_audit.py') or record['manifest_sha256']!=sha(HERE/'run/manifest.json'):
                raise RuntimeError('native fidelity provenance')
            if len(record['results'])!=2:raise RuntimeError('native fidelity case count')
            for result in record['results']:
                if sha(HERE/'run/rows'/(result['case']+'.json'))!=result['row_sha256']:
                    raise RuntimeError('native fidelity assignment drift')
                fidelity_cases+=1
    provenance=json.loads((HERE/'analysis/provenance.json').read_text())
    if provenance['report_sha256']!=sha(HERE/'report.py'):raise RuntimeError('native report source drift')
    for name,digest in provenance['generated_files'].items():
        if sha(PAPER/name)!=digest:raise RuntimeError('native generated table/number drift')
    for name,digest in provenance['observed_row_hashes'].items():
        if sha(HERE/'run/rows'/(name+'.json'))!=digest:raise RuntimeError('native row changed')
    old=json.loads((HERE.parent/'revision_20261026/verification/completion_gate.json').read_text())
    controlled=json.loads((HERE.parent/'revision_20261103_controls/verification/completion.json').read_text())
    if not old['technical_checks_passed'] or not controlled['passed']:raise RuntimeError('existing manuscript/evidence gate')
    result=dict(passed=True,expected_cases=30,completed_cases=len(rows),budget_outcomes=len(failures),
                metadata_payload_bytes=ledger['received_bytes'],regression_tests=5,
                all_saved_assignment_and_metric_aggregations_replayed=True,
                independent_rdkit_pairs=pairs,independent_full_training_queries=queries,
                returned_cut_fidelity_cases=fidelity_cases,
                native_roles_preserved=True,source_and_table_identities_checked=True,
                native_creation_times='not supplied and never imputed from loading',
                author_scientific_approval='Separate author responsibility, not inferred')
    atomic(HERE/'verification/completion.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
