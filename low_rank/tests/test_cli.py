"""Input-boundary and CLI smoke regressions; numerical methods stay unchanged."""
import csv
import json

import numpy as np
import pytest

from lowrank import SplitSpec
from lowrank.splitters.cli import _load_features, _parse_params, main


@pytest.mark.parametrize('ratios', [[np.nan, 1], [np.inf, 1], [0, 1], [-1, 1], ['8', 2]])
def test_bad_fold_ratios_are_rejected(ratios):
    with pytest.raises(ValueError, match='finite positive'):
        SplitSpec(ratios, ['train', 'test'])


@pytest.mark.parametrize('names', [['same', 'same'], ['', 'test'], [None, 'test']])
def test_fold_labels_cannot_collapse_assignments(names):
    with pytest.raises(ValueError, match='names'):
        SplitSpec([8, 2], names)


@pytest.mark.parametrize('kwargs', [{'epsilon': np.nan}, {'epsilon': -1}, {'seed': -1}, {'seed': 1.5}])
def test_invalid_seed_or_tolerance_is_rejected(kwargs):
    with pytest.raises(ValueError):
        SplitSpec(**kwargs)


def test_valid_geometry_and_scalar_parameter_types_are_preserved():
    spec = SplitSpec([8, 2], ['train', 'test'], seed=np.int64(7))
    assert spec.splits == [8, 2] and spec.epsilon == 0.05
    assert _parse_params(['rank=64', 'fm=false', 'must_link=0.4', 'metric=tanimoto']) == {
        'rank': 64, 'fm': False, 'must_link': 0.4, 'metric': 'tanimoto'}


@pytest.mark.parametrize('item', ['rank', '=12', 'rank='])
def test_malformed_method_parameters_fail_early(item):
    with pytest.raises(ValueError, match='key=value'):
        _parse_params([item])


@pytest.mark.parametrize('raw', [[], {}, {'a': [1], 'b': [1, 2]}, {'a': [np.nan]}])
def test_invalid_json_features_are_reported(tmp_path, raw):
    path = tmp_path/'input.json'
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        _load_features(path)


@pytest.mark.parametrize('ids', [np.array(['same', 'same']), np.array(['only-one'])])
def test_npz_ids_cannot_drop_or_misalign_entities(tmp_path, ids):
    path = tmp_path/'input.npz'
    np.savez(path, X=np.ones((2, 3)), ids=ids)
    with pytest.raises(ValueError, match='IDs|ids'):
        _load_features(path)


def test_trusted_npz_object_ids_keep_historical_compatibility(tmp_path):
    path = tmp_path/'input.npz'
    np.savez(path, X=np.eye(2, dtype=np.float32), ids=np.array(['a', 'b'], dtype=object))
    result = _load_features(path)
    assert list(result) == ['a', 'b'] and result['a'].dtype == np.float32


def test_cli_real_split_creates_parent_directory_and_complete_csv(tmp_path, capsys):
    path = tmp_path/'features.json'
    path.write_text(json.dumps({str(i): [int(i % 2 == 0), 1, int(i % 3 == 0)] for i in range(10)}))
    output = tmp_path/'nested'/'assignments.csv'
    assert main(['split', '--method', 'lowrank', '--features', str(path),
                 '--param', 'rank=3', '--param', 'metric=tanimoto',
                 '--param', 'leakage_diagnostic=none', '--out', str(output)]) == 0
    with output.open() as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 10 and len({r['id'] for r in rows}) == 10
    assert sum(r['split'] == 'train' for r in rows) == 8
    assert 'wrote 10 assignments' in capsys.readouterr().out


def test_cli_missing_input_has_actionable_error_without_traceback(tmp_path, capsys):
    with pytest.raises(SystemExit) as error:
        main(['split', '--method', 'lowrank', '--features', str(tmp_path/'missing.json')])
    assert error.value.code == 2
    output = capsys.readouterr().err
    assert 'missing.json' in output and 'Traceback' not in output


def test_describe_unknown_method_does_not_succeed_silently(capsys):
    with pytest.raises(SystemExit) as error:
        main(['describe', '--method', 'does-not-exist'])
    assert error.value.code == 2 and 'unknown method' in capsys.readouterr().err
