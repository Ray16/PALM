"""Replay frozen ODAC25 producers with the included original factor implementation.

Defaults to retained-data comparison; --stage collect optionally re-fetches only
the pinned metadata columns (about 63.4 MB production payload), never full shards.
"""
import argparse
import importlib
from pathlib import Path
import shutil
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--stage', choices=['collect', 'split', 'compare', 'evaluate', 'audit', 'finalize'], default='evaluate')
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--resume', action='store_true')
args, remaining = parser.parse_known_args()
SOURCE = Path(__file__).resolve().parent
ROOT = args.output.resolve()
if ROOT == SOURCE or SOURCE in ROOT.parents:
    parser.error('Replay output must be outside the retained case evidence')
if ROOT.exists() and any(ROOT.iterdir()) and not args.resume:
    parser.error('Nonempty output requires --resume')
if not ROOT.exists() or not any(ROOT.iterdir()):
    shutil.copytree(SOURCE, ROOT, dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__'))
sys.path.insert(0, str(ROOT / 'method_source'))
sys.path.insert(1, str(ROOT))
import lowrank
import build_split
build_split.PACKAGE = ROOT / 'method_source'
if not Path(lowrank.__file__).resolve().is_relative_to(ROOT / 'method_source'):
    raise RuntimeError('Frozen ODAC25 method source was shadowed')

modules = dict(collect='collect', split='build_split', compare='compare_original',
               evaluate='evaluate_original_formulas', audit='audit_original_result', finalize='finalize_original')
sys.argv = [sys.argv[0]] + remaining
importlib.import_module(modules[args.stage]).main()
