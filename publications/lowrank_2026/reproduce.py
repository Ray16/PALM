"""Run a frozen paper producer with explicit, separate output."""
from pathlib import Path
import argparse, os, runpy, sys, shutil
p=argparse.ArgumentParser(add_help=False)
p.add_argument('study')
p.add_argument('--output', type=Path, required=True)
p.add_argument('--resume', action='store_true')
p.add_argument('--gpu', action='store_true')
a, extra=p.parse_known_args()
base=Path(__file__).resolve().parent
scripts=base/'reproduction/comparisons/paper_lowrank'
allowed={f.stem for f in scripts.glob('revision_*.py') if 'worker' not in f.stem and 'cached' not in f.stem}
if a.study not in allowed: p.error('select a study producer: '+', '.join(sorted(allowed)))
out=a.output.resolve()
if out==base/'paper' or base/'paper' in out.parents: p.error('output must not overwrite retained paper evidence')
if out.exists() and any(out.iterdir()) and not a.resume: p.error('nonempty output requires explicit --resume')
out.mkdir(parents=True,exist_ok=True)
if not a.gpu: os.environ['CUDA_VISIBLE_DEVICES']=''
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']:os.environ[key]='2'
os.environ['OMP_WAIT_POLICY']='PASSIVE'
os.environ['PYTORCH_NVML_BASED_CUDA_CHECK']='1'
os.environ['LOWRANK_DATA']=str(base/'inputs')
sys.path.insert(0,str(scripts))
import mlst_study
mlst_study.OUT=out
if a.study=='revision_external_control':shutil.copytree(base/'paper/external_simpd',out/'external_simpd',dirs_exist_ok=True)
sys.argv=[str(scripts/(a.study+'.py'))]+extra
runpy.run_path(sys.argv[0],run_name='__main__')
