"""Export only app sources, public demos and a pinned namespaced low-rank wheel.

No existing uploads, user credentials, model caches or unrelated benchmark data
are part of the build context. Output must be a new directory.
"""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--lowrank-wheel',type=Path,required=True)
    p.add_argument('--palm-source',type=Path,default=Path(__file__).resolve().parents[2])
    a=p.parse_args();a.output.mkdir(parents=True,exist_ok=False)
    app=Path(__file__).resolve().parents[1]
    for f in app.rglob('*'):
        if not f.is_file() or any(s in f.relative_to(app).parts for s in ['__pycache__','.pytest_cache','deployment','tests']):
            continue
        if f.suffix not in ['.py','.html','.css','.js','.csv','.json']:
            continue
        dest=a.output/'PALM/webapp'/f.relative_to(app);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    for f in a.palm_source.rglob('*.py'):
        rel=f.relative_to(a.palm_source)
        if rel.parts[0] in ('webapp','low_rank','publications'):  # app copied above; low-rank ships as the wheel
            continue
        if any(s in rel.parts for s in ['__pycache__','tests','data','docs']):
            continue
        dest=a.output/'PALM'/f.relative_to(a.palm_source);dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(f,dest)
    resource = a.palm_source/'features/data/mat2vec_embeddings.json'
    if resource.is_file():
        dest = a.output/'PALM/features/data/mat2vec_embeddings.json'
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(resource,dest)
    for name in ['LICENSE']:
        shutil.copy2(a.palm_source/name,a.output/'PALM'/name)
    deploy=app/'deployment';(a.output/'PALM/webapp/deployment').mkdir(parents=True,exist_ok=True)
    shutil.copy2(deploy/'serve.py',a.output/'PALM/webapp/deployment/serve.py')
    for source,target in [('Dockerfile','Dockerfile'),('requirements-cpu.txt','requirements-cpu.txt'),('compose.yaml','compose.yaml'),('space.README.md','README.md')]:
        shutil.copy2(deploy/source,a.output/target)
    (a.output/'vendor').mkdir();shutil.copy2(a.lowrank_wheel,a.output/'vendor'/a.lowrank_wheel.name)
    (a.output/'.dockerignore').write_text('__pycache__/\n*.pyc\n.git/\n.env\n')
    inventory={str(f.relative_to(a.output)):hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(a.output.rglob('*')) if f.is_file()}
    (a.output/'BUILD_INPUTS_SHA256.json').write_text(json.dumps(inventory,indent=2)+'\n')
    print('Build context:',a.output,';',len(inventory),'files')


if __name__=='__main__':
    main()
