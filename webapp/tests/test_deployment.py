"""Deployment regressions: private access, compute limits and cache correctness."""
import io
import json
import os
import subprocess
import sys
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PALM.webapp import jobs, worker, reduction
from PALM.webapp.uploads import extract_dataset_zip


def test_api_startup_does_not_import_scientific_pipeline():
    code = "import PALM.webapp.app,sys; assert 'torch' not in sys.modules; assert 'numpy' not in sys.modules; assert 'PALM.benchmarks.datasail_pipeline.pipeline' not in sys.modules"
    subprocess.run([sys.executable, '-c', code], cwd=worker.ROOT_DIR, check=True)


def test_defaults_match_pipeline():
    from PALM.webapp.embedding import DEFAULT_FEATURE_SETS
    from PALM.features.molecule_features import _MOLECULE_DEFAULT_SETS
    assert DEFAULT_FEATURE_SETS['molecule'] == _MOLECULE_DEFAULT_SETS


def test_reduction_cache_reuses_only_identical_features_and_seed(tmp_path, monkeypatch):
    monkeypatch.setenv('PALM_CACHE_DIR', str(tmp_path));calls=[]
    def native(x, method, seed=42):
        calls.append((x.copy(), seed));return np.column_stack([x[:, 0], np.full(len(x), seed)])
    monkeypatch.setattr(reduction, '_native_reduce', native)
    x=np.arange(24,dtype=float).reshape(6,4)
    first=reduction.cached_reduce(x,'tsne',seed=42)
    assert np.array_equal(first,reduction.cached_reduce(x.copy(),'tsne',seed=42))
    assert len(calls)==1
    changed=x.copy();changed[-1,-1]+=1
    reduction.cached_reduce(changed,'tsne',seed=42)
    reduction.cached_reduce(x,'tsne',seed=43)
    assert len(calls)==3


def test_reduction_concurrent_requests_compute_once(tmp_path, monkeypatch):
    monkeypatch.setenv('PALM_CACHE_DIR',str(tmp_path));calls=[]
    def native(x, method, seed=42):
        calls.append(1);time.sleep(.05);return np.asarray(x[:,:2])
    monkeypatch.setattr(reduction,'_native_reduce',native)
    x=np.arange(24,dtype=float).reshape(6,4)
    with ThreadPoolExecutor(2) as pool:out=list(pool.map(lambda _:reduction.cached_reduce(x,'tsne'),range(2)))
    assert len(calls)==1 and np.array_equal(*out)


def test_shared_compute_budget_and_cancellation(monkeypatch):
    monkeypatch.setattr(worker,'_WORK_SLOTS',threading.BoundedSemaphore(1))
    started=threading.Event();release=threading.Event();waiting=threading.Event();calls=[]
    def execute(task, progress, **kw):
        calls.append(task['kind']);started.set();assert release.wait(3)
    monkeypatch.setattr(worker,'_run_task',execute)
    def cancel(pct,msg):
        waiting.set();raise jobs.RunCancelled()
    with ThreadPoolExecutor(2) as pool:
        first=pool.submit(worker.run_task,{'kind':'run'},lambda p,m:None,tick=.02)
        assert started.wait(2)
        second=pool.submit(worker.run_task,{'kind':'map'},cancel,tick=.02)
        assert waiting.wait(2)
        with pytest.raises(jobs.RunCancelled):second.result(2)
        assert calls==['run'];release.set();first.result(2)


def test_queue_limits_are_atomic_and_client_scoped(tmp_path):
    store=jobs.Store(tmp_path,lambda r,p:None);store.max_pending=2;store.max_client_pending=1
    store.submit('dataset',{},'alice','a')
    with pytest.raises(jobs.QueueFull):store.submit('dataset',{},'alice','a2')
    store.submit('dataset',{},'bob','b')
    with pytest.raises(jobs.QueueFull):store.submit('dataset',{},'carol','c')
    assert len(store.list_runs(None))==2


@pytest.fixture
def api(tmp_path,monkeypatch):
    import PALM.webapp.app as module
    store=jobs.Store(tmp_path,lambda r,p:None);monkeypatch.setattr(module,'store',store)
    monkeypatch.setattr(module,'RATE_LIMIT_MAX',10000)
    ds_id,directory=store.new_dataset_dir();csv=directory/'data.csv';csv.write_text('smiles,y\nCCO,1\nCCC,2\n')
    store.add_dataset(ds_id,csv,csv.name,'alice',{'columns':['smiles','y']})
    run=store.submit(ds_id,{'dataset_name':'demo'},'alice','demo');store.runs[run['id']]['status']='completed'
    return module,store,ds_id,run['id'],TestClient(module.app,headers={'X-Client-Id':'alice'})


def test_private_app_requires_password(api,monkeypatch):
    module,store,ds_id,run_id,client=api;monkeypatch.setenv('PALM_AUTH_PASSWORD','test-password')
    assert client.get('/healthz').status_code==200
    assert client.get('/').status_code==401
    assert client.get('/api/info').status_code==401
    assert client.get('/',auth=('palm','test-password')).status_code==200
    assert client.get('/',auth=('palm','wrong')).status_code==401


def test_other_browser_cannot_access_or_cancel_runs(api):
    module,store,ds_id,run_id,client=api
    assert client.get(f'/api/datasets/{ds_id}').status_code==200
    for path in [f'/api/datasets/{ds_id}',f'/api/runs/{run_id}',f'/api/runs/{run_id}/files',f'/api/runs/{run_id}/metrics']:
        assert client.get(path,headers={'X-Client-Id':'bob'}).status_code==404
    assert client.post(f'/api/runs/{run_id}/cancel',headers={'X-Client-Id':'bob'}).status_code==404
    assert client.delete(f'/api/runs/{run_id}',headers={'X-Client-Id':'bob'}).status_code==404


def test_stream_auth_and_no_gzip_buffering(api):
    module,store,ds_id,run_id,client=api
    other=TestClient(module.app)
    assert other.get(f'/api/runs/{run_id}/stream?client_id=bob').status_code==404
    response=other.get(f'/api/runs/{run_id}/stream?client_id=alice',headers={'Accept-Encoding':'gzip'})
    assert response.status_code==200 and 'data:' in response.text
    assert response.headers.get('content-encoding') is None


def test_completed_download_is_built_once(api,monkeypatch):
    module,store,ds_id,run_id,client=api
    out=store.output_dir(store.get_run(run_id));out.mkdir();(out/'one.txt').write_text('data')
    original=module._zip_dir;calls=[]
    def build(*args):calls.append(1);return original(*args)
    monkeypatch.setattr(module,'_zip_dir',build)
    a=client.get(f'/api/runs/{run_id}/download');b=client.get(f'/api/runs/{run_id}/download')
    assert a.status_code==b.status_code==200 and a.content==b.content and calls==[1]


def test_upload_size_and_zip_paths_are_enforced(api,monkeypatch):
    module,store,ds_id,run_id,client=api;monkeypatch.setattr(module,'MAX_UPLOAD_BYTES',8)
    assert client.post('/api/upload',files={'file':('data.csv',b'over-the-limit')}).status_code==413
    monkeypatch.setattr(module,'MAX_UPLOAD_BYTES',10000)
    buffer=io.BytesIO()
    with zipfile.ZipFile(buffer,'w') as z:z.writestr('../outside.csv','private')
    assert client.post('/api/upload',files={'file':('data.zip',buffer.getvalue())}).status_code==400
    assert len(list(store.root.glob('*/dataset.json')))==1


def test_zip_expansion_is_bounded_before_writing(tmp_path):
    p=tmp_path/'input.zip'
    with zipfile.ZipFile(p,'w',zipfile.ZIP_DEFLATED) as z:z.writestr('large.csv','x'*5000)
    target=tmp_path/'extracted'
    with pytest.raises(ValueError,match='extraction limit'):extract_dataset_zip(p,target,1000)
    assert not target.exists()


def test_browser_cookie_authorizes_direct_download_links(api):
    module,store,ds_id,run_id,client=api
    assert client.get('/api/info').status_code==200
    assert client.cookies.get('palm_client')=='alice'
    client.headers.pop('X-Client-Id')
    out=store.output_dir(store.get_run(run_id));out.mkdir();(out/'one.txt').write_text('data')
    assert client.get(f'/api/runs/{run_id}/download').status_code==200


def test_corrupt_coordinate_cache_is_recomputed(tmp_path,monkeypatch):
    monkeypatch.setenv('PALM_CACHE_DIR',str(tmp_path));calls=[]
    def native(x,method,seed=42):calls.append(1);return x[:,:2].copy()
    monkeypatch.setattr(reduction,'_native_reduce',native)
    x=np.arange(24,dtype=float).reshape(6,4)
    expected=reduction.cached_reduce(x,'tsne')
    path=next((tmp_path/'webapp_reductions').glob('*.npz'));path.write_bytes(path.read_bytes()[:10])
    assert np.array_equal(expected,reduction.cached_reduce(x,'tsne')) and len(calls)==2


def test_datasail_solver_honors_worker_thread_budget(monkeypatch):
    from datasail.solver.utils import solve
    import inspect
    original=solve.__defaults__
    monkeypatch.setenv('WEBAPP_WORKER_THREADS','2')
    try:
        worker._configure_datasail_threads()
        assert inspect.signature(solve).parameters['num_threads'].default==2
    finally:
        solve.__defaults__=original


def test_datasail_custom_distance_matrix_callback_preserves_results():
    from datasail.sail import datasail
    names=[str(i) for i in range(20)]
    matrix=np.abs(np.arange(20)[:,None]-np.arange(20)[None,:])/19
    original=(names,matrix)
    kwargs=worker._matrix_callables(dict(e_dist=original,f_dist=original))
    assert kwargs['e_dist']() is original and kwargs['f_dist']() is original
    result=datasail(techniques=['R'],splits=[8,2],names=['train','test'],runs=1,
                    e_type='O',f_type='O',e_data={n:np.array([i,1.]) for i,n in enumerate(names)},
                    f_data={n:np.array([i,1.]) for i,n in enumerate(names)},
                    inter=[(n,n) for n in names],**kwargs)[2]['R'][0]
    assert len(result)==20 and sum(v=='train' for v in result.values())==16


def test_download_owner_token_works_when_browser_cookies_are_blocked(api):
    module,store,ds_id,run_id,client=api
    out=store.output_dir(store.get_run(run_id));out.mkdir();(out/'one.txt').write_text('data')
    independent=TestClient(module.app)
    assert independent.get(f'/api/runs/{run_id}/download?client_id=alice').status_code==200
    assert independent.get(f'/api/runs/{run_id}/download?client_id=bob').status_code==404
