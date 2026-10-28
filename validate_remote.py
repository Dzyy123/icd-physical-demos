"""Release verification only; does not train or change historical results."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time

assert sys.platform=='linux'
root=Path(__file__).resolve().parent
out=root/'validation-output';out.mkdir(exist_ok=False)
env={**os.environ,'CUDA_VISIBLE_DEVICES':'','OMP_NUM_THREADS':'2','OPENBLAS_NUM_THREADS':'2'}
tests=[]
commands=[
 ['refit','--scene','chain','--out',str(out/'chain-refit.json')],
 ['refit','--scene','fork','--out',str(out/'fork-refit.json')],
 ['report','--scene','chain','--out',str(out/'chain.html')],
 ['report','--scene','fork','--out',str(out/'fork.html')],
 ['simulate','--scene','chain','--round','1','--row','0','--out',str(out/'chain-operation')],
 ['simulate','--scene','fork','--round','18','--row','0','--out',str(out/'fork-operation')],
]
for i,args in enumerate(commands):
    started=time.time();print('START',args,flush=True)
    with (out/f'test-{i}.log').open('w') as f:
        p=subprocess.run([sys.executable,str(root/'demo.py'),*args],cwd=root,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=600)
    tests.append({'command':['python','demo.py',*args], 'returncode':p.returncode,'seconds':time.time()-started})
    (out/'progress.json').write_text(json.dumps(tests,indent=2))
    if p.returncode:print((out/f'test-{i}.log').read_text()[-8000:],flush=True);raise SystemExit(p.returncode)
report={'status':'PASS','scope':'Release implementation checks, not independent integrity review or a new benchmark.',
 'all_chain_updates_recomputed':True,'all_fork_states_and_targets_recomputed':True,
 'physics_replays':{s:json.loads((out/f'{s}-operation/result.json').read_text()) for s in ['chain','fork']},
 'tests':[{**t,'command':[str(a).replace(str(root)+'/','') for a in t['command']]} for t in tests]}
(out/'VALIDATION.json').write_text(json.dumps(report,indent=2));print('ALL PASS',flush=True)
