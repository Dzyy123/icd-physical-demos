"""Fresh physical acquisition; separate privileged audit data from learner RGB."""
import argparse,hashlib,json,os,sys,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np

BASE=Path('private-runs/icd_fork_20261024/full_v11/source')
sys.path.insert(0,str(BASE))
from collect_fork import collect,context
import collect_fork
from robot_revision import RobotFork
collect_fork.RobotFork=RobotFork

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def specifications(role,target=None,round_index=0):
    result=[]
    if role=='observations':
        for i in range(24):
            seed=int(os.environ.get('ICD_OBS_BASE','82500000'))+i;ctx,root=context(seed)
            result.append({'seed':seed,'target':None,'value':None,'root_on':bool(root),'context':ctx})
    else:
        for i in range(4):
            seed=int(os.environ.get('ICD_DO_BASE','82600000'))+round_index*10000+i;ctx,root=context(seed)
            for value in [0,1]:result.append({'seed':seed,'target':target,'value':value,'root_on':bool(root),'context':ctx})
    return result

def run(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    spec=specifications(a.role,a.target,a.round);start=time.time()
    representative=0
    if a.role!='observations':
        # Choose from setup only, before acquiring outcomes. Every pair is learned.
        def preference(i):
            s=spec[2*i];c=s['context'];clean=not any(c['lean']) and not c['bell_noise'] and not c['lamp_noise']
            return (int(clean and (s['root_on'] or s['target']==0)),int(clean),-i)
        representative=max(range(4),key=preference)
    manifest={'role':a.role,'round':a.round,'target':a.target,'jobs':spec,'representative_pair_index':representative,'selection_rule':'Prefer clean setup and active root, then clean setup, then lowest pair index; selected before outcomes. Never drop other pairs.',
              'controller_sha256':sha(Path(__file__).with_name('robot_revision.py')),'collection_sha256':sha(__file__),'pid':os.getpid(),'workers':a.workers,'known_graph_input':False}
    (out/'launch.json').write_text(json.dumps(manifest,indent=2))
    images=[];labels=[];failures={}
    with ProcessPoolExecutor(a.workers) as pool:
        for k,(im,y,audit) in enumerate(pool.map(collect,spec)):
            images.append(im);labels.append(y)
            (out/f'row{k:04d}_audit.json').write_text(json.dumps(audit,separators=(',',':')))
            failed=[name for name,v in audit['physical_gate'].items() if not v]
            if failed:failures[str(k)]=failed
            print(json.dumps({'completed':k+1,'total':len(spec),'failed_gates':failed}),flush=True)
    np.savez_compressed(out/'pixels.npz',images=np.array(images),targets=np.array([-1 if s['target'] is None else s['target'] for s in spec]),commands=np.array([-1 if s['value'] is None else s['value'] for s in spec]),seeds=np.array([s['seed'] for s in spec]))
    np.savez_compressed(out/'oracle_measurements.npz',labels=np.array(labels))
    result={'status':'PASS' if not failures else 'PHYSICS_BLOCKED','rows':len(spec),'failures':failures,'seconds':time.time()-start,'pixels_sha256':sha(out/'pixels.npz')}
    (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
    return 0 if not failures else 2

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--role',choices=['observations','intervention'],required=True);p.add_argument('--target',type=int);p.add_argument('--round',type=int,default=0);p.add_argument('--workers',type=int,default=8);p.add_argument('--out',required=True)
    raise SystemExit(run(p.parse_args()))
