"""Remote RGB acquisition. Oracle measurements are written to a separate file.

Observations use independent physical initial conditions; bell background
activation is an actual gripper action. Calibration and graph experiments use
paired clean contexts with the same root input, not different control contexts.
"""
import argparse,collections,hashlib,json,os,time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from robot_seven import RobotSeven

BASES={'smoke':81000000,'train':81100000,'validation':81200000,'readout_test':81300000,'observations':81400000,'calibration':81500000,'intervention':81600000}
CLEAN={'lean':[False]*4,'bell_noise':False,'lamp_noise':False}

def context(seed):
 rng=np.random.default_rng(seed+1234567)
 return {'lean':(rng.random(4)<.16).tolist(),'bell_noise':bool(rng.random()<.15),'lamp_noise':bool(rng.random()<.12)},bool(rng.random()<.25)

def jobs(role,target=None,round_index=0):
 base=BASES[role];out=[]
 def obs(n):
  for i in range(n):
   seed=base+i;ctx,root=context(seed);out.append({'seed':seed,'target':None,'value':None,'root_on':root,'context':ctx})
 def pairs(per_target,targets=range(7),offset=1000):
  for j in targets:
   for i in range(per_target):
    for v in [0,1]:out.append({'seed':base+offset+j*10000+i,'target':j,'value':v,'root_on':True,'context':dict(CLEAN)})
 if role=='smoke':obs(16);pairs(1)
 elif role=='train':obs(196);pairs(14)
 elif role=='validation':obs(28);pairs(2)
 elif role=='readout_test':obs(56);pairs(4)
 elif role=='observations':obs(512)
 elif role=='calibration':pairs(20)
 elif role=='intervention':pairs(32,[int(target)],offset=round_index*100000)
 else:raise ValueError(role)
 return out

def gate(a):
 return {'target':bool(a['target_realized']),'stationary':bool(a['settled']),'penetration':a['min_contact_distance']>=-.001,'no_unintended_robot_contact':not a['unexpected_robot_contacts'],'pusher_travel':bool(a['pusher_within_physical_travel']),'no_post_release_reset':a['post_release_pose_resets']==0}

def collect(job):
 env=RobotSeven()
 try:
  image,labels,audit=env.sample_robot(**job)
  full_contacts=audit.pop('contacts');summary={}
  for c in full_contacts:
   key=f"{c['actor']}:{c['other']}:{c['link']}:{c['phase']}"
   r=summary.setdefault(key,{'count':0,'min_distance_m':0.,'max_force_N':0.});r['count']+=1;r['min_distance_m']=min(r['min_distance_m'],c['distance']);r['max_force_N']=max(r['max_force_N'],c['force'])
  audit['contact_summary']=summary;audit['physical_gate']=gate(audit)
  if not all(audit['physical_gate'].values()):audit['failed_full_contacts']=full_contacts
  return image,np.array(labels,np.float32),audit
 finally:env.close()

def run(role,out,target=None,round_index=0,workers=4):
 out=Path(out);out.mkdir(parents=True,exist_ok=False);start=time.time();spec=jobs(role,target,round_index)
 manifest={'role':role,'jobs':spec,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'controller_sha256':hashlib.sha256(Path(__file__).with_name('robot_seven.py').read_bytes()).hexdigest(),'pid':os.getpid(),'workers':workers,'graph_truth_supplied':False,'target':target,'round':round_index}
 (out/'launch.json').write_text(json.dumps(manifest,indent=2));images=[];labels=[];audits=[]
 with ProcessPoolExecutor(workers) as pool:
  for k,(im,y,a) in enumerate(pool.map(collect,spec)):
   images.append(im);labels.append(y);audits.append(a)
   (out/f'row{k:04d}_audit.json').write_text(json.dumps(a,separators=(',',':')))
   print(json.dumps({'role':role,'completed':k+1,'total':len(spec),'physical_gate':all(a['physical_gate'].values())}),flush=True)
 targets=np.array([-1 if s['target'] is None else s['target'] for s in spec]);values=np.array([-1 if s['value'] is None else s['value'] for s in spec]);seeds=np.array([s['seed'] for s in spec])
 # Only anchor roles contain full labels; the graph/readout path has no oracle.
 data={'images':np.array(images),'targets':targets,'commands':values,'seeds':seeds}
 if role in ['train','validation','readout_test']:data['labels']=np.array(labels)
 np.savez_compressed(out/'pixels.npz',**data);np.savez_compressed(out/'oracle_measurements.npz',labels=np.array(labels),seeds=seeds)
 failures={str(k):[n for n,v in a['physical_gate'].items() if not v] for k,a in enumerate(audits) if not all(a['physical_gate'].values())}
 result={'status':'PASS' if not failures else 'PHYSICS_BLOCKED','rows':len(spec),'failures':failures,'seconds':time.time()-start,'pixels_sha256':hashlib.sha256((out/'pixels.npz').read_bytes()).hexdigest(),'role':role}
 (out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True);return result

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--role',choices=list(BASES),required=True);ap.add_argument('--out',required=True);ap.add_argument('--target',type=int);ap.add_argument('--round',type=int,default=0);ap.add_argument('--workers',type=int,default=4);a=ap.parse_args();r=run(a.role,a.out,a.target,a.round,a.workers);raise SystemExit(0 if r['status']=='PASS' else 2)
