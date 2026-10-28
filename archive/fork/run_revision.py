"""Closed loop: select from hypotheses, acquire fresh RGB, then revise hypotheses."""
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
import numpy as np
import torch
from revisable_graph import fit_pool,select_experiment,revision

BASE=Path('private-runs/icd_fork_20261024/full_v11')
sys.path.insert(0,str(BASE/'source'))
sys.path.insert(0,str(BASE/'source/original'))
from min_pixel_v1 import VAE
from acc_domino_mec_node_calibration import NodeCalibration

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,d):Path(p).write_text(json.dumps(d,indent=2))

def main(a):
    root=Path(a.out).resolve();root.mkdir(parents=True,exist_ok=False);source=Path(__file__).resolve().parent;start=time.time();torch.set_num_threads(2)
    ckpath=BASE/'visual/encoder_decoder.pt';calpath=BASE/'graph/calibration.json'
    assert sha(ckpath)=='a005a5a4e14006bd55904f6a0eeb15a24ef64fa47f35adcdaa824ddf5ac20e0f'
    ck=torch.load(ckpath,map_location='cpu',weights_only=False);model=VAE(7,H=128,W=128,C=3);model.load_state_dict(ck['state_dict']);model.eval()
    cal=NodeCalibration(json.loads(calpath.read_text()),sha(ckpath),ck['roles'])
    pcal=None
    if a.p_calibration:
        from recalibrate_p import piecewise
        pcal=json.loads(Path(a.p_calibration).read_text());assert pcal['status']=='PASS_FROZEN_P_READOUT'
        assert pcal['checkpoint_sha256']==sha(ckpath) and pcal['base_calibration_sha256']==sha(calpath)
    manifest={'pid':os.getpid(),'checkpoint_sha256':sha(ckpath),'calibration_sha256':sha(calpath),'controller_sha256':sha(source/'robot_revision.py'),'source_sha256':{p.name:sha(p) for p in source.glob('*.py')},'round_budget':12,'images_per_round':8,'initial_observations':24,'roles':ck['roles'],'hypothesis_layer':'truncated BDeu, exact score top-32, max 2 parents, ESS 1, fixed seeded numerical ties','truth_access':False,'privacy':'private exploratory preview; not paper result or website publication'}
    save(root/'launch.json',manifest)
    manifest.update(observation_seed_base=int(os.environ.get('ICD_OBS_BASE','82500000')),intervention_seed_base=int(os.environ.get('ICD_DO_BASE','82600000')),P_recalibration_sha256=sha(a.p_calibration) if pcal else None,additional_calibration_images=pcal['calibration_calls'] if pcal else 0)
    save(root/'launch.json',manifest)
    def acquire(role,target=None,r=0):
        folder=root/('observations' if role=='observations' else f'round{r:02d}_pixels')
        cmd=[sys.executable,str(source/'collect_revision.py'),'--role',role,'--round',str(r),'--workers','8','--out',str(folder)]
        if target is not None:cmd+=['--target',str(target)]
        with (root/f'acquisition_{r:02d}.log').open('w') as f:p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        assert p.returncode==0,('PHYSICAL_GATE_BLOCKED',folder,p.returncode)
        assert json.loads((folder/'result.json').read_text())['status']=='PASS'
        data=np.load(folder/'pixels.npz');images=data['images'];encoded=[]
        with torch.no_grad():
            for i in range(0,len(images),32):encoded.append(model.encode(torch.from_numpy(images[i:i+32].transpose(0,3,1,2).copy()).float()/255)[0].numpy())
        z=np.concatenate(encoded);base_read=cal.transform(z)*ck['std']+ck['mean'];read=base_read.copy()
        if pcal:read[:,0]=piecewise(read[:,0],pcal['knots'],pcal['values'])
        binary=(read>=.5).astype(np.int8)
        masks=np.zeros_like(binary)
        for i,j in enumerate(data['targets']):
            if j>=0:masks[i,j]=1
        np.savez_compressed(folder/'pixel_readouts.npz',raw=z,base_calibrated=base_read,calibrated=read,binary=binary,masks=masks)
        return folder,binary,masks
    folder,x,masks=acquire('observations')
    pool,cpd,_=fit_pool(x,masks);save(root/'state_00.json',pool)
    trace=[];allx=[x];allm=[masks]
    for r in range(1,13):
        target,reason=select_experiment(pool,cpd)
        decision={'round':r,'target':target,'target_name':ck['roles'][target],'previous_rows':sum(len(z) for z in allx),'reason':reason,'hypothesis_state_sha256':sha(root/f'state_{r-1:02d}.json'),'decision_before_acquisition':True}
        save(root/f'decision_{r:02d}.json',decision)
        print(json.dumps({'stage':'SELECT','round':r,'target':target,'score':reason['selected_score_nats']}),flush=True)
        folder,x,masks=acquire('intervention',target,r);allx.append(x);allm.append(masks)
        before=pool;pool,cpd,_=fit_pool(np.concatenate(allx),np.concatenate(allm));save(root/f'state_{r:02d}.json',pool)
        means=[x[v::2].mean(0).tolist() for v in [0,1]]
        record={'round':r,'target':target,'decision':decision,'rows':len(x),'pairs':len(x)//2,'observed_means_by_command':means,'observed_paired_difference':(x[1::2]-x[0::2]).mean(0).tolist(),'revision':revision(before,pool),'map_edges':pool['map_edges'],'edge_support':pool['edge_support'],'acquisition_sha256':sha(folder/'pixels.npz'),'readout_sha256':sha(folder/'pixel_readouts.npz'),'representation_frozen':True}
        trace.append(record);save(root/'trace.json',trace)
        print(json.dumps({'stage':'REVISE','round':r,**record['revision']}),flush=True)
    result={'status':'FIXED_BUDGET_COMPLETE','rounds':12,'total_new_images':sum(len(z) for z in allx),'initial_observations':24,'intervention_images':96,'calibration_reused_images':280,'final_map_edges':pool['map_edges'],'rounds_with_withdrawals':[r['round'] for r in trace if r['revision']['withdrawn']],'roles':ck['roles'],'seconds':time.time()-start,'claim_scope':'One private exploratory revisable-graph run. No reference-graph accuracy checked by learner.'}
    result['additional_calibration_images']=pcal['calibration_calls'] if pcal else 0
    result['P_recalibration_sha256']=sha(a.p_calibration) if pcal else None
    save(root/'result.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);p.add_argument('--p-calibration');a=p.parse_args()
    try:main(a)
    except Exception:
        root=Path(a.out);root.mkdir(parents=True,exist_ok=True);save(root/'exit.json',{'status':'BLOCKED','traceback':traceback.format_exc()});raise
    else:save(Path(a.out)/'exit.json',{'status':'COMPLETE','returncode':0})
