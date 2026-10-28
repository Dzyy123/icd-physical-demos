"""Append a fixed eight rounds without changing the parent learner or data.

Uses only saved pixel readouts, interventions, and frozen inference inputs.
No evaluation file, mechanism reference, or oracle measurement is loaded.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='2'
os.environ['OPENBLAS_NUM_THREADS']='2'


def main(a):
    os.umask(0o077)
    parent=Path(a.parent).resolve()
    source=parent.parent/'source'
    sys.path.insert(0,str(source))
    import run_revision as base
    from recalibrate_p import piecewise
    np,torch=base.np,base.torch
    sha,save=base.sha,base.save
    torch.set_num_threads(2)
    root=Path(a.out).resolve()
    root.mkdir(parents=True,exist_ok=False)
    start=time.time()
    launch=json.loads((parent/'launch.json').read_text())
    collector_source=Path(a.collector_source).resolve() if a.collector_source else source
    repair=None
    if a.collector_source:
        repair=json.loads(Path(a.repair_check).read_text())
        assert repair['status']=='PASS' and repair['selected_offset']==-.005
        assert sha(collector_source/'collect_revision.py')==launch['source_sha256']['collect_revision.py']
    old_result=json.loads((parent/'result.json').read_text())
    assert old_result['status']=='FIXED_BUDGET_COMPLETE' and old_result['rounds']==12
    assert a.additional_rounds==8
    for name in ['run_revision.py','revisable_graph.py','collect_revision.py','robot_revision.py','recalibrate_p.py']:
        assert sha(source/name)==launch['source_sha256'][name],name
    ckpath=base.BASE/'visual/encoder_decoder.pt'
    calpath=base.BASE/'graph/calibration.json'
    ppath=parent.parent/'P_calibration/P_recalibration.json'
    assert sha(ckpath)==launch['checkpoint_sha256']
    assert sha(calpath)==launch['calibration_sha256']
    assert sha(ppath)==launch['P_recalibration_sha256']
    ck=torch.load(ckpath,map_location='cpu',weights_only=False)
    assert ck['roles']==launch['roles']
    model=base.VAE(7,H=128,W=128,C=3)
    model.load_state_dict(ck['state_dict']);model.eval()
    cal=base.NodeCalibration(json.loads(calpath.read_text()),sha(ckpath),ck['roles'])
    pcal=json.loads(ppath.read_text())
    assert pcal['status']=='PASS_FROZEN_P_READOUT'
    os.environ['ICD_OBS_BASE']=str(launch['observation_seed_base'])
    os.environ['ICD_DO_BASE']=str(launch['intervention_seed_base'])
    trace=json.loads((parent/'trace.json').read_text())
    assert len(trace)==12
    inherited={}
    allx=[];allm=[];old_seeds=set()
    for folder in [parent/'observations']+[parent/f'round{r:02d}_pixels' for r in range(1,13)]:
        (root/folder.name).symlink_to(folder,target_is_directory=True)
        d=np.load(folder/'pixel_readouts.npz')
        allx.append(d['binary']);allm.append(d['masks'])
        old_seeds.update(np.load(folder/'pixels.npz')['seeds'].tolist())
        inherited[str(folder/'pixel_readouts.npz')]=sha(folder/'pixel_readouts.npz')
    for pattern in ['state_*.json','decision_*.json']:
        for p in parent.glob(pattern):
            (root/p.name).symlink_to(p)
            inherited[str(p)]=sha(p)
    pool,cpd,_=base.fit_pool(np.concatenate(allx),np.concatenate(allm))
    prev=json.loads((parent/'state_12.json').read_text())
    assert pool['map_edges']==prev['map_edges'] and abs(pool['map_score']-prev['map_score'])<1e-9
    assert [g['code'] for g in pool['graphs']]==[g['code'] for g in prev['graphs']]
    budget=12+a.additional_rounds
    manifest={**launch,'pid':os.getpid(),'round_budget':budget,'parent_run':str(parent),
              'parent_result_sha256':sha(parent/'result.json'),'inherited_files_sha256':inherited,
              'continuation_source_sha256':sha(Path(__file__)),'additional_round_budget':a.additional_rounds,
              'starting_pool_recomputed_exactly':True,'original_algorithm_unchanged':True,
              'adaptive_extension_after_original_result':True,'truth_access':False}
    if repair:
        manifest.update(controller_sha256=sha(collector_source/'robot_revision.py'),
                        parent_controller_sha256=launch['controller_sha256'],
                        controller_repair='F1 hold lateral offset +25 mm -> +20 mm; all gates unchanged',
                        repair_check_sha256=sha(a.repair_check),collector_source=str(collector_source))
    save(root/'launch.json',manifest);save(root/'trace.json',trace)
    for r in range(13,budget+1):
        target,reason=base.select_experiment(pool,cpd)
        decision={'round':r,'target':target,'target_name':ck['roles'][target],
                  'previous_rows':sum(map(len,allx)),'reason':reason,
                  'hypothesis_state_sha256':sha(root/f'state_{r-1:02d}.json'),'decision_before_acquisition':True}
        save(root/f'decision_{r:02d}.json',decision)
        print(json.dumps({'stage':'SELECT','round':r,'target':target,'score':reason['selected_score_nats']}),flush=True)
        folder=root/f'round{r:02d}_pixels'
        cmd=[sys.executable,str(collector_source/'collect_revision.py'),'--role','intervention','--round',str(r),
             '--workers','8','--out',str(folder),'--target',str(target)]
        with (root/f'acquisition_{r:02d}.log').open('w') as f:
            p=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
        assert p.returncode==0,('PHYSICAL_GATE_BLOCKED',str(folder),p.returncode)
        assert json.loads((folder/'result.json').read_text())['status']=='PASS'
        data=np.load(folder/'pixels.npz')
        seeds=set(data['seeds'].tolist());assert not seeds & old_seeds
        old_seeds.update(seeds)
        images=data['images'];encoded=[]
        with torch.no_grad():
            for i in range(0,len(images),32):
                encoded.append(model.encode(torch.from_numpy(images[i:i+32].transpose(0,3,1,2).copy()).float()/255)[0].numpy())
        z=np.concatenate(encoded)
        base_read=cal.transform(z)*ck['std']+ck['mean'];read=base_read.copy()
        read[:,0]=piecewise(read[:,0],pcal['knots'],pcal['values'])
        x=(read>=.5).astype(np.int8);masks=np.zeros_like(x)
        for i,j in enumerate(data['targets']):
            if j>=0:masks[i,j]=1
        np.savez_compressed(folder/'pixel_readouts.npz',raw=z,base_calibrated=base_read,calibrated=read,binary=x,masks=masks)
        allx.append(x);allm.append(masks)
        before=pool;pool,cpd,_=base.fit_pool(np.concatenate(allx),np.concatenate(allm))
        save(root/f'state_{r:02d}.json',pool)
        record={'round':r,'target':target,'decision':decision,'rows':len(x),'pairs':len(x)//2,
                'observed_means_by_command':[x[v::2].mean(0).tolist() for v in [0,1]],
                'observed_paired_difference':(x[1::2]-x[0::2]).mean(0).tolist(),
                'revision':base.revision(before,pool),'map_edges':pool['map_edges'],'edge_support':pool['edge_support'],
                'acquisition_sha256':sha(folder/'pixels.npz'),'readout_sha256':sha(folder/'pixel_readouts.npz'),
                'representation_frozen':True}
        trace.append(record);save(root/'trace.json',trace)
        print(json.dumps({'stage':'REVISE','round':r,**record['revision']}),flush=True)
    # Parent artifacts are read-only inputs; verify they were not altered.
    assert all(sha(Path(p))==v for p,v in inherited.items())
    result={**old_result,'status':'FIXED_BUDGET_COMPLETE','rounds':budget,
            'total_new_images':sum(map(len,allx)),'intervention_images':8*budget,
            'parent_run':str(parent),'continuation_new_images':8*a.additional_rounds,
            'final_map_edges':pool['map_edges'],'rounds_with_withdrawals':[t['round'] for t in trace if t['revision']['withdrawn']],
            'seconds':time.time()-start,'parent_unchanged':True,
            'claim_scope':'Exploratory eight-round continuation; original algorithm frozen. No reference evaluation by learner.'}
    if repair:result['controller_repair']=manifest['controller_repair']
    save(root/'result.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--parent',required=True);p.add_argument('--out',required=True)
    p.add_argument('--additional-rounds',type=int,default=8)
    p.add_argument('--collector-source');p.add_argument('--repair-check')
    a=p.parse_args()
    try:main(a)
    except Exception:
        root=Path(a.out);root.mkdir(parents=True,exist_ok=True)
        (root/'exit.json').write_text(json.dumps({'status':'BLOCKED','error':traceback.format_exc()},indent=2));raise
    else:(Path(a.out)/'exit.json').write_text(json.dumps({'status':'COMPLETE','returncode':0}))
