"""P-only monotone calibration from fresh verified operation values, not graph GT."""
import argparse,hashlib,json,os,sys,time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2';os.environ['CUDA_VISIBLE_DEVICES']=''
import numpy as np
import torch
from collect_revision import collect,context
BASE=Path('private-runs/icd_fork_20261024/full_v11')
sys.path.insert(0,str(BASE/'source/original'))
from min_pixel_v1 import VAE
from acc_domino_mec_node_calibration import NodeCalibration

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def piecewise(x,knots,values):
    x=np.asarray(x,float);knots=np.asarray(knots,float);values=np.asarray(values,float)
    out=np.interp(x,knots,values)
    out=np.where(x<knots[0],values[0]+(x-knots[0])*(values[1]-values[0])/(knots[1]-knots[0]),out)
    out=np.where(x>knots[-1],values[-1]+(x-knots[-1])*(values[-1]-values[-2])/(knots[-1]-knots[-2]),out)
    return out

def main(a):
    out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.time();torch.set_num_threads(2)
    groups=[(None,None),(1,0),(1,1),(2,0)];jobs=[];splits=[]
    for group,(target,value) in enumerate(groups):
        for i in range(8):
            seed=86000000+group*10000+i;ctx,_=context(seed)
            for p_command in [0,1]:
                jobs.append({'seed':seed,'target':target,'value':value,'root_on':bool(p_command),'context':ctx});splits.append('train' if i<6 else 'validation')
    (out/'launch.json').write_text(json.dumps({'jobs':jobs,'splits':splits,'fitting_rule':'Training-only 5/95 percentiles for each verified P command map to -.025,.025,.975,1.025; strictly increasing piecewise-linear map, no clipping.','groups':groups,'readout_threshold':.5,'learning_seed_cohort':'Separate from diagnostic and subsequent fresh graph cohort.','no_graph_reference':True,'pid':os.getpid()},indent=2))
    images=[];oracle=[];failures={}
    with ProcessPoolExecutor(8) as pool:
        for i,(image,labels,audit) in enumerate(pool.map(collect,jobs)):
            checks=dict(audit['physical_gate']);checks['verified_P_command']=bool(int(labels[0])==int(jobs[i]['root_on']))
            if not all(checks.values()):failures[str(i)]=[k for k,v in checks.items() if not v]
            audit['P_calibration_gate']=checks;(out/f'row{i:04d}_audit.json').write_text(json.dumps(audit,separators=(',',':')))
            images.append(image);oracle.append(labels);print(json.dumps({'stage':'P_CAL_ACQUIRE','completed':i+1,'total':64,'pass':all(checks.values())}),flush=True)
    commands=np.array([int(j['root_on']) for j in jobs]);images=np.array(images);train=np.array([s=='train' for s in splits])
    np.savez_compressed(out/'pixels.npz',images=images,p_commands=commands,train=train,seeds=np.array([j['seed'] for j in jobs]))
    np.savez_compressed(out/'oracle_measurements.npz',labels=np.array(oracle))
    if failures:
        (out/'result.json').write_text(json.dumps({'status':'PHYSICS_BLOCKED','failures':failures},indent=2));return 2
    cp=BASE/'visual/encoder_decoder.pt';ck=torch.load(cp,map_location='cpu',weights_only=False);model=VAE(7,H=128,W=128,C=3);model.load_state_dict(ck['state_dict']);model.eval()
    ca=BASE/'graph/calibration.json';cal=NodeCalibration(json.loads(ca.read_text()),sha(cp),ck['roles'])
    zs=[]
    with torch.no_grad():
        for i in range(0,len(images),32):zs.append(model.encode(torch.from_numpy(images[i:i+32].transpose(0,3,1,2).copy()).float()/255)[0].numpy())
    raw=np.concatenate(zs);old=cal.transform(raw)*ck['std']+ck['mean'];signal=old[:,0]
    lo=np.quantile(signal[train&(commands==0)],[.05,.95]);hi=np.quantile(signal[train&(commands==1)],[.05,.95]);knots=np.r_[lo,hi];values=np.array([-.025,.025,.975,1.025])
    if not np.all(np.diff(knots)>1e-8):
        (out/'result.json').write_text(json.dumps({'status':'CALIBRATION_ORDER_BLOCKED','knots':knots.tolist()},indent=2));return 2
    prediction=piecewise(signal,knots,values);validation=~train
    checks={'validation_binary_accuracy':float(np.mean((prediction[validation]>=.5)==commands[validation])),
            'old_validation_binary_accuracy':float(np.mean((signal[validation]>=.5)==commands[validation])),
            'validation_mse':float(np.mean((prediction[validation]-commands[validation])**2)),
            'old_validation_mse':float(np.mean((signal[validation]-commands[validation])**2)),
            'inverse_max_error':float(np.max(np.abs(piecewise(prediction,values,knots)-signal)))}
    passed=checks['validation_binary_accuracy']>=.95 and checks['inverse_max_error']<1e-8
    payload={'status':'PASS_FROZEN_P_READOUT' if passed else 'READOUT_VALIDATION_BLOCKED','kind':'positive_piecewise_linear_P_only',
             'input':'P coordinate AFTER the original affine ACC and denormalization','knots':knots.tolist(),'values':values.tolist(),
             'checkpoint_sha256':sha(cp),'base_calibration_sha256':sha(ca),'training_images':int(train.sum()),'validation_images':int(validation.sum()),
             'calibration_calls':64,'teacher':'physically verified commanded P value only; other-node simulator labels never used in fitting',
             'unchanged_coordinates':[1,2,3,4,5,6],'checks':checks,'pixels_sha256':sha(out/'pixels.npz'),'seconds':time.time()-start,'protocol_selected_before_data':True}
    (out/'P_recalibration.json').write_text(json.dumps(payload,indent=2));(out/'result.json').write_text(json.dumps(payload,indent=2))
    np.savez_compressed(out/'readouts.npz',raw=raw,old_P=signal,new_P=prediction,p_commands=commands,train=train)
    print(json.dumps(payload),flush=True);return 0 if passed else 2

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',required=True);raise SystemExit(main(p.parse_args()))
