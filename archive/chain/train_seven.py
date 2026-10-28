"""Original anchored RGB VAE, final-epoch checkpoint, independent readout gate."""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True);ap.add_argument('--source',required=True);ap.add_argument('--gpu',required=True);a=ap.parse_args()
 gpu_rows=subprocess.check_output(['nvidia-smi','--query-gpu=uuid,memory.used','--format=csv,noheader,nounits'],text=True).splitlines()
 used={r.split(',')[0].strip():int(r.split(',')[1]) for r in gpu_rows};assert a.gpu in used and used[a.gpu]<500,used
 os.environ['CUDA_VISIBLE_DEVICES']=a.gpu;os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
 import torch
 from scipy.stats import rankdata
 sys.path.insert(0,a.source)
 from min_pixel_v2 import train_vae_anchored
 torch.set_num_threads(2);torch.manual_seed(42);np.random.seed(42);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 data=Path(a.data);out=Path(a.out);out.mkdir(parents=True,exist_ok=False);start=time.time();files={role:data/role/'pixels.npz' for role in ['train','validation','readout_test']}
 for role in files:assert json.loads((data/role/'result.json').read_text())['status']=='PASS'
 ds={k:np.load(v) for k,v in files.items()};train=ds['train'];mean=train['labels'].mean(0);std=train['labels'].std(0);assert np.all(std>.05)
 manifest={'seed':42,'epochs':100,'batch_size':32,'beta':144.,'lr':.001,'gpu_uuid':a.gpu,'pid':os.getpid(),'data_sha256':{k:sha(v) for k,v in files.items()},'source_sha256':{'runner':sha(__file__),'train_vae_anchored':sha(Path(a.source)/'min_pixel_v2.py'),'VAE':sha(Path(a.source)/'min_pixel_v1.py')},'selection':'final_epoch_only','roles':['P','F1','F2','F3','F4','B','L'],'true_graph_input':False}
 (out/'launch.json').write_text(json.dumps(manifest,indent=2));print(json.dumps({'status':'TRAINING','manifest':manifest}),flush=True)
 x=train['images'].transpose(0,3,1,2).astype(np.float32)/255;y=(train['labels']-mean)/std
 model,history=train_vae_anchored(x,y,7,torch.device('cuda'),epochs=100,bs=32,lr=.001,beta=144.,anchor=True,return_history=True)
 model.eval();torch.save({'state_dict':model.cpu().state_dict(),'mean':mean,'std':std,'roles':manifest['roles'],'config':manifest},out/'encoder_decoder.pt');model.cuda()
 checks={}
 with torch.no_grad():
  for role in ['validation','readout_test']:
   d=ds[role];zs=[]
   for i in range(0,len(d['images']),32):
    xb=torch.from_numpy(d['images'][i:i+32].transpose(0,3,1,2).copy()).float().cuda()/255;zs.append(model.encode(xb)[0].cpu().numpy())
   z=np.concatenate(zs);read=z*std+mean;labels=d['labels'];correct=((read>=.5)==labels);acc=correct.mean(0);mse=float(np.mean((z-(labels-mean)/std)**2));strata={str(int(j)):correct[d['targets']==j].mean(0).tolist() for j in np.unique(d['targets'])};strata_min=min(min(v) for v in strata.values());auc=[]
   for j in range(7):
    positive=labels[:,j]>.5;np_=int(positive.sum());nn_=len(positive)-np_;assert np_ and nn_;ranks=rankdata(z[:,j]);auc.append(float((ranks[positive].sum()-np_*(np_+1)/2)/(np_*nn_)))
   checks[role]={'binary_accuracy_per_node':acc.tolist(),'accuracy_by_intervention_target':strata,'min_stratum_accuracy':strata_min,'normalized_mse':mse,'auc_per_node':auc,'rows':len(labels),'gate':'visual separation before command calibration; absolute-unit gates apply after calibration','pass':bool(min(auc)>=.95)}
   np.savez_compressed(out/f'{role}_predictions.npz',z=z,readouts=read,labels=labels,seeds=d['seeds'])
 (out/'history.json').write_text(json.dumps(history));result={'status':'PASS' if all(v['pass'] for v in checks.values()) else 'READOUT_BLOCKED','checks':checks,'checkpoint_sha256':sha(out/'encoder_decoder.pt'),'seconds':time.time()-start,'peak_gpu_allocated_bytes':torch.cuda.max_memory_allocated()};(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
 raise SystemExit(0 if result['status']=='PASS' else 2)

if __name__=='__main__':main()
