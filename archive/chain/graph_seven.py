"""Frozen RGB -> command calibration -> original PC/direct-do/Meek.

No reference adjacency or simulator labels are loaded by graph discovery.
Every active batch is acquired after target selection; results are not chosen
from a pre-rendered pool. Controlled contexts are declared in FULL_DEMO_PLAN.
"""
import argparse,hashlib,json,os,subprocess,sys,time
from pathlib import Path
import numpy as np

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,obj):Path(p).write_text(json.dumps(obj,indent=2,default=lambda x:x.tolist() if hasattr(x,'tolist') else str(x)))
def gd(g):return {'directed':g.directed.tolist(),'skeleton':g.skeleton.tolist(),'undirected':[list(e) for e in sorted(g.undirected)],'valid':g.valid,'failure_reasons':list(g.failure_reasons),'source':g.source,'metadata':g.metadata}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--visual',required=True);ap.add_argument('--source',required=True);ap.add_argument('--out',required=True);ap.add_argument('--reuse-round1');a=ap.parse_args()
 os.environ['CUDA_VISIBLE_DEVICES']='';os.environ['OMP_NUM_THREADS']='2';os.environ['OPENBLAS_NUM_THREADS']='2'
 import torch
 torch.set_num_threads(2);sys.path.insert(0,a.source)
 from min_pixel_v1 import VAE
 from acc_domino_mec_node_calibration import fit_command_affine,NodeCalibration
 from acc_frozen_pipeline import FrozenCoordinates
 from acq_closed import learn_observational_cpdag,discover,select_target
 import acq_closed
 from variation_guard import install,PROTOCOL
 install(acq_closed)
 out=Path(a.out).resolve();out.mkdir(parents=True,exist_ok=False);data=Path(a.data).resolve();visual=Path(a.visual).resolve();start=time.time()
 assert json.loads((visual/'result.json').read_text())['status']=='PASS'
 for role in ['observations','calibration']:assert json.loads((data/role/'result.json').read_text())['status']=='PASS'
 ck=torch.load(visual/'encoder_decoder.pt',map_location='cpu',weights_only=False);roles=ck['roles'];mean=ck['mean'];std=ck['std'];model=VAE(7,H=128,W=128,C=3);model.load_state_dict(ck['state_dict']);model.eval()
 def enc(images):
  zs=[]
  with torch.no_grad():
   for i in range(0,len(images),32):zs.append(model.encode(torch.from_numpy(images[i:i+32].transpose(0,3,1,2).copy()).float()/255)[0].numpy())
  return np.concatenate(zs)
 manifest={'checkpoint_sha256':sha(visual/'encoder_decoder.pt'),'roles':roles,'policy':'coverage','cap_per_target':1,'images_per_round':64,'graph_thresholds':{'alpha':.01,'min_eff_skeleton':.03,'min_eff':.03,'do_thr':.12,'do_margin':0.},'binary_threshold':.5,'true_graph_input':False,'graph_inputs':['RGB images','intervention targets and commands'],'source_sha256':{'runner':sha(__file__),'backend':sha(Path(a.source)/'acq_closed.py'),'calibration':sha(Path(a.source)/'acc_domino_mec_node_calibration.py')},'data_sha256':{r:sha(data/r/'pixels.npz') for r in ['observations','calibration']},'pid':os.getpid()};save(out/'launch.json',manifest)
 manifest['orientation_guard']=PROTOCOL;manifest['source_sha256']['variation_guard']=sha(Path(__file__).with_name('variation_guard.py'));save(out/'launch.json',manifest)
 ca=np.load(data/'calibration/pixels.npz');zc=enc(ca['images']);groups=[zc[ca['targets']==j] for j in range(7)];commands=[ca['commands'][ca['targets']==j] for j in range(7)]
 payload=fit_command_affine(groups,commands,mean,std,manifest['checkpoint_sha256'],roles);save(out/'calibration.json',payload);cal=NodeCalibration(payload,manifest['checkpoint_sha256'],roles)
 # Separate held-out readout validation, before graph estimation/selection.
 te=np.load(data/'readout_test/pixels.npz');zt=enc(te['images']);cal_t=cal.transform(zt);read=cal_t*std+mean;correct=((read>=.5)==te['labels']);acc=correct.mean(0);mse=float(np.mean((cal_t-(te['labels']-mean)/std)**2));strata={str(int(j)):correct[te['targets']==j].mean(0).tolist() for j in np.unique(te['targets'])};checks={'binary_accuracy_per_node':acc.tolist(),'accuracy_by_intervention_target':strata,'normalized_mse':mse,'rows':len(read),'pass':bool(min(acc)>=.95 and mse<=.05 and min(min(v) for v in strata.values())>=.875)};save(out/'calibrated_readout_gate.json',checks)
 if not checks['pass']:save(out/'result.json',{'status':'CALIBRATED_READOUT_BLOCKED','checks':checks});return 2
 ob=np.load(data/'observations/pixels.npz');zo=enc(ob['images']);bo=(cal.transform(zo)*std+mean>=.5).astype(float);assert np.all(bo.std(0)>.02),'Constant observational readout'
 coord=FrozenCoordinates(bo);g=learn_observational_cpdag(coord.encode(bo),7,alpha=.01,max_order=None,min_eff=.03,source='PC_from_frozen_RGB_ACC_binary_readouts');save(out/'initial_graph.json',gd(g));np.savez_compressed(out/'observational_readouts.npz',raw=zo,binary=bo,seeds=ob['seeds'])
 z_all=[bo];m_all=[np.zeros_like(bo)];n_done=np.zeros(7,int);rng=np.random.default_rng(42);trace=[]
 if not g.valid:save(out/'result.json',{'status':'INITIAL_GRAPH_INVALID','graph':gd(g)});return 2
 for r in range(1,8):
  if not g.undirected:break
  j,_=select_target('coverage',g.undirected,g.directed,None,n_done,1,7,rng,0);j=int(j)
  decision={'round':r,'target':j,'graph_before':gd(g),'coverage_score':[sum(i in e for e in g.undirected) for i in range(7)],'previous_counts':n_done.tolist()};save(out/f'round{r}_decision.json',decision)
  batch=out/f'round{r}_pixels';cmd=[sys.executable,str(Path(__file__).with_name('collect_seven.py')),'--role','intervention','--target',str(j),'--round',str(r),'--out',str(batch),'--workers','8']
  print(json.dumps({'stage':'ACQUIRE','round':r,'target':j}),flush=True)
  reused=None
  if r==1 and a.reuse_round1:
   from collect_seven import jobs
   previous=Path(a.reuse_round1).resolve();old=json.loads((previous/'launch.json').read_text());assert old['jobs']==jobs('intervention',j,r),'Reused batch does not match the freshly selected target/context'
   assert old['controller_sha256']==sha(Path(__file__).with_name('robot_seven.py'))
   assert old['source_sha256']==sha(Path(__file__).with_name('collect_seven.py'))
   assert json.loads((previous/'result.json').read_text())['status']=='PASS'
   batch.symlink_to(previous,target_is_directory=True);reused=str(previous);save(out/f'round{r}_reused_acquisition.json',{'source':reused,'reason':'Same initial PC graph selects the same target; retain the already acquired unfiltered batch. Only admissibility of constant-endpoint evidence changed.'})
  else:
   with (out/f'round{r}_acquisition.log').open('w') as f:proc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'OMP_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1'})
   if proc.returncode:save(out/'result.json',{'status':'ACQUISITION_BLOCKED','round':r,'returncode':proc.returncode});return 2
  d=np.load(batch/'pixels.npz');z=enc(d['images']);b=(cal.transform(z)*std+mean>=.5).astype(float);m=np.zeros_like(b);m[:,j]=1;z_all.append(b);m_all.append(m);n_done[j]+=1
  np.savez_compressed(out/f'round{r}_readouts.npz',raw=z,binary=b,seeds=d['seeds'],commands=d['commands'])
  nextg,ev=discover(coord.encode(np.concatenate(z_all)),np.concatenate(m_all),g.skeleton,7,alpha=.01,min_eff=.03,do_thr=.12,do_margin=0.,initial_graph=g,legacy_skeleton=False,return_graph=True)
  item={'round':r,'target':j,'rows':len(b),'contexts':len(b)//2,'graph':gd(nextg),'evidence':ev,'provenance':[{'edge':list(k),'source':v} for k,v in nextg.provenance.items()],'batch_pixels_sha256':sha(batch/'pixels.npz'),'all_action_gates_pass':True,'representative_pair':{'seed':int(d['seeds'][0]),'context':{'lean':[False]*4,'bell_noise':False,'lamp_noise':False},'root_on':True}};trace.append(item);save(out/'graph_trace.json',trace);g=nextg
  item['acquisition_reused_from']=reused;save(out/'graph_trace.json',trace)
  print(json.dumps({'stage':'GRAPH_UPDATED','round':r,'target':j,'undirected':len(g.undirected),'valid':g.valid}),flush=True)
  if not g.valid:break
 result={'status':'DIRECTED_GRAPH_COMPLETE' if g.valid and not g.undirected else 'PARTIAL_GRAPH','roles':roles,'graph':gd(g),'rounds':len(trace),'intervention_images':int(sum(v['rows'] for v in trace)),'calibration_images':len(zc),'total_operational_images':len(zc)+int(sum(v['rows'] for v in trace)),'calibration_sha256':sha(out/'calibration.json'),'checkpoint_sha256':manifest['checkpoint_sha256'],'true_graph_given':False,'seconds':time.time()-start,'scope':'One frozen seven-node physical demonstrator; correctness is checked by a separate evaluator, not this learner'};save(out/'result.json',result);print(json.dumps(result),flush=True);return 0

if __name__=='__main__':raise SystemExit(main())
