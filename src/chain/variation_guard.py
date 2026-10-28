"""Do not interpret undefined correlation as evidence of causal absence.

This adapter leaves PC, thresholds, positive-effect scoring and Meek unchanged.
It withholds an endpoint's do-score when either measured variable is constant
in that batch, so another intervention must supply the missing evidence.
"""
import numpy as np

PROTOCOL='direct_do_endpoint_variation_guard_v1'

def install(backend):
 legacy=backend.edge_orientation_evidence
 def guarded(Z,M,i,j,engine='direct_do',config=None,do_thr=.12,do_margin=0.,dep='linear'):
  ev=legacy(Z,M,i,j,engine=engine,config=config,do_thr=do_thr,do_margin=do_margin,dep=dep)
  if engine!='direct_do':return ev
  support={};scores=[];single=np.isclose(np.asarray(M).sum(1),1.)
  for node,key in [(i,'score_i_to_j'),(j,'score_j_to_i')]:
   rows=np.flatnonzero((np.asarray(M)[:,node]>.5)&single);std=np.std(np.asarray(Z)[rows][:,[i,j]],axis=0) if len(rows) else np.zeros(2);valid=bool(len(rows)>30 and np.all(std>1e-8))
   support[str(node)]={'rows':len(rows),'endpoint_std':std.tolist(),'admissible':valid}
   score=ev[key] if valid else None;ev[key]=score;scores.append(score)
  x,y=scores;thr=ev['config']['do_thr'];margin=ev['config']['do_margin'];r=0
  if x is not None and y is not None:
   if x>thr and y<thr and x-y>=margin:r=1
   elif y>thr and x<thr and y-x>=margin:r=-1
  elif x is not None and abs(x-thr)>=margin:r=1 if x>thr else -1
  elif y is not None and abs(y-thr)>=margin:r=-1 if y>thr else 1
  ev['legacy_orientation_before_variation_guard']=ev['orientation'];ev['orientation']=r;ev['guard_protocol']=PROTOCOL;ev['endpoint_variation_support']=support;ev['orientation_source']='marginal_single_target_dependence_with_endpoint_variation_gate';ev['abstain_reason']=None if r else ('insufficient_endpoint_variation' if x is None and y is None else 'threshold_or_margin_ambiguous')
  return ev
 backend.edge_orientation_evidence=guarded
