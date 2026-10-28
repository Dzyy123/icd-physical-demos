"""Seven-node contact-dynamics extension. No ground-truth graph or SCM rollout.

Pusher and probe are position-controlled prismatic actuators. Bell is a free
striker carriage that physically hits a gong. A contact-sensor relay drives L.
Labels are evaluator/anchor measurements, never a dynamics input.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','1')
from pathlib import Path
import argparse,json,math
import numpy as np
from PIL import Image
import pybullet as p

COLORS=[[.28,.34,.64,1],[.40,.32,.60,1],[.56,.31,.49,1],[.65,.33,.39,1]]
class PhysicalSeven:
 def __init__(self):
  self.cid=p.connect(p.DIRECT)
  p.setGravity(0,0,-9.81);p.setTimeStep(1/480)
  p.setPhysicsEngineParameter(numSolverIterations=180,deterministicOverlappingPairs=1,numSubSteps=4)
  self.top=.30;self.dt=1/480
  self.box([2,2,.01],[.94,.95,.96,1],[.4,0,-.02])
  self.table=self.box([.57,.29,.02],[.73,.70,.62,1],[.42,0,.28])
  for x in [-.08,.90]:
   for y in [-.22,.22]:self.box([.022,.022,.14],[.43,.46,.49,1],[x,y,.14])
  # A stationary rail and a translating contact plate: the visible pusher.
  self.pusher_rail=self.box([.065,.022,.014],[.35,.39,.43,1],[.075,0,.314])
  c=p.createCollisionShape(p.GEOM_BOX,halfExtents=[.01,.037,.034]);v=p.createVisualShape(p.GEOM_BOX,halfExtents=[.01,.037,.034],rgbaColor=[.91,.61,.20,1])
  self.pusher=p.createMultiBody(baseMass=0,basePosition=[.065,0,.414],linkMasses=[.06],linkCollisionShapeIndices=[c],linkVisualShapeIndices=[v],linkPositions=[[0,0,0]],linkOrientations=[[0,0,0,1]],linkInertialFramePositions=[[0,0,0]],linkInertialFrameOrientations=[[0,0,0,1]],linkParentIndices=[0],linkJointTypes=[p.JOINT_PRISMATIC],linkJointAxis=[[1,0,0]])
  # Bronze striker: plate + shaft + hammer head, all one rigid link.
  positions=[[0,0,0],[.025,0,.06],[.05,0,.06]]
  bc=p.createCollisionShapeArray(shapeTypes=[p.GEOM_BOX,p.GEOM_BOX,p.GEOM_SPHERE],halfExtents=[[.006,.043,.04],[.025,.005,.005],[0,0,0]],radii=[0,0,.012],collisionFramePositions=positions)
  bv=p.createVisualShapeArray(shapeTypes=[p.GEOM_BOX,p.GEOM_BOX,p.GEOM_SPHERE],halfExtents=[[.006,.043,.04],[.025,.005,.005],[0,0,0]],radii=[0,0,.012],visualFramePositions=positions,rgbaColors=[[.50,.37,.20,1],[.45,.33,.18,1],[.58,.43,.20,1]])
  self.bell=p.createMultiBody(baseMass=0,basePosition=[.64,0,.35],linkMasses=[.018],linkCollisionShapeIndices=[bc],linkVisualShapeIndices=[bv],linkPositions=[[0,0,0]],linkOrientations=[[0,0,0,1]],linkInertialFramePositions=[[0,0,0]],linkInertialFrameOrientations=[[0,0,0,1]],linkParentIndices=[0],linkJointTypes=[p.JOINT_PRISMATIC],linkJointAxis=[[1,0,0]])
  self.box([.05,.025,.004],[.38,.41,.43,1],[.67,0,.304])
  gc=p.createCollisionShape(p.GEOM_CYLINDER,radius=.055,height=.012,collisionFrameOrientation=p.getQuaternionFromEuler([0,np.pi/2,0]));gv=p.createVisualShape(p.GEOM_CYLINDER,radius=.055,length=.012,rgbaColor=[.83,.65,.18,1],visualFrameOrientation=p.getQuaternionFromEuler([0,np.pi/2,0]))
  self.gong=p.createMultiBody(0,gc,gv,basePosition=[.744,0,.41]);self.box([.008,.008,.05],[.38,.40,.42,1],[.752,0,.35])
  self.box([.025,.025,.012],[.30,.33,.35,1],[.86,.08,.312]);self.box([.008,.008,.08],[.38,.40,.42,1],[.86,.08,.40])
  lv=p.createVisualShape(p.GEOM_SPHERE,radius=.030,rgbaColor=[.08,.20,.11,1]);self.lamp=p.createMultiBody(0,-1,lv,basePosition=[.86,.08,.50])
  # Visible fixed signal wire, not a kinematic connection to the dominoes.
  self.box([.059,.002,.002],[.18,.24,.20,1],[.803,.08,.317])
  # Cartesian contact probe. It parks identically in every settled image.
  self.probe_parts=[self.box([.015,.015,.19],[.57,.61,.65,1],[-.07,-.21,.49]),self.box([.36,.012,.012],[.57,.61,.65,1],[.29,-.21,.68])]
  pv=p.createVisualShape(p.GEOM_BOX,halfExtents=[.022,.018,.018],rgbaColor=[.86,.88,.90,1])
  tipc=p.createCollisionShape(p.GEOM_BOX,halfExtents=[.008,.031,.012]);tipv=p.createVisualShape(p.GEOM_BOX,halfExtents=[.008,.031,.012],rgbaColor=[.87,.90,.93,1])
  self.probe=p.createMultiBody(baseMass=0,basePosition=[0,-.21,.64],linkMasses=[.03,.02,.02],linkCollisionShapeIndices=[-1,-1,tipc],linkVisualShapeIndices=[pv,pv,tipv],linkPositions=[[0,0,0],[0,0,0],[0,0,0]],linkOrientations=[[0,0,0,1]]*3,linkInertialFramePositions=[[0,0,0]]*3,linkInertialFrameOrientations=[[0,0,0,1]]*3,linkParentIndices=[0,1,2],linkJointTypes=[p.JOINT_PRISMATIC]*3,linkJointAxis=[[1,0,0],[0,1,0],[0,0,1]])
  for b in [self.pusher,self.bell,self.probe]:
   for j in range(p.getNumJoints(b)):p.changeDynamics(b,j,linearDamping=.1,angularDamping=.1,restitution=0)
  self.view=p.computeViewMatrix([.93,-1.32,1.03],[.43,.015,.37],[0,0,1]);self.proj=p.computeProjectionMatrixFOV(47,1,.05,5)
  self.wide=p.computeViewMatrix([.93,-1.32,1.03],[.43,.015,.37],[0,0,1]);self.wproj=p.computeProjectionMatrixFOV(39,1040/585,.05,5)
 def box(self,half,color,pos,mass=0):
  c=p.createCollisionShape(p.GEOM_BOX,halfExtents=half);v=p.createVisualShape(p.GEOM_BOX,halfExtents=half,rgbaColor=color)
  return p.createMultiBody(mass,c,v,basePosition=pos)
 def close(self):p.disconnect(self.cid)
 def image(self,wide=False):
  w,h=(1040,585) if wide else (128,128)
  im=p.getCameraImage(w,h,viewMatrix=self.wide if wide else self.view,projectionMatrix=self.wproj if wide else self.proj,renderer=p.ER_TINY_RENDERER,shadow=1,lightDirection=[-1,-1,3])
  return np.asarray(im[2],dtype=np.uint8).reshape(h,w,4)[:,:,:3].copy()
 def sample(self,seed,target=None,value=None,clean=False,root_on=None,capture=None):
  assert target is None or target in range(7)
  rng=np.random.default_rng(seed);dx=rng.uniform(-.002,.002,4);dy=float(rng.uniform(-.003,.003))
  lean=rng.random(4)<.16;drive=bool(rng.random()<.25);bell_noise=bool(rng.random()<.15);lamp_test=bool(rng.random()<.12)
  if clean:lean[:]=False;drive=False;bell_noise=False;lamp_test=False
  if root_on is not None:drive=bool(root_on)
  if target==0:drive=bool(value)
  if target==5:bell_noise=bool(value)
  if target in [1,2,3,4]:lean[target-1]=False
  bodies=[];initial=[]
  for i,x in enumerate([.18,.29,.40,.51]):
   angle=math.radians(20 if lean[i] else 0);pivot=x+dx[i]+.012
   pos=[pivot-.012*math.cos(angle)+.09*math.sin(angle),dy,self.top+.012*math.sin(angle)+.09*math.cos(angle)+.0001]
   b=self.box([.012,.05,.09],COLORS[i],pos);p.resetBasePositionAndOrientation(b,pos,p.getQuaternionFromEuler([0,angle,0]));p.changeDynamics(b,-1,lateralFriction=.6,restitution=0,linearDamping=.04,angularDamping=.04)
   bodies.append(b);initial.append(pos)
  p.setJointMotorControl2(self.pusher,0,p.POSITION_CONTROL,targetPosition=0,force=3)
  p.setJointMotorControl2(self.bell,0,p.POSITION_CONTROL,targetPosition=0,force=3)
  for i,q in enumerate([.05,0,0]):p.resetJointState(self.probe,i,q)
  p.setJointMotorControlArray(self.probe,[0,1,2],p.POSITION_CONTROL,targetPositions=[.05,0,0],forces=[10,10,10])
  folder=Path(capture) if capture else None
  if folder:folder.mkdir(parents=True,exist_ok=False)
  frames=[];contacts={};strike=False;light=False;stepno=0;phase='initial';worst=0.;wrong=set();first_strike=None
  def snapshot():
   Image.fromarray(self.image(True)).save(folder/f'{len(frames):04d}.png');frames.append({'step':stepno,'phase':phase,'bell_contact':strike,'lamp':light,'pusher_q':p.getJointState(self.pusher,0)[0],'striker_q':p.getJointState(self.bell,0)[0]})
  def step(n):
   nonlocal strike,light,stepno,worst,first_strike
   for _ in range(n):
    p.stepSimulation();stepno+=1
    for c in p.getContactPoints():
     worst=min(worst,float(c[8]))
     if c[9]>.001:
      pair=tuple(sorted([int(c[1]),int(c[2])]))
      contacts[pair]=contacts.get(pair,0)+1
      if set(pair)=={self.bell,self.gong} and not strike:strike=True;first_strike=stepno
      if self.probe in pair:
       other=c[2] if c[1]==self.probe else c[1]
       if other in bodies and (target not in [1,2,3,4] or other!=bodies[target-1]):wrong.add(bodies.index(other)+1)
    now=(bool(value) if target==6 else (strike or lamp_test))
    if now!=light:
     light=now;p.changeVisualShape(self.lamp,-1,rgbaColor=[.15,.85,.30,1] if light else [.08,.20,.11,1])
    if folder and stepno%24==0:snapshot()
  def probe_move(x,y,z,seconds):
   start=np.array([p.getJointState(self.probe,j)[0] for j in range(3)]);end=np.array([x,y+.21,z-.64]);n=max(1,int(seconds*60))
   for k in range(n):
    u=(k+1)/n;s=u*u*(3-2*u);q=start*(1-s)+end*s
    p.setJointMotorControlArray(self.probe,[0,1,2],p.POSITION_CONTROL,targetPositions=q.tolist(),forces=[6,6,6],positionGains=[.15]*3,velocityGains=[1]*3);step(8)
  if folder:snapshot()
  if target in [1,2,3,4] and value==1:
   x=initial[target-1][0];phase='probe_approach';probe_move(x-.035,dy,.64,.6)
   phase='probe_descend';probe_move(x-.035,dy,.43,.5)
  # Release initial staging; after this point poses are never reassigned.
  for i,b in enumerate(bodies):p.changeDynamics(b,-1,mass=0 if target==i+1 and value==0 else .08)
  release=stepno
  if target in [1,2,3,4] and value==0:
   x=initial[target-1][0]
   # Visible fixed brace denotes the immobilizing fixture.
   self.box([.016,.008,.018],[.44,.49,.53,1],[x,dy-.065,self.top+.018])
  p.setJointMotorControl2(self.pusher,0,p.POSITION_CONTROL,targetPosition=.125 if drive else 0,force=.7,maxVelocity=.16,positionGain=.15)
  if target==5 and value==0:p.setJointMotorControl2(self.bell,0,p.POSITION_CONTROL,targetPosition=0,force=20)
  elif bell_noise:p.setJointMotorControl2(self.bell,0,p.POSITION_CONTROL,targetPosition=.05,force=.35,maxVelocity=.12)
  else:p.setJointMotorControl2(self.bell,0,p.VELOCITY_CONTROL,targetVelocity=0,force=.003)
  if target in [1,2,3,4] and value==1:
   phase='probe_push';probe_move(x+.04,dy,.43,.3)
   phase='probe_lift';probe_move(x-.035,dy,.64,.4)
   phase='probe_return';probe_move(.05,-.21,.64,.7)
  phase='settle';stable=0
  for _ in range(2400):
   step(1);v=[p.getBaseVelocity(b) for b in bodies]
   quiet=max(np.linalg.norm(a) for a,b in v)<.003 and max(np.linalg.norm(b) for a,b in v)<.025 and abs(p.getJointState(self.bell,0)[1])<.004
   stable=stable+1 if quiet else 0
   if stepno-release>960 and stable>=240:break
  angles=[float(math.degrees(math.acos(np.clip(np.array(p.getMatrixFromQuaternion(p.getBasePositionAndOrientation(b)[1])).reshape(3,3)[2,2],-1,1)))) for b in bodies]
  q=p.getJointState(self.pusher,0)[0];bq=p.getJointState(self.bell,0)[0]
  labels=np.array([q>=.04]+[a>=10 for a in angles]+[strike,light],dtype=np.float32)
  named={self.pusher:'P',self.bell:'B',self.gong:'gong',self.probe:'probe',self.table:'table',**{b:f'F{i+1}' for i,b in enumerate(bodies)}}
  contact_rows=[{'a':named.get(a,str(a)),'b':named.get(b,str(b)),'steps':n} for (a,b),n in contacts.items() if a in named and b in named]
  audit={'seed':seed,'target':target,'value':value,'labels':labels.tolist(),'angles':angles,'pusher_q':q,'striker_q':bq,'first_strike_step':first_strike,'min_contact_distance':worst,'settled':stable>=240,'target_realized':bool(target is None or labels[target]==value),'probe_non_target_contacts':sorted(wrong),'bell_contact_observable':(not strike and bq<.015) or (strike and bq>.02),'contacts':contact_rows,'initial_tilts':lean.tolist(),'drive':drive,'bell_excitation':bell_noise,'lamp_test_switch':lamp_test,'release_step':release,'steps':stepno,'frame_phases':frames,'post_release_pose_resets':0,'lamp_mechanism':'Actual gong contact sensor OR independent test switch; direct L interventions override the relay. Not acoustic or field-level circuit simulation.'}
  if folder:Image.fromarray(self.image()).save(folder/'model_input.png');(folder/'audit.json').write_text(json.dumps(audit,indent=2))
  return self.image(),labels,audit

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--target',type=int);ap.add_argument('--value',type=int,default=1);ap.add_argument('--root-on',type=int);ap.add_argument('--seed',type=int,default=80000000);ap.add_argument('--capture',action='store_true');a=ap.parse_args()
 env=PhysicalSeven()
 try:
  im,y,audit=env.sample(a.seed,a.target,a.value,clean=True,root_on=a.root_on,capture=a.out if a.capture else None)
  if not a.capture:
   out=Path(a.out);out.mkdir(parents=True,exist_ok=False);Image.fromarray(im).save(out/'model_input.png');Image.fromarray(env.image(True)).save(out/'scene.png');(out/'audit.json').write_text(json.dumps(audit,indent=2))
  print(json.dumps({k:v for k,v in audit.items() if k not in ['contacts','frame_phases']}),flush=True)
 finally:env.close()
