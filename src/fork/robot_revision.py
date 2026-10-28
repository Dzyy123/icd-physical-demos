"""Robot-contact pilot, separate from the Cartesian mechanics pilot.

Measured rigid-body motion drives P/F/B; a measured three-position switch and
gong contact drive the explicitly modeled lamp relay. No learned graph here.
"""
import argparse, hashlib, json, math
from collections import deque
from pathlib import Path
import numpy as np
from PIL import Image
import pybullet as p
import pybullet_data
from physical7 import PhysicalSeven, COLORS

class RobotFork(PhysicalSeven):
 def __init__(self):
  super().__init__()
  # Observe every solver substep. Retain the validated 1/3840 s contact step;
  # penetration; control and capture remain 480/20 Hz.
  self.physics_substeps=8
  p.setTimeStep(1/3840);p.setPhysicsEngineParameter(numSubSteps=1)
  for b in [self.probe]+self.probe_parts:p.removeBody(b)
  # Contact handles extend toward the viewer, away from the falling lane.
  p.removeBody(self.pusher)
  p.removeBody(self.pusher_rail)
  self.box([.10,.020,.008],[.35,.39,.43,1],[.12,-.15,.308])
  # Physical end stops and a U-shaped carriage, not an unconstrained visual
  # rail. The moving carriage stays on the rail after the gripper withdraws.
  self.pusher_stops=[self.box([.007,.030,.020],[.29,.33,.37,1],[x,-.15,.320]) for x in [.023,.217]]
  self.pusher_bounds=(-.017,.127)
  orange=[.91,.61,.20,1];steel=[.48,.52,.56,1]
  self.pusher=self.slider([.065,0,.414],.06,
   [[.01,.037,.034],[.006,.075,.006],[.012,.012,.025],[.008,.012,.043],[.018,.026,.006],[.018,.004,.008],[.018,.004,.008]],
   [[0,0,0],[0,-.075,0],[0,-.15,0],[0,-.15,-.043],[0,-.15,-.092],[0,-.174,-.106],[0,-.126,-.106]],
   [orange]*4+[steel]*3)
  # Attach an off-lane handle to the same striker rigid body (not a driver).
  p.removeBody(self.bell)
  positions=[[0,0,0],[.025,0,.06],[.05,0,.06],[0,.075,.04],[0,.15,.04]]
  half=[[.006,.043,.04],[.025,.005,.005],[0,0,0],[.006,.075,.006],[.012,.012,.025]]
  types=[p.GEOM_BOX,p.GEOM_BOX,p.GEOM_SPHERE,p.GEOM_BOX,p.GEOM_BOX]
  c=p.createCollisionShapeArray(shapeTypes=types,halfExtents=half,radii=[0,0,.012,0,0],collisionFramePositions=positions)
  v=p.createVisualShapeArray(shapeTypes=types,halfExtents=half,radii=[0,0,.012,0,0],visualFramePositions=positions,rgbaColors=[[.58,.43,.20,1]]*5)
  self.bell=self.link([.64,0,.35],.018,c,v)
  # OFF / AUTO / ON switch: actual slider q, not the requested target, sets L.
  self.box([.047,.030,.008],[.38,.41,.43,1],[.84,.17,.308])
  self.switch=self.slider([.84,.17,.348],.01,[[.009,.013,.028]],[[0,0,0]],[.17,.47,.41,1])
  self.switch_q0=0.;self.switch_hold=0.
  p.setAdditionalSearchPath(pybullet_data.getDataPath())
  self.arms={}
  for name,pos,yaw in [('primary',[.40,.43,0],-np.pi/2),('root',[-.40,-.20,0],0)]:
   b=p.loadURDF('franka_panda/panda.urdf',pos,p.getQuaternionFromEuler([0,0,yaw]),useFixedBase=True)
   q=np.array([0,-.785,0,-2.356,0,1.571,.785])
   for j,vq in enumerate(q):p.resetJointState(b,j,float(vq))
   for j in [9,10]:
    p.resetJointState(b,j,.008)
    p.changeDynamics(b,j,contactStiffness=300000,contactDamping=1000)
   self.arms[name]={'body':b,'home':q,'gap':.008,'orn':p.getQuaternionFromEuler([np.pi,0,np.pi/2])}
   self.arms[name]['grasp_orn']=p.getLinkState(b,11,computeForwardKinematics=True)[5]
   self.arms[name]['push_orn']=p.multiplyTransforms([0,0,0],p.getQuaternionFromEuler([0,0,np.pi/2]),[0,0,0],self.arms[name]['grasp_orn'])[1]
   self.motor(name,q)
  self.wide=p.computeViewMatrix([1.10,-1.65,1.10],[.36,.05,.42],[0,0,1])
  self.wproj=p.computeProjectionMatrixFOV(44,1040/585,.05,5)
  self.view=p.computeViewMatrix([.93,-1.32,1.03],[.43,.015,.37],[0,0,1])
  # A fork is driven by two independently compliant contact tips. Blocking
  # one tip compresses that tip rather than rigidly arresting the other lane.
  p.removeBody(self.pusher)
  halves=[[.006,.15,.006],[.012,.012,.025],[.008,.012,.043],[.018,.026,.006],[.018,.004,.008],[.018,.004,.008]]
  positions=[[0,0,0],[0,-.15,0],[0,-.15,-.043],[0,-.15,-.092],[0,-.174,-.106],[0,-.126,-.106]]
  # Sleeves behind the crossbar and visible telescoping shafts make the
  # compliant prismatic mechanism legible. Adjacent links model the guides.
  halves += [[.03,.008,.008],[.03,.008,.008]]
  positions += [[-.03,-.06,0],[-.03,.08,0]]
  cc=p.createCollisionShapeArray(shapeTypes=[p.GEOM_BOX]*len(halves),halfExtents=halves,collisionFramePositions=positions)
  cv=p.createVisualShapeArray(shapeTypes=[p.GEOM_BOX]*len(halves),halfExtents=halves,visualFramePositions=positions,rgbaColors=[orange]*3+[steel]*5)
  tc=p.createCollisionShape(p.GEOM_BOX,halfExtents=[.008,.025,.018])
  tv=p.createVisualShapeArray(shapeTypes=[p.GEOM_BOX,p.GEOM_BOX],halfExtents=[[.008,.025,.018],[.035,.004,.004]],visualFramePositions=[[0,0,0],[-.035,0,0]],rgbaColors=[orange,steel])
  self.pusher=p.createMultiBody(baseMass=0,basePosition=[.065,0,.414],
    linkMasses=[.06,.008,.008],linkCollisionShapeIndices=[cc,tc,tc],linkVisualShapeIndices=[cv,tv,tv],
    linkPositions=[[0,0,0],[.045,-.06,0],[.045,.08,0]],linkOrientations=[[0,0,0,1]]*3,
    linkInertialFramePositions=[[0,0,0]]*3,linkInertialFrameOrientations=[[0,0,0,1]]*3,
    linkParentIndices=[0,1,1],linkJointTypes=[p.JOINT_PRISMATIC]*3,linkJointAxis=[[1,0,0]]*3)
  for j in [1,2]:p.setJointMotorControl2(self.pusher,j,p.POSITION_CONTROL,targetPosition=0,force=.30,positionGain=.02)
  # Keep the bell and its physical contact sensor on the far lane.
  for body in [self.bell,self.gong]:
   pos,orn=p.getBasePositionAndOrientation(body);p.resetBasePositionAndOrientation(body,[pos[0]-.22,pos[1]+.08,pos[2]],orn)
  # Move the corresponding rail and gong stand too, so the bell is mounted
  # rather than visually floating above the table. Extend the visible wire.
  for k in range(p.getNumBodies()):
   body=p.getBodyUniqueId(k);pos,orn=p.getBasePositionAndOrientation(body)
   if any(np.allclose(pos,old,atol=1e-8) for old in [[.67,0,.304],[.752,0,.35]]):
    p.resetBasePositionAndOrientation(body,[pos[0]-.22,pos[1]+.08,pos[2]],orn)
   if np.allclose(pos,[.803,.08,.317],atol=1e-8):
    p.changeVisualShape(body,-1,rgbaColor=[0,0,0,0])
  self.box([.168,.002,.002],[.18,.24,.20,1],[.692,.08,.317])

 def link(self,pos,mass,c,v):
  return p.createMultiBody(baseMass=0,basePosition=pos,linkMasses=[mass],linkCollisionShapeIndices=[c],linkVisualShapeIndices=[v],linkPositions=[[0,0,0]],linkOrientations=[[0,0,0,1]],linkInertialFramePositions=[[0,0,0]],linkInertialFrameOrientations=[[0,0,0,1]],linkParentIndices=[0],linkJointTypes=[p.JOINT_PRISMATIC],linkJointAxis=[[1,0,0]])

 def slider(self,pos,mass,half,positions,color):
  c=p.createCollisionShapeArray(shapeTypes=[p.GEOM_BOX]*len(half),halfExtents=half,collisionFramePositions=positions)
  colors=color if isinstance(color[0],(list,tuple)) else [color]*len(half)
  v=p.createVisualShapeArray(shapeTypes=[p.GEOM_BOX]*len(half),halfExtents=half,visualFramePositions=positions,rgbaColors=colors)
  return self.link(pos,mass,c,v)

 def motor(self,name,q):
  a=self.arms[name];b=a['body']
  p.setJointMotorControlArray(b,list(range(7)),p.POSITION_CONTROL,targetPositions=list(q[:7]),forces=[160]*7,positionGains=[.08]*7,velocityGains=[1]*7)
  p.setJointMotorControlArray(b,[9,10],p.POSITION_CONTROL,targetPositions=[a['gap']]*2,forces=[a.get('grip_force',10)]*2)

 def move(self,name,end,seconds,force_limit=None):
  a=self.arms[name];b=a['body'];start=np.array(p.getLinkState(b,11,computeForwardKinematics=True)[4]);end=np.array(end)
  for k in range(max(1,int(seconds*60))):
   u=(k+1)/max(1,int(seconds*60));s=u*u*(3-2*u);pos=start*(1-s)+end*s
   q=p.calculateInverseKinematics(b,11,pos.tolist(),targetOrientation=a['orn'],maxNumIterations=150,residualThreshold=.0001)
   self.motor(name,q);self.step(8)
   if force_limit is not None:
    body,limit=force_limit
    forces=[c['force'] for c in self.contacts if c['step']>self.steps-8 and c['actor']==name and c['other']==self.names[body]]
    mean_force=sum(forces)/(8*self.physics_substeps)
    if mean_force>limit:
     self.motor(name,[p.getJointState(b,j)[0] for j in range(7)])
     self.force_stops.append({'step':self.steps,'actor':name,'mean_N':mean_force,'peak_N':max(forces),'limit_N':limit});break
  actual=p.getLinkState(b,11,computeForwardKinematics=True)[4]
  self.waypoints.append({'phase':self.phase,'actor':name,'goal':end.tolist(),'actual':list(actual),'error':float(np.linalg.norm(end-np.array(actual)))})
  if getattr(self,'debug_folder',None):
   self.debug_folder.mkdir(parents=True,exist_ok=True)
   Image.fromarray(self.image(True)).save(self.debug_folder/f'{len(self.waypoints):02d}_{self.phase}.png')

 def home(self,name):
  a=self.arms[name];q0=np.array([p.getJointState(a['body'],j)[0] for j in range(7)])
  for k in range(90):
   u=(k+1)/90;s=u*u*(3-2*u);self.motor(name,q0*(1-s)+a['home']*s);self.step(8)

 @staticmethod
 def stationarity_report(rows):
  xyz=np.array([[pose[0] for pose in row[0]] for row in rows]);quat=np.array([[pose[1] for pose in row[0]] for row in rows]);sliders=np.array([row[1] for row in rows])
  report={'window_seconds':len(rows)/480,'max_translation_range_m':float(np.max(np.linalg.norm(np.ptp(xyz,axis=0),axis=1))),'max_rotation_from_first_deg':float(np.max(2*np.degrees(np.arccos(np.clip(np.abs(np.sum(quat*quat[0:1],axis=2)),0,1))))),'max_slider_range_m':float(np.max(np.ptp(sliders,axis=0))),'scope':'All four dominoes, both robot end effectors, bell and pusher sliders'}
  report['pass']=len(rows)==240 and report['max_translation_range_m']<.0002 and report['max_rotation_from_first_deg']<.1 and report['max_slider_range_m']<.0002
  report['per_body_translation_range_m']=np.linalg.norm(np.ptp(xyz,axis=0),axis=1).tolist()
  report['per_body_rotation_range_deg']=np.max(2*np.degrees(np.arccos(np.clip(np.abs(np.sum(quat*quat[0:1],axis=2)),0,1))),axis=0).tolist()
  report['body_order']=['F1','F2','F3','F4','primary_EE','root_EE']
  return report

 def snapshot(self):
  Image.fromarray(self.image(True)).save(self.folder/f'{len(self.frames):04d}.png')
  self.frames.append({'step':self.steps,'phase':self.phase,'bell_contact':self.strike,'lamp':self.light})

 def step(self,n):
  for _ in range(n*self.physics_substeps):
   p.stepSimulation();self.steps+=1/self.physics_substeps
   pq=float(p.getJointState(self.pusher,0)[0]);self.pusher_q_range[0]=min(self.pusher_q_range[0],pq);self.pusher_q_range[1]=max(self.pusher_q_range[1],pq)
   for c in p.getContactPoints():
    if float(c[8])<self.worst:
     self.worst=float(c[8]);self.worst_contact={'step':self.steps,'phase':self.phase,'a':self.names.get(c[1],str(c[1])),'b':self.names.get(c[2],str(c[2])),'link_a':c[3],'link_b':c[4]}
    pair={c[1],c[2]}
    if c[9]>.001:
     if pair=={self.bell,self.gong} and c[9]>.001 and not self.strike:self.strike=True;self.strike_step=self.steps
     for name,a in self.arms.items():
      if a['body'] in pair:
       other=c[2] if c[1]==a['body'] else c[1];link=c[3] if c[1]==a['body'] else c[4]
       self.contacts.append({'step':self.steps,'phase':self.phase,'actor':name,'other':self.names.get(other,str(other)),'link':link,'distance':float(c[8]),'force':float(c[9])})
   sq=float(p.getJointState(self.switch,0)[0])
   light=bool(sq>.018 or (sq>=-.018 and self.strike))
   if light!=self.light:
    self.light=light;p.changeVisualShape(self.lamp,-1,rgbaColor=[.15,.85,.30,1] if light else [.08,.20,.11,1])
   if self.folder and self.steps%24==0:self.snapshot()

 def release_staging(self):
  if not self.released:
   for b in self.bodies:p.changeDynamics(b,-1,mass=.08)
   self.released=True;self.release_step=self.steps

 def push_handle(self,name,body,x,y,z,stroke,release=True):
  self.phase=f'{name}_approach';self.move(name,[x-.10,y,.66],1.2)
  self.phase=f'{name}_descend';self.move(name,[x-.07,y,z],1.)
  self.step(120)
  if release:self.release_staging()
  p.setJointMotorControl2(body,0,p.VELOCITY_CONTROL,targetVelocity=0,force=.03 if body==self.pusher else .003)
  self.phase=f'{name}_push';self.move(name,[x+stroke,y,z],1.,(body,4.) if body==self.pusher else None)
  if body==self.pusher:
   # A carriage brake resists spring recoil after the robot's push. It applies
   # bounded zero-velocity friction, not a desired position or a state reset.
   p.setJointMotorControl2(body,0,p.VELOCITY_CONTROL,targetVelocity=0,force=.75)
  # Lift vertically clear of the handle before moving back over the rig.
  actual=p.getLinkState(self.arms[name]['body'],11,computeForwardKinematics=True)[4]
  self.phase=f'{name}_retract';self.move(name,[actual[0],actual[1],.67],.8)
  self.phase=f'{name}_return';self.home(name)

 def sample_robot(self,seed=80000000,target=0,value=1,root_on=False,capture=None,context=None):
  assert target is None or target in range(7)
  self.folder=Path(capture) if capture else None
  if self.folder:self.folder.mkdir(parents=True,exist_ok=False)
  self.frames=[];self.contacts=[];self.waypoints=[];self.force_stops=[];self.steps=0;self.phase='initial';self.strike=False;self.light=False;self.strike_step=None;self.worst=0.;self.worst_contact=None;self.pusher_q_range=[0.,0.]
  self.names={self.pusher:'P',self.bell:'B',self.gong:'gong',self.switch:'switch',self.table:'table',**{b:f'pusher_stop_{i}' for i,b in enumerate(self.pusher_stops)},**{a['body']:name for name,a in self.arms.items()}}
  rng=np.random.default_rng(seed);dx=rng.uniform(-.002,.002,4);dy=float(rng.uniform(-.003,.003))
  ctx={'lean':[False]*4,'bell_noise':False,'lamp_noise':False} if context is None else dict(context)
  lean=np.array(ctx['lean'],bool).copy()
  if target in [1,2,3,4]:lean[target-1]=False
  self.released=not any(lean);self.release_step=0 if self.released else None
  bodies=[];self.bodies=bodies
  for i,x in enumerate([.18,.18,.29,.29]):
   lane_y=dy+(-.06 if i%2==0 else .08)
   angle=math.radians(20 if lean[i] else 0);pivot=x+dx[i]+.012
   pos=[pivot-.012*math.cos(angle)+.09*math.sin(angle),lane_y,self.top+.012*math.sin(angle)+.09*math.cos(angle)+.0001]
   b=self.box([.012,.05,.09],COLORS[i],pos,.08 if self.released else 0)
   p.resetBasePositionAndOrientation(b,pos,p.getQuaternionFromEuler([0,angle,0]));p.changeDynamics(b,-1,lateralFriction=.6,restitution=0,linearDamping=.04,angularDamping=.04)
   bodies.append(b);self.names[b]=f'F{i+1}'
  for body in [self.pusher,self.bell,self.switch]:p.setJointMotorControl2(body,0,p.POSITION_CONTROL,targetPosition=0,force=3)
  if ctx['lamp_noise'] and target!=6:
   p.resetJointState(self.switch,0,.035)
   p.setJointMotorControl2(self.switch,0,p.POSITION_CONTROL,targetPosition=.035,force=.2)
  self.step(240)
  p.setJointMotorControl2(self.bell,0,p.VELOCITY_CONTROL,targetVelocity=0,force=.003)
  if self.folder:self.snapshot()
  if ctx['bell_noise'] and target!=5:
   # Independent factual context: an actual gripper push of the striker.
   # Domino staging remains engaged until the requested intervention is ready.
   self.push_handle('primary',self.bell,.42,.23,.405,-.0065,release=False)
  context_cutoff=self.steps
  if target is None:
   self.release_staging()
   if root_on:self.push_handle('root',self.pusher,.065,-.15,.429,.08)
  elif target==0:
   if value:self.push_handle('root',self.pusher,.065,-.15,.429,.08)
   else:
    a=self.arms['root'];a['gap']=.020;a['grip_force']=3
    self.phase='root_approach';self.move('root',[.065,-.15,.66],1.2)
    self.phase='root_descend';self.move('root',[.065,-.15,.429],1.)
    a['gap']=.0115;self.phase='root_hold';self.motor('root',[p.getJointState(a['body'],j)[0] for j in range(7)]);self.step(240)
    p.setJointMotorControl2(self.pusher,0,p.VELOCITY_CONTROL,targetVelocity=0,force=.003)
    self.release_staging()
  elif target in range(1,5):
   dy=p.getBasePositionAndOrientation(bodies[target-1])[0][1]
   x=p.getBasePositionAndOrientation(bodies[target-1])[0][0];a=self.arms['primary'];a['gap']=.020 if value==0 else .008
   a['orn']=a['grasp_orn'] if value==0 else a['push_orn']
   # Grasp locations are actuator geometry, not latent-state corrections.
   # F2 is pinched centrally to avoid the edge pivot seen in held controls.
   gy=dy+(.025 if target==1 else 0. if target==2 else .025) if value==0 else dy
   grasp_z=.484
   self.phase='primary_approach';self.move('primary',[x if value==0 else x-.05,gy,.67],1.2)
   self.phase='primary_descend';self.move('primary',[x if value==0 else x-.026,gy,grasp_z if value==0 else .485],1.)
   if value==0:
    a['gap']=.010;self.phase='primary_hold';self.motor('primary',[p.getJointState(a['body'],j)[0] for j in range(7)]);self.step(240)
    self.release_staging()
   else:
    self.release_staging()
    self.phase='primary_push';self.move('primary',[x+.055,dy,.485],.3)
    self.phase='primary_retract';self.move('primary',[x-.025,dy,.67],.8);self.home('primary')
   if root_on:self.push_handle('root',self.pusher,.065,-.15,.429,.08)
  elif target==5:
   if value:self.push_handle('primary',self.bell,.42,.23,.405,-.0065)
   else:
    # Grasp the striker's separate handle; disable its motor after closing.
    a=self.arms['primary'];a['gap']=.020;a['orn']=a['grasp_orn']
    self.phase='primary_approach';self.move('primary',[.42,.23,.67],1.3)
    self.phase='primary_descend';self.move('primary',[.42,.23,.405],1.)
    a['gap']=.012;a['grip_force']=2;self.phase='primary_hold';self.motor('primary',[p.getJointState(a['body'],j)[0] for j in range(7)]);self.step(240)
    p.setJointMotorControl2(self.bell,0,p.VELOCITY_CONTROL,targetVelocity=0,force=.003)
    self.release_staging()
   if root_on:self.push_handle('root',self.pusher,.065,-.15,.429,.08)
  else:
   # Direct lamp intervention must move its physical selector via contact.
   a=self.arms['primary'];a['gap']=.020;a['orn']=a['grasp_orn']
   self.phase='switch_approach';self.move('primary',[.84,.17,.67],1.4)
   self.phase='switch_descend';self.move('primary',[.84,.17,.365],1.)
   a['gap']=.009;a['grip_force']=3;self.motor('primary',[p.getJointState(a['body'],j)[0] for j in range(7)]);self.step(240)
   p.setJointMotorControl2(self.switch,0,p.VELOCITY_CONTROL,targetVelocity=0,force=.003)
   self.phase='switch_select';self.move('primary',[.84+(.035 if value else -.035),.17,.365],.6)
   self.switch_hold=float(p.getJointState(self.switch,0)[0])
   # Mechanical detent holds the physically attained position, not the command.
   p.setJointMotorControl2(self.switch,0,p.POSITION_CONTROL,targetPosition=self.switch_hold,force=.2)
   a['gap']=.04;self.motor('primary',[p.getJointState(a['body'],j)[0] for j in range(7)]);self.step(120)
   self.phase='switch_retract';self.move('primary',[.84,.17,.67],.8);self.home('primary')
   self.release_staging()
   if root_on:self.push_handle('root',self.pusher,.065,-.15,.429,.08)
  # Bell remains passive unless directly grasped; never set from desired state.
  self.phase='observe';stable=0;pose_window=deque(maxlen=240);stationarity={'pass':False}
  for _ in range(14400):
   self.step(1);vv=[p.getBaseVelocity(b) for b in bodies]
   poses=[p.getBasePositionAndOrientation(b) for b in bodies]+[p.getLinkState(a['body'],11,computeForwardKinematics=True)[4:6] for a in self.arms.values()]
   pose_window.append((poses,[float(p.getJointState(b,0)[0]) for b in [self.bell,self.pusher]]))
   quiet=max(np.linalg.norm(a) for a,b in vv)<.003 and max(np.linalg.norm(b) for a,b in vv)<.025 and abs(p.getJointState(self.bell,0)[1])<.004
   stable=stable+1 if quiet else 0
   if len(pose_window)==240 and self.steps%24==0:
    stationarity=self.stationarity_report(pose_window)
    if stationarity['pass']:break
  angles=[float(math.degrees(math.acos(np.clip(np.array(p.getMatrixFromQuaternion(p.getBasePositionAndOrientation(b)[1])).reshape(3,3)[2,2],-1,1)))) for b in bodies]
  pq=float(p.getJointState(self.pusher,0)[0]);bq=float(p.getJointState(self.bell,0)[0]);sq=float(p.getJointState(self.switch,0)[0])
  labels=[int(pq>=.04)]+[int(a>=10) for a in angles]+[int(self.strike),int(self.light)]
  allowed={'root':{'P'},'primary':{f'F{target}' if target in [1,2,3,4] else ('B' if target in [5,None] else 'switch')}}
  wrong=[c for c in self.contacts if c['other'] not in allowed[c['actor']] and not(c['step']<=context_cutoff and ctx['bell_noise'] and c['actor']=='primary' and c['other']=='B')]
  result={'seed':seed,'target':target,'value':value,'root_on':root_on,'labels':labels,'angles':angles,'pusher_q':pq,'striker_q':bq,'switch_q':sq,'settled':stable>=240,'target_realized':target is None or labels[target]==value,'first_strike_step':self.strike_step,'min_contact_distance':self.worst,'unexpected_robot_contacts':len(wrong),'unexpected_pairs':sorted(set((c['actor'],c['other'],c['link']) for c in wrong)),'contacts':self.contacts,'waypoints':self.waypoints,'frame_phases':self.frames,'steps':self.steps,'post_release_pose_resets':0,'all_dominoes_dynamic':True,'scope':'Robot-contact physical acquisition; graph learner receives RGB only','controller_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
  result['context']=ctx;result['initial_tilts']=lean.tolist();result['release_step']=self.release_step;result['staged_initial_condition']=context is not None
  result['worst_contact']=self.worst_contact
  result['force_stops']=self.force_stops
  result['final_velocities']=[{'linear':list(p.getBaseVelocity(b)[0]),'angular':list(p.getBaseVelocity(b)[1])} for b in bodies]
  result['striker_velocity']=float(p.getJointState(self.bell,0)[1])
  result['quiet_steps_at_measurement']=stable
  result['stationarity_diagnostics']=self.stationarity_report(pose_window)
  result['instantaneous_velocity_gate_pass']=stable>=240
  result['settled']=result['stationarity_diagnostics']['pass']
  result['settling_protocol']='pose_window_0.5s_0.2mm_0.1deg_v1'
  result['pusher_q_range']=self.pusher_q_range
  result['pusher_physical_bounds']=list(self.pusher_bounds)
  result['pusher_within_physical_travel']=self.pusher_q_range[0]>=self.pusher_bounds[0]-.001 and self.pusher_q_range[1]<=self.pusher_bounds[1]+.001
  im=self.image()
  if self.folder:Image.fromarray(im).save(self.folder/'model_input.png');(self.folder/'audit.json').write_text(json.dumps(result,indent=2))
  return im,labels,result

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--target',type=int,default=0);ap.add_argument('--value',type=int,default=1);ap.add_argument('--root-on',action='store_true');ap.add_argument('--capture',action='store_true');ap.add_argument('--seed',type=int,default=80000000);a=ap.parse_args()
 env=RobotFork();env.debug_folder=Path(a.out+'_waypoints') if not a.capture else None
 try:
  im,ys,audit=env.sample_robot(a.seed,a.target,a.value,a.root_on,a.out if a.capture else None)
  if not a.capture:
   out=Path(a.out);out.mkdir(parents=True,exist_ok=False);Image.fromarray(im).save(out/'model_input.png');Image.fromarray(env.image(True)).save(out/'scene.png');(out/'audit.json').write_text(json.dumps(audit,indent=2))
  print(json.dumps({k:v for k,v in audit.items() if k not in ['contacts','frame_phases','waypoints']}),flush=True)
 finally:env.close()

