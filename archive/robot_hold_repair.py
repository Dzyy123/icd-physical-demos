"""Small F1 hold-position correction; original physics and gates unchanged."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pybullet as p

BASE=Path('private-runs/icd_fork_revision_20261025/v6/source/robot_revision.py')
sys.path.insert(0,'private-runs/icd_fork_20261024/full_v11/source')
spec=importlib.util.spec_from_file_location('frozen_fork_controller',BASE)
original=importlib.util.module_from_spec(spec);spec.loader.exec_module(original)
assert hashlib.sha256(BASE.read_bytes()).hexdigest()=='a3936bbf3143ab6f38d3b7be408682840eb6adcae558e3e414a13b771fbb70b4'


class RobotFork(original.RobotFork):
    hold_y_correction=-.005

    def sample_robot(self,seed=80000000,target=0,value=1,root_on=False,capture=None,context=None):
        self.repair_hold=target==1 and value==0
        im,labels,audit=super().sample_robot(seed,target,value,root_on,capture,context)
        audit['base_controller_sha256']=hashlib.sha256(BASE.read_bytes()).hexdigest()
        audit['controller_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        audit['F1_hold_y_correction_m']=self.hold_y_correction
        return im,labels,audit

    def move(self,name,end,seconds,force_limit=None):
        dest=np.array(end,dtype=float).copy()
        if getattr(self,'repair_hold',False) and name=='primary' and self.phase in ('primary_approach','primary_descend') and getattr(self,'bodies',[]):
            x,y,_=p.getBasePositionAndOrientation(self.bodies[0])[0]
            # Only the F1 grasp, not the earlier off-lane background bell action.
            if abs(dest[0]-x)<1e-7 and abs(dest[1]-(y+.025))<1e-7:
                dest[1]+=self.hold_y_correction
        return super().move(name,dest,seconds,force_limit)
