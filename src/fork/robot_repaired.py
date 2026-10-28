"""Portable form of the recorded 5 mm continuation-only F1 hold correction."""
import numpy as np
import pybullet as p
from robot_revision import RobotFork as OriginalRobotFork

class RobotFork(OriginalRobotFork):
    hold_y_correction=-.005
    def sample_robot(self,seed=80000000,target=0,value=1,root_on=False,capture=None,context=None):
        self.repair_hold=target==1 and value==0
        im,labels,audit=super().sample_robot(seed,target,value,root_on,capture,context)
        audit['F1_hold_y_correction_m']=self.hold_y_correction
        return im,labels,audit
    def move(self,name,end,seconds,force_limit=None):
        dest=np.array(end,dtype=float).copy()
        if getattr(self,'repair_hold',False) and name=='primary' and self.phase in ('primary_approach','primary_descend') and getattr(self,'bodies',[]):
            x,y,_=p.getBasePositionAndOrientation(self.bodies[0])[0]
            if abs(dest[0]-x)<1e-7 and abs(dest[1]-(y+.025))<1e-7:dest[1]+=self.hold_y_correction
        return super().move(name,dest,seconds,force_limit)
