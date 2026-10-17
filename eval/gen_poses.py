import sys, json, numpy as np, yourdfpy
from pathlib import Path
import os; REF=os.environ.get('ARM_PKG','smart-methods/arduino_robot_arm/robot_arm_pkg')
u=yourdfpy.URDF.load(f"{REF}/urdf/arduino_robot_arm.urdf",load_meshes=False,build_collision_scene_graph=False,filename_handler=lambda f:str(Path(REF)/"meshes/stl"/Path(f).name))
print(u.actuated_joint_names,[ (j.name,j.parent,j.child,j.axis) for j in u.robot.joints])
POSES={"A":[0,0,0,0],"B":[0.6,0.7,-0.8,0.5],"C":[-0.7,-0.6,0.7,-0.6],"D":[1.0,0.5,0.6,0.9],"R":[0,0.6,-0.9,0.5],"K1":[0.7,0.6,-0.9,0.5],"K2":[0,-0.1,-0.9,0.5],"K3":[0,0.6,-0.2,0.5],"K4":[0,0.6,-0.9,1.2]}
NAMES=("base","waist","arm1","arm2","gripper")
u.update_cfg(np.zeros(4)); T0={n:u.get_transform(n) for n in NAMES}
u.update_cfg(np.array([0,0.6,-0.9,0.5]))
J0={}
for j in u.robot.joints:
    if j.type=="revolute":
        Tp=u.get_transform(j.parent); Tj=Tp@j.origin
        J0[j.child]=dict(parent=j.parent,xyz=Tj[:3,3].tolist(),axis=(Tj[:3,:3]@np.array(j.axis)).tolist(),lower=j.limit.lower,upper=j.limit.upper)
json.dump(J0,open("gt_joints.json","w"),indent=1)
for k,cfg in POSES.items():
    u.update_cfg(np.array(cfg)); d={n:(u.get_transform(n)@np.linalg.inv(T0[n])).flatten().tolist() for n in NAMES}
    json.dump(d,open(f"pose_{k}.json","w")); json.dump(cfg,open(f"cfg_{k}.json","w"))
print(json.dumps(J0,indent=0)[:600])
