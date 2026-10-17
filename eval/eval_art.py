"""Compare an auto-detected URDF against the original robot over random joint configurations."""
import sys, json, numpy as np, trimesh, yourdfpy, open3d as o3d
from pathlib import Path
from scipy.spatial import cKDTree
sys.path.insert(0,'.')
from eval_lib import rz
OUT=Path(sys.argv[1]); URDF=OUT/sys.argv[2]; seed=int(sys.argv[3]) if len(sys.argv)>3 else 0
import os; REF=os.environ.get('ARM_PKG','smart-methods/arduino_robot_arm/robot_arm_pkg'); REST=np.array([0,0.6,-0.9,0.5])
def load(urdf, md): return yourdfpy.URDF.load(str(urdf),load_meshes=False,build_collision_scene_graph=False,filename_handler=lambda f:str(Path(md)/Path(f.replace('%20',' ')).name))
def assembled(u,md,cfg):
    u.update_cfg(cfg); ms=[]
    for l in u.robot.links:
        for v in l.visuals:
            m=v.geometry.mesh
            if m is None: continue
            mesh=trimesh.load(Path(md)/Path(m.filename.replace('%20',' ')).name,force='mesh')
            if m.scale is not None: mesh.apply_scale(np.array(m.scale))
            mesh.apply_transform(u.get_transform(l.name)@(v.origin if v.origin is not None else np.eye(4))); ms.append(mesh)
    return trimesh.util.concatenate(ms)
ours=load(URDF,OUT/"meshes"); ref=load(f"{REF}/urdf/arduino_robot_arm.urdf",f"{REF}/meshes/stl")
# GT rest world frame is that of REF at zero; our frame = scan frame -> align at q=0 / rest
A=assembled(ours,OUT/"meshes",np.zeros(len(ours.actuated_joint_names))); G=assembled(ref,f"{REF}/meshes/stl",REST)
Pa=np.asarray(A.sample(50000)); Pg=np.asarray(G.sample(50000)); tg=cKDTree(Pg)
best=None
for deg in range(0,360,3):
    Q=Pa@rz(np.radians(deg)).T; Q=Q-[Q[:,0].mean()-Pg[:,0].mean(),Q[:,1].mean()-Pg[:,1].mean(),Q[:,2].min()-Pg[:,2].min()]
    d=tg.query(Q[::8])[0].mean()
    if best is None or d<best[0]: best=(d,deg)
R0=rz(np.radians(best[1])); S=Pa@R0.T; off=np.array([S[:,0].mean()-Pg[:,0].mean(),S[:,1].mean()-Pg[:,1].mean(),S[:,2].min()-Pg[:,2].min()])
src=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(S-off)); dst=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(Pg))
icp=o3d.pipelines.registration.registration_icp(src,dst,0.01,np.eye(4),o3d.pipelines.registration.TransformationEstimationPointToPoint())
T=np.eye(4); T[:3,:3]=R0; pre=np.eye(4); pre[:3,3]=-off; Tfull=icp.transformation@pre@T
# joint axis sign mapping: compare our (aligned) axes to GT axes at rest
gt=json.load(open("gt_joints.json")); gnames=["waist","arm1","arm2","gripper"]
signs=[]
for j,gn in zip(ours.robot.joints,gnames): pass
def chamfer(a,b):
    P=np.asarray(a.sample(30000)); Q=np.asarray(b.sample(30000)); d1=cKDTree(Q).query(P)[0]; d2=cKDTree(P).query(Q)[0]
    return (d1.mean()+d2.mean())/2*1000, (d1<.005).mean()*100,(d2<.005).mean()*100
# our actuated joints in order of ours.actuated_joint_names; need sign wrt GT: derive from small test: rotate joint, compare
rng=np.random.default_rng(seed); n=len(ours.actuated_joint_names)
# determine sign per joint by testing +0.5 against both +/-0.5 in GT
signs=[]
for i in range(n):
    q=np.zeros(n); q[i]=0.5; a=assembled(ours,OUT/"meshes",q); a.apply_transform(Tfull)
    errs=[]
    for sg in (1,-1):
        c=REST.copy(); c[i]+=sg*0.5; errs.append(chamfer(a,assembled(ref,f"{REF}/meshes/stl",c))[0])
    signs.append(1 if errs[0]<errs[1] else -1)
print("detected sign per joint (+1 = same direction as the original):",signs, "| rest alignment Chamfer %.2f mm"%chamfer(assembled(ours,OUT/"meshes",np.zeros(n)).apply_transform(Tfull) or assembled(ours,OUT/"meshes",np.zeros(n)),G)[0] if False else "")
rows=[]
for t in range(6):
    q=np.zeros(n) if t==0 else rng.uniform(-0.6,0.6,n)
    a=assembled(ours,OUT/"meshes",q); a.apply_transform(Tfull)
    g=assembled(ref,f"{REF}/meshes/stl",REST+np.array(signs)*q)
    c=chamfer(a,g); rows.append(c); print(f"pose{t} q={np.round(q,2)}  Chamfer {c[0]:.2f} mm | precision {c[1]:.0f}% recall {c[2]:.0f}%")
print("MEAN Chamfer %.2f mm | precision %.0f%% | recall %.0f%%"%tuple(np.mean(rows,0)))
# ---- joint parameter errors vs the original robot (rest configuration) ----
ours.update_cfg(np.zeros(n)); GT=json.load(open("gt_joints.json")); rows=[]
order=["waist","arm1","arm2","gripper"]
jl=[j for j in ours.robot.joints if j.type in ("revolute","continuous","prismatic")]
for j in jl:
    Tj=Tfull@ours.get_transform(j.parent)@(j.origin if j.origin is not None else np.eye(4)); ax=Tj[:3,:3]@np.array(j.axis); ax/=np.linalg.norm(ax); xyz=Tj[:3,3]
    best=None
    for gn in order:
        g=GT[gn]; ga=np.array(g["axis"]); ga/=np.linalg.norm(ga)
        err=np.degrees(np.arccos(min(1,abs(ax@ga)))); d=np.array(g["xyz"])-xyz; perp=np.linalg.norm(d-(d@ga)*ga)
        if best is None or err+perp*500<best[0]: best=(err+perp*500,gn,err,perp)
    print(f"{j.name}: matches GT '{best[1]}'  axis error {best[2]:.2f}°  position error (to axis line) {best[3]*1000:.1f} mm")
