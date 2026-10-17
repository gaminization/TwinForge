import sys, json, logging, time; import pathlib; sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent.parent)); sys.path.insert(0,'.')
logging.basicConfig(level=logging.INFO,format="%(message)s")
import numpy as np, open3d as o3d, warnings; warnings.filterwarnings("ignore")
from core import jointscan as js
from mp_eval import synth, similarity, NAMES, GT
noise=float(sys.argv[1]) if len(sys.argv)>1 else 0.0015; seed=int(sys.argv[2]) if len(sys.argv)>2 else 0
rng=np.random.default_rng(seed)
P0,l0=synth("R",seed,noise); ref=similarity(P0,rng.uniform(0,6.28),1.0,0)
tr=np.array([P0[:,0].mean(),P0[:,1].mean(),P0[:,2].min()])
# similarity() rotates about origin first: recover exact GT frame mapping ref->world by fitting later; instead compute GT in ref frame by applying same transform
def tf(P,yaw,s_,shift): 
    c,sn=np.cos(yaw),np.sin(yaw); R=np.array([[c,-sn,0],[sn,c,0],[0,0,1]]); return R,(P@R.T)*s_
yaw0=rng.uniform(0,6.28); R0,Q=tf(P0,yaw0,1.0,0); sh=np.array([Q[:,0].mean(),Q[:,1].mean(),Q[:,2].min()]); ref=Q-sh
scans=[]
for j in ("K1","K2","K3","K4"):
    Pj,_=synth(j,seed+7,noise); yaw=rng.uniform(0,6.28); s_=rng.uniform(0.6,1.5); R,Q2=tf(Pj,yaw,s_,0)
    Q2=Q2-np.array([Q2[:,0].mean()+rng.normal(0,.02),Q2[:,1].mean(),Q2[:,2].min()]); scans.append(Q2)
t=time.time(); res=js.estimate(ref,scans); print("time %.0fs"%(time.time()-t))
gl=l0
print("links sizes",np.bincount(res.labels))
for L in range(len(res.link_ids)):
    c=np.bincount(gl[res.labels==L],minlength=5); print(res.link_ids[L],dict(zip(NAMES,c)))
GTchild=["waist","arm1","arm2","gripper"]
for i,j in enumerate(res.joints):
    G=GT[GTchild[i]]; gx=R0@np.array(G["xyz"])-sh; ga=R0@np.array(G["axis"]); ga/=np.linalg.norm(ga)
    ax=np.degrees(np.arccos(min(1,abs(j.axis@ga)))); d=gx-j.xyz; perp=np.linalg.norm(d-(d@ga)*ga); along=abs(d@ga)
    sgn=np.sign(j.axis@ga); gt_ang=[0.7,0.7,0.7,0.7][i]*sgn
    print(f"{j.name} ({j.parent}->{j.child}) GT={GTchild[i]}: axis err {ax:.2f}° | pos perp-to-axis {perp*1000:.1f} mm | rms {j.rms_mm:.1f} mm inl {j.inlier:.2f} {j.confidence} | angle est {np.degrees(j.observed):.1f}° vs GT {abs(np.degrees(gt_ang)):.1f}°")
