import sys, json, logging, time; sys.path.insert(0,'/home/claude/twinforge3d'); sys.path.insert(0,'.')
logging.basicConfig(level=logging.INFO,format="%(message)s")
import numpy as np, open3d as o3d
from core import jointscan as mp
NAMES=("base","waist","arm1","arm2","gripper")
GT=json.load(open("gt_joints.json"))
def synth(pose, seed, noise=0.0015, yaw=None, scale=None, n=40000):
    rng=np.random.default_rng(seed); T=np.array(json.load(open(f"pose_{pose}.json")) if False else 0)
    pj=json.load(open(f"pose_{pose}.json")); pts=[];lab=[]
    for i,nm in enumerate(NAMES):
        m=o3d.io.read_triangle_mesh(f"world_{nm}.stl"); P=np.asarray(m.sample_points_uniformly(n//5).points)
        M=np.array(pj[nm]).reshape(4,4); P=P@M[:3,:3].T+M[:3,3]; pts.append(P); lab+= [i]*len(P)
    P=np.vstack(pts)+rng.normal(0,noise,(len(pts)*0+sum(len(p) for p in pts),3)); lab=np.array(lab)
    return P,lab
def similarity(P, yaw, s, seed):
    c,sn=np.cos(yaw),np.sin(yaw); R=np.array([[c,-sn,0],[sn,c,0],[0,0,1]]); Q=(P@R.T)*s
    Q=Q-[Q[:,0].mean()+0.03*seed,Q[:,1].mean(),Q[:,2].min()]; return Q
if __name__=="__main__":
    poses=sys.argv[1].split(","); noise=float(sys.argv[2]) if len(sys.argv)>2 else 0.0015
    clouds=[];labs=[]
    for k,p in enumerate(poses):
        P,l=synth(p,k,noise); 
        P=similarity(P, 0 if k==0 else np.radians(37*k), 1.0 if k==0 else 1+0.17*k, k)
        clouds.append(P); labs.append(l)
    # ref is pose 0 but its own placement: undo for GT comparison -> compute GT joints in ref cloud frame via base alignment later
    t=time.time(); aligned,links,joints,parent=mp.estimate(clouds); print("time",time.time()-t)
    # evaluation vs GT in ref frame: need ref frame mapping: ref cloud = similarity(synth(poses[0])) ; undo
    P0,_=synth(poses[0],0,0); P0s=similarity(P0,0,1.0,0)
    off=P0s[:,:3].mean(0)-P0[:,:3].mean(0)   # not exact; use direct fit below
    # exact: P0s = P0 - [mean_x,mean_y,minz] -> translation
    tr=np.array([P0[:,0].mean(),P0[:,1].mean(),P0[:,2].min()])
    print("links:",[(L.id,len(L.ref_idx)) for L in links])
    # link purity vs GT labels of ref
    ref=clouds[0]; gl=labs[0]
    for L in links:
        cnt=np.bincount(gl[L.ref_idx],minlength=5); print(L.id,"GT composition",dict(zip(NAMES,cnt)))
    for j in joints:
        # match GT by child majority label
        ch=[L for L in links if L.id==j.child][0]; g=NAMES[np.bincount(gl[ch.ref_idx],minlength=5).argmax()]
        G=GT[g]; gx=np.array(G["xyz"])-tr; ga=np.array(G["axis"]); ga/=np.linalg.norm(ga)
        axerr=np.degrees(np.arccos(min(1,abs(j.axis@ga)))); 
        # point-to-line distance of estimated point to GT axis line
        d=gx-j.xyz; perp=np.linalg.norm(d-(d@ga)*ga)
        print(f"{j.name} child~{g}: type {j.type} axis err {axerr:.2f}° | point-to-GT-axis {perp*1000:.1f} mm | spread {j.axis_spread_deg:.1f}° conf {j.confidence} angles {np.round(np.degrees(j.angles),0)}")
