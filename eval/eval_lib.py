import numpy as np, open3d as o3d
from scipy.spatial import cKDTree
NAMES=("base","waist","arm1","arm2","gripper")
def _gt():
    ms=[o3d.io.read_triangle_mesh(f"./world_{i}.stl") for i in NAMES]
    g=o3d.geometry.TriangleMesh()
    for m in ms: g+=m
    return ms,g
def rz(a): c,s=np.cos(a),np.sin(a); return np.array([[c,-s,0],[s,c,0],[0,0,1]])
def evaluate(mesh, verbose=True):
    """mesh: metric, z-up, base at z=0 (arbitrary yaw). Returns dict + 4x4 transform into GT frame."""
    ms,gt=_gt(); G=np.asarray(gt.sample_points_uniformly(60000).points); tg=cKDTree(G)
    P=np.asarray(mesh.sample_points_uniformly(60000).points)
    best=None
    for deg in range(0,360,5):
        Q=P@rz(np.radians(deg)).T
        Q=Q-np.array([Q[:,0].mean()-G[:,0].mean(),Q[:,1].mean()-G[:,1].mean(),Q[:,2].min()-G[:,2].min()])
        d=tg.query(Q[::6])[0].mean()
        if best is None or d<best[0]: best=(d,deg)
    R0=rz(np.radians(best[1])); S=P@R0.T
    off=np.array([S[:,0].mean()-G[:,0].mean(),S[:,1].mean()-G[:,1].mean(),S[:,2].min()-G[:,2].min()])
    src=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(S-off)); dst=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(G))
    icp=o3d.pipelines.registration.registration_icp(src,dst,0.01,np.eye(4),o3d.pipelines.registration.TransformationEstimationPointToPoint())
    src.transform(icp.transformation); Q=np.asarray(src.points)
    d1=tg.query(Q)[0]; d2=cKDTree(Q).query(G)[0]
    pre=np.eye(4); pre[:3,3]=-off; R=np.eye(4); R[:3,:3]=R0
    out=dict(chamfer_mm=(d1.mean()+d2.mean())/2*1000,p95_mm=np.percentile(np.r_[d1,d2],95)*1000,
             prec5=(d1<.005).mean()*100,rec5=(d2<.005).mean()*100,tris=len(mesh.triangles))
    if verbose: print("tris %d | Chamfer %.2f mm | p95 %.1f mm | within5mm: precision %.0f%% recall %.0f%%"%(out["tris"],out["chamfer_mm"],out["p95_mm"],out["prec5"],out["rec5"]))
    return out, icp.transformation@pre@R
