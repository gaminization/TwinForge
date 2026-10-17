import sys, logging; import pathlib; sys.path.insert(0,str(pathlib.Path(__file__).resolve().parent.parent))
logging.basicConfig(level=logging.INFO, format="%(message)s")
import open3d as o3d, numpy as np
from core.photo_recon import reconstruct_from_photos
P=sys.argv[1]; size=float(sys.argv[2]) if len(sys.argv)>2 else None
src=f"photos_{P}"
r=reconstruct_from_photos(src,f"work_{P}",size_m=size)
print(P,{k:(round(v,4) if isinstance(v,float) else v) for k,v in r.report.items()})
o3d.io.write_triangle_mesh(f"scans/{P}.ply", r.mesh); np.save(f"scans/{P}_cams.npy", r.cam_centres)
