"""
kinematic_tree.py — Build kinematic tree from components and joints.
Assigns depth (distance from root) to each component for animation ordering.
"""
import logging
from collections import defaultdict, deque
from typing import List

log = logging.getLogger("twinforge.tree")


class KinematicTreeBuilder:
    def __init__(self, comps, joints):
        self.comps  = {c.id: c for c in comps}
        self.joints = joints

    def assign_depths(self):
        """BFS from root to assign depth to each component."""
        children = defaultdict(list)
        for j in self.joints:
            children[j.parent_id].append(j.child_id)

        # Root = component with no parent in any joint
        all_children = {j.child_id for j in self.joints}
        roots = [cid for cid in self.comps if cid not in all_children]
        if not roots:
            roots = [list(self.comps.keys())[0]]

        visited = set()
        queue = deque((r, 0) for r in roots)
        while queue:
            cid, depth = queue.popleft()
            if cid in visited:
                continue
            visited.add(cid)
            if cid in self.comps:
                self.comps[cid].depth = depth
            for child in children.get(cid, []):
                queue.append((child, depth + 1))

        log.info(f"Kinematic tree: root(s)={roots}, {len(visited)} nodes")
        return list(self.comps.values())
