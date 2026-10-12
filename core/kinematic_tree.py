"""
Kinematic Tree Construction
============================
Module 2 — Uses BFSKinematicSearch and DFSKinematicSearch (R&N §3.4)
Module 4 — Uses ResolutionVerifier for structural validation (R&N §9.5)
Module 6 — Tree structure is the state representation for planning (R&N §10.1)

The kinematic tree is the central data structure of TwinForge:
  - Nodes  = rigid links (Components)
  - Edges  = joints (parent → child directed)
  - Root   = base link (fixed to world)
  - Leaves = end-effectors, wheel hubs, tool flanges

BFS is used to enumerate levels and find adjacency pairs.
DFS is used for forward kinematics (recursive transform propagation).
Cycle detection (DFS) ensures the tree is a valid acyclic directed graph.
"""

from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class KinematicNode:
    link_id:        str
    joint_to_parent: Optional[object] = None   # Joint | None (root)
    children:       List["KinematicNode"] = field(default_factory=list)
    component:      Optional[object] = None    # Component ref
    depth:          int = 0                    # BFS level from root


@dataclass
class KinematicTree:
    root:   KinematicNode
    nodes:  Dict[str, KinematicNode] = field(default_factory=dict)
    joints: List[object] = field(default_factory=list)

    # ── Accessors ─────────────────────────────────────────────────────────────

    def all_links(self) -> List[str]:
        """BFS order (R&N §3.4.1)."""
        from core.ai_reasoning import BFSKinematicSearch
        adj = self._adjacency()
        return BFSKinematicSearch.bfs(self.root.link_id, adj)

    def adjacency(self) -> Dict[str, List[str]]:
        return self._adjacency()

    def _adjacency(self) -> Dict[str, List[str]]:
        adj: Dict[str, List[str]] = {}
        for node in self.nodes.values():
            for child in node.children:
                adj.setdefault(node.link_id, []).append(child.link_id)
        return adj

    def depth_of(self, link_id: str) -> int:
        return self.nodes[link_id].depth if link_id in self.nodes else -1

    def leaves(self) -> List[str]:
        """Links with no children."""
        return [lid for lid, node in self.nodes.items()
                if not node.children]

    def adjacent_pairs(self) -> List[Tuple[str, str]]:
        """
        BFS-enumerated (parent, child) edge list.
        Used by SRDF generator for collision disabling.
        """
        from core.ai_reasoning import BFSKinematicSearch
        return BFSKinematicSearch.adjacent_pairs(
            self.root.link_id, self._adjacency())


class KinematicTreeBuilder:
    """
    Assembles components + joints into a valid KinematicTree.

    Algorithm
    ---------
    1. Find root: link that appears as no joint's child
    2. Build node map
    3. Attach joints → parent.children
    4. Assign BFS depths
    5. Cycle detection via DFS (R&N §3.4.3)
    6. Resolution verification (Module 4)
    7. Log tree with DFS ordering
    """

    def build(self, components: list, joints: list) -> KinematicTree:
        from core.ai_reasoning import BFSKinematicSearch, DFSKinematicSearch, ResolutionVerifier

        # ── Step 1: Find root ─────────────────────────────────────────────────
        children_set = {j.child_link for j in joints}
        root_candidates = [c for c in components if c.id not in children_set]

        if not root_candidates:
            log.warning("No clear root — using largest component")
            root_comp = max(components,
                            key=lambda c: c.n_points if hasattr(c,"n_points") else 0)
        else:
            # Among candidates, prefer the one typed 'base'
            bases = [c for c in root_candidates if c.type == "base"]
            root_comp = bases[0] if bases else root_candidates[0]

        log.info(f"Kinematic tree root: '{root_comp.id}' (type={root_comp.type})")

        # ── Step 2: Build node map ────────────────────────────────────────────
        nodes: Dict[str, KinematicNode] = {
            c.id: KinematicNode(link_id=c.id, component=c)
            for c in components
        }

        # ── Step 3: Attach joints ─────────────────────────────────────────────
        for joint in joints:
            child_node  = nodes.get(joint.child_link)
            parent_node = nodes.get(joint.parent_link)
            if child_node is None or parent_node is None:
                log.warning(f"Joint '{joint.id}' references unknown link — skipped")
                continue
            child_node.joint_to_parent = joint
            parent_node.children.append(child_node)

        root_node = nodes[root_comp.id]

        # ── Step 4: BFS depth assignment ──────────────────────────────────────
        adj = {}
        for node in nodes.values():
            for child in node.children:
                adj.setdefault(node.link_id, []).append(child.link_id)

        for lid in BFSKinematicSearch.bfs(root_comp.id, adj):
            level = BFSKinematicSearch.find_level(root_comp.id, lid, adj)
            nodes[lid].depth = level

        # ── Step 5: Cycle detection via DFS ───────────────────────────────────
        if DFSKinematicSearch.has_cycle(root_comp.id, adj):
            raise ValueError(
                "Cycle detected in kinematic graph — URDF would be invalid. "
                "Check joint parent/child assignments.")

        # ── Step 6: Resolution verification ──────────────────────────────────
        verifier   = ResolutionVerifier()
        violations = verifier.verify_constraints(components, joints)
        if violations:
            for v in violations:
                log.warning(f"  Constraint violation: {v}")
        else:
            log.info("  All structural constraints satisfied (Resolution ✔)")

        tree = KinematicTree(root=root_node, nodes=nodes, joints=joints)

        # ── Step 7: Log tree (DFS order) ──────────────────────────────────────
        dfs_order = DFSKinematicSearch.dfs_order(root_comp.id, adj)
        log.info("Kinematic tree (DFS order):")
        for lid in dfs_order:
            node   = nodes[lid]
            indent = "  " * node.depth + ("└─ " if node.depth > 0 else "")
            jtype  = (node.joint_to_parent.type
                      if node.joint_to_parent else "root")
            log.info(f"   {indent}{lid}  [{jtype}]  depth={node.depth}")

        return tree
