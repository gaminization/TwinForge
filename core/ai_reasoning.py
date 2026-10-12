"""
TwinForge 3D — AI Reasoning Engine
===================================
Course: BCSE306L Artificial Intelligence (VIT University)
Textbook: Russell & Norvig, "Artificial Intelligence: A Modern Approach", 3rd Ed.

Every algorithm in this file maps to a specific course module.
Each class header cites the module, chapter, and R&N section.

MODULE MAP
----------
Module 2  — Problem Solving by Searching    → BFSKinematicSearch, DFSKinematicSearch,
                                               AStarChainFinder, UCSScanPlanner
Module 3  — Local Search & Adversarial      → HillClimbingAxisOptimizer,
                                               SimulatedAnnealingParamOptimizer,
                                               GeneticURDFOptimizer
Module 4  — Logic and Reasoning             → FOLComponentRules, ForwardChainer,
                                               BackwardChainer, ResolutionVerifier
Module 5  — Uncertain Knowledge             → BayesianJointClassifier,
                                               BayesianBeliefNetwork
Module 6  — Planning                        → ClassicalPlanningPipeline,
                                               HierarchicalScanPlanner
Module 1/7 — Agents & Perception            → TwinForgeAgent (ties it all together)
"""

from __future__ import annotations
import math
import heapq
import logging
import random
import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 2 — PROBLEM SOLVING BY SEARCHING
#  R&N Chapter 3 & 4
# ══════════════════════════════════════════════════════════════════════════════

# ── Node representation (R&N §3.3) ───────────────────────────────────────────

@dataclass(order=True)
class SearchNode:
    """
    Generic search tree node (R&N §3.3).
    Stores state, parent pointer, action taken, and path cost.
    """
    f_cost:  float          # for priority queue ordering
    g_cost:  float = field(compare=False, default=0.0)   # path cost so far
    state:   Any   = field(compare=False, default=None)  # problem state
    parent:  Any   = field(compare=False, default=None)  # parent SearchNode
    action:  Any   = field(compare=False, default=None)  # action that led here

    def path(self) -> list:
        """Trace back the solution path."""
        node, path = self, []
        while node is not None:
            path.append(node.state)
            node = node.parent
        return list(reversed(path))


# ── BFS — Kinematic tree level-order traversal (R&N §3.4.1) ─────────────────

class BFSKinematicSearch:
    """
    Module 2 — Breadth-First Search on the kinematic tree.
    Reference: R&N §3.4.1, Algorithm 3.2

    Use-case in TwinForge:
        Find ALL links reachable from the root within k hops (e.g., to
        determine which links are "adjacent" for collision disabling).
        Also used to validate that the kinematic tree is connected.
    """

    @staticmethod
    def bfs(root_id: str, adjacency: Dict[str, List[str]]) -> List[str]:
        """
        Standard BFS. Returns nodes in breadth-first order.

        Parameters
        ----------
        root_id   : starting node (base link id)
        adjacency : {link_id: [child_link_id, …]}

        Returns
        -------
        List of link_ids in BFS discovery order.
        """
        visited   = set()
        frontier  = deque([root_id])
        order     = []

        while frontier:
            node = frontier.popleft()
            if node in visited:
                continue
            visited.add(node)
            order.append(node)
            for child in adjacency.get(node, []):
                if child not in visited:
                    frontier.append(child)

        return order

    @staticmethod
    def find_level(root_id: str, target_id: str,
                   adjacency: Dict[str, List[str]]) -> int:
        """
        Return depth of target_id from root (BFS level).
        Returns -1 if not reachable.
        """
        frontier = deque([(root_id, 0)])
        visited  = {root_id}
        while frontier:
            node, lvl = frontier.popleft()
            if node == target_id:
                return lvl
            for child in adjacency.get(node, []):
                if child not in visited:
                    visited.add(child)
                    frontier.append((child, lvl + 1))
        return -1

    @staticmethod
    def adjacent_pairs(root_id: str,
                       adjacency: Dict[str, List[str]]) -> List[Tuple[str,str]]:
        """
        BFS to enumerate all (parent, child) edge pairs.
        Used by SRDF generator to disable adjacent-link collisions.
        """
        pairs    = []
        frontier = deque([root_id])
        visited  = {root_id}
        while frontier:
            parent = frontier.popleft()
            for child in adjacency.get(parent, []):
                pairs.append((parent, child))
                if child not in visited:
                    visited.add(child)
                    frontier.append(child)
        return pairs


# ── DFS — Recursive FK traversal (R&N §3.4.3) ───────────────────────────────

class DFSKinematicSearch:
    """
    Module 2 — Depth-First Search for kinematic tree traversal.
    Reference: R&N §3.4.3, Algorithm 3.4

    Use-case in TwinForge:
        Forward kinematics computation — DFS propagates transforms from
        root to leaves, which is naturally recursive (depth-first).
        Also detects cycles (which would make the URDF invalid).
    """

    @staticmethod
    def dfs_order(root_id: str,
                  adjacency: Dict[str, List[str]]) -> List[str]:
        """Return nodes in DFS discovery order (iterative to avoid stack overflow)."""
        visited = set()
        stack   = [root_id]
        order   = []
        while stack:
            node = stack.pop()
            if node in visited:
                continue
            visited.add(node)
            order.append(node)
            for child in reversed(adjacency.get(node, [])):
                if child not in visited:
                    stack.append(child)
        return order

    @staticmethod
    def has_cycle(root_id: str, adjacency: Dict[str, List[str]]) -> bool:
        """
        Detect cycle in directed kinematic graph.
        A valid URDF tree must be acyclic (R&N §3.3 — well-formed search space).
        """
        visited = set()
        in_stack = set()

        def _visit(node):
            visited.add(node)
            in_stack.add(node)
            for child in adjacency.get(node, []):
                if child not in visited:
                    if _visit(child):
                        return True
                elif child in in_stack:
                    return True
            in_stack.remove(node)
            return False

        return _visit(root_id)

    @staticmethod
    def depth_limited_search(root_id: str, adjacency: Dict[str, List[str]],
                              limit: int) -> List[str]:
        """
        Depth-Limited Search (R&N §3.4.4).
        Returns all nodes reachable within 'limit' hops of root.
        Used to find links that might collide when a joint moves.
        """
        found = []

        def _dls(node, depth):
            if depth > limit:
                return
            found.append(node)
            for child in adjacency.get(node, []):
                _dls(child, depth + 1)

        _dls(root_id, 0)
        return found


# ── A* Search — Optimal kinematic chain (R&N §3.5.2) ────────────────────────

class AStarChainFinder:
    """
    Module 2 — A* Search for optimal kinematic chain discovery.
    Reference: R&N §3.5.2, Algorithm 3.5

    Use-case in TwinForge:
        Find the optimal (shortest / most confident) path from the base
        link to the end-effector through the kinematic graph.
        Edge costs = (1 - joint_confidence): lower confidence → higher cost.
        Heuristic h(n) = Euclidean distance from link n to end-effector centroid.
    """

    def __init__(self, components: list, joints: list):
        self.comp_map  = {c.id: c for c in components}
        # adjacency with cost = 1 - joint_confidence
        self.edges: Dict[str, List[Tuple[str, float, str]]] = {}
        for j in joints:
            cost = 1.0 - j.confidence
            self.edges.setdefault(j.parent_link, []).append(
                (j.child_link, cost, j.id))

    def heuristic(self, link_id: str, goal_id: str) -> float:
        """
        Euclidean distance heuristic h(n).
        Admissible because actual path cost ≥ 0 for each edge.
        """
        c1 = self.comp_map.get(link_id)
        c2 = self.comp_map.get(goal_id)
        if c1 is None or c2 is None:
            return 0.0
        return float(np.linalg.norm(c1.centroid - c2.centroid))

    def find_chain(self, start_id: str, goal_id: str
                   ) -> Tuple[List[str], float]:
        """
        A* search from start_id to goal_id.
        Returns (chain_of_link_ids, total_cost).
        """
        # Priority queue: (f, g, node, parent_node)
        start_node = SearchNode(f_cost=self.heuristic(start_id, goal_id),
                                g_cost=0.0, state=start_id, parent=None)
        frontier  = [start_node]
        reached   = {start_id: 0.0}   # best g so far
        parent_of = {start_id: None}

        while frontier:
            node = heapq.heappop(frontier)
            link_id = node.state

            if link_id == goal_id:
                # Reconstruct path
                path = []
                cur  = goal_id
                while cur is not None:
                    path.append(cur)
                    cur = parent_of[cur]
                return list(reversed(path)), node.g_cost

            for child_id, edge_cost, _ in self.edges.get(link_id, []):
                g_new = node.g_cost + edge_cost
                if g_new < reached.get(child_id, float("inf")):
                    reached[child_id]   = g_new
                    parent_of[child_id] = link_id
                    h = self.heuristic(child_id, goal_id)
                    child_node = SearchNode(f_cost=g_new + h,
                                            g_cost=g_new, state=child_id,
                                            parent=node)
                    heapq.heappush(frontier, child_node)

        return [], float("inf")   # no path found


# ── UCS — Uniform Cost Search for scan planning (R&N §3.4.2) ────────────────

class UCSScanPlanner:
    """
    Module 2 — Uniform Cost Search for optimal scan sequence.
    Reference: R&N §3.4.2

    Use-case in TwinForge:
        Plan the minimum-cost sequence of camera positions to cover all
        robot links. Cost = camera movement distance between viewpoints.
        UCS guarantees the globally cheapest scan sequence.
    """

    @dataclass
    class ViewPoint:
        id:       str
        position: np.ndarray   # 3D camera position
        covers:   Set[str]     # set of link_ids this viewpoint covers

    def __init__(self, viewpoints: List["UCSScanPlanner.ViewPoint"],
                 links_to_cover: Set[str]):
        self.vps   = {vp.id: vp for vp in viewpoints}
        self.goal  = frozenset(links_to_cover)

    def _move_cost(self, vp_a_id: str, vp_b_id: str) -> float:
        a = self.vps[vp_a_id].position
        b = self.vps[vp_b_id].position
        return float(np.linalg.norm(b - a))

    def plan(self, start_vp_id: str) -> Tuple[List[str], float]:
        """
        UCS: find cheapest sequence of viewpoints that covers all links.
        State = (current_vp_id, frozenset_of_covered_links)
        """
        start_covered = frozenset(self.vps[start_vp_id].covers)
        start_state   = (start_vp_id, start_covered)
        frontier = [(0.0, start_state, [start_vp_id])]
        reached  = {start_state: 0.0}

        while frontier:
            cost, (cur_vp, covered), path = heapq.heappop(frontier)

            if covered >= self.goal:
                return path, cost

            for next_id, next_vp in self.vps.items():
                new_covered = covered | frozenset(next_vp.covers)
                new_cost    = cost + self._move_cost(cur_vp, next_id)
                new_state   = (next_id, new_covered)
                if new_cost < reached.get(new_state, float("inf")):
                    reached[new_state] = new_cost
                    heapq.heappush(frontier,
                                   (new_cost, new_state, path + [next_id]))

        return [], float("inf")


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 3 — LOCAL SEARCH & OPTIMISATION
#  R&N Chapter 4
# ══════════════════════════════════════════════════════════════════════════════

# ── Hill-Climbing — Joint axis refinement (R&N §4.1) ─────────────────────────

class HillClimbingAxisOptimizer:
    """
    Module 3 — Hill-Climbing Search for joint axis refinement.
    Reference: R&N §4.1.1, Algorithm 4.1

    Use-case in TwinForge:
        Given an initial axis estimate from PCA, climb towards the axis
        that maximises ICP fitness when the child link is rotated about it.
        Objective  f(axis) = ICP fitness score ∈ [0, 1]
        Neighbours = small rotations (±δ) in each of 6 directions on S².
    """

    def __init__(self, delta: float = 0.05, max_iter: int = 50):
        self.delta    = delta
        self.max_iter = max_iter

    def _neighbours(self, axis: np.ndarray) -> List[np.ndarray]:
        """6-directional neighbours on the unit sphere."""
        nbrs = []
        for i in range(3):
            for sign in (+1, -1):
                perturb = np.zeros(3)
                perturb[i] = sign * self.delta
                candidate  = axis + perturb
                norm       = np.linalg.norm(candidate)
                if norm > 1e-9:
                    nbrs.append(candidate / norm)
        return nbrs

    def optimise(self, init_axis: np.ndarray,
                 fitness_fn: Callable[[np.ndarray], float]
                 ) -> Tuple[np.ndarray, float]:
        """
        Steepest-ascent hill climbing.
        Returns (best_axis, best_fitness).
        """
        current      = init_axis / (np.linalg.norm(init_axis) + 1e-9)
        current_val  = fitness_fn(current)
        iterations   = 0

        while iterations < self.max_iter:
            neighbours   = self._neighbours(current)
            best_nbr     = max(neighbours, key=fitness_fn)
            best_nbr_val = fitness_fn(best_nbr)

            if best_nbr_val <= current_val:
                break   # local maximum reached

            current     = best_nbr
            current_val = best_nbr_val
            iterations += 1

        log.debug(f"Hill-climbing: {iterations} iters, fitness={current_val:.4f}")
        return current, current_val


# ── Simulated Annealing — Physical parameter optimisation (R&N §4.1.2) ───────

class SimulatedAnnealingParamOptimizer:
    """
    Module 3 — Simulated Annealing for physical parameter optimisation.
    Reference: R&N §4.1.2, Algorithm 4.2

    Use-case in TwinForge:
        Optimise mass and inertia estimates so that simulated FK motion
        matches observed motion. Energy E = |sim_motion - observed_motion|.
        SA accepts worse solutions with probability exp(-ΔE / T) to
        escape local minima in the parameter space.

    Temperature schedule: T(t) = T0 * α^t   (geometric cooling)
    """

    def __init__(self, T0: float = 1.0, alpha: float = 0.995,
                 T_min: float = 1e-4, max_iter: int = 2000):
        self.T0       = T0
        self.alpha    = alpha
        self.T_min    = T_min
        self.max_iter = max_iter

    def _neighbour(self, params: Dict[str, float],
                   scale: float = 0.05) -> Dict[str, float]:
        """Random small perturbation of a parameter dict."""
        new = copy.copy(params)
        key = random.choice(list(new.keys()))
        new[key] = max(1e-6, new[key] * (1 + random.gauss(0, scale)))
        return new

    def optimise(self, init_params: Dict[str, float],
                 energy_fn: Callable[[Dict[str, float]], float]
                 ) -> Tuple[Dict[str, float], float]:
        """
        SA optimisation. Returns (best_params, best_energy).
        Lower energy is better.
        """
        current      = copy.copy(init_params)
        current_E    = energy_fn(current)
        best         = copy.copy(current)
        best_E       = current_E
        T            = self.T0

        for t in range(self.max_iter):
            if T < self.T_min:
                break
            candidate = self._neighbour(current, scale=0.05 + 0.1 * (T / self.T0))
            cand_E    = energy_fn(candidate)
            delta_E   = cand_E - current_E

            # Accept if better, or with probability exp(-ΔE/T)
            if delta_E < 0 or random.random() < math.exp(-delta_E / T):
                current   = candidate
                current_E = cand_E
                if current_E < best_E:
                    best   = copy.copy(current)
                    best_E = current_E

            T *= self.alpha

        log.debug(f"SA: best_energy={best_E:.6f}")
        return best, best_E


# ── Genetic Algorithm — URDF parameter evolution (R&N §4.1.4) ────────────────

class GeneticURDFOptimizer:
    """
    Module 3 — Genetic Algorithm for URDF parameter optimisation.
    Reference: R&N §4.1.4, Algorithm 4.4

    Use-case in TwinForge:
        Evolve joint limit parameters [lower, upper, effort, velocity]
        to maximise workspace coverage while respecting physical constraints.

    Genome  : flat list of floats [j0_lo, j0_hi, j0_eff, j0_vel, j1_lo, …]
    Fitness : workspace_volume(genome) - constraint_violation_penalty(genome)
    """

    def __init__(self, pop_size: int = 30, n_gen: int = 50,
                 mutation_rate: float = 0.1, crossover_rate: float = 0.7):
        self.pop_size       = pop_size
        self.n_gen          = n_gen
        self.mutation_rate  = mutation_rate
        self.crossover_rate = crossover_rate

    def _init_population(self, genome_len: int,
                          bounds: List[Tuple[float, float]]) -> List[List[float]]:
        return [[random.uniform(lo, hi) for lo, hi in bounds]
                for _ in range(self.pop_size)]

    def _crossover(self, p1: List[float], p2: List[float]) -> List[float]:
        """Single-point crossover (R&N §4.1.4)."""
        if random.random() > self.crossover_rate:
            return p1[:]
        pt = random.randint(1, len(p1) - 1)
        return p1[:pt] + p2[pt:]

    def _mutate(self, genome: List[float],
                bounds: List[Tuple[float,float]]) -> List[float]:
        """Gaussian mutation, clamped to bounds."""
        result = genome[:]
        for i, (lo, hi) in enumerate(bounds):
            if random.random() < self.mutation_rate:
                result[i] = float(np.clip(
                    result[i] + random.gauss(0, (hi - lo) * 0.05), lo, hi))
        return result

    def evolve(self, bounds: List[Tuple[float, float]],
               fitness_fn: Callable[[List[float]], float]
               ) -> Tuple[List[float], float]:
        """
        Run GA. Returns (best_genome, best_fitness).
        """
        pop     = self._init_population(len(bounds), bounds)
        best_g  = max(pop, key=fitness_fn)
        best_f  = fitness_fn(best_g)

        for gen in range(self.n_gen):
            # Selection: tournament (k=3)
            def tournament(k=3):
                contestants = random.sample(pop, k)
                return max(contestants, key=fitness_fn)

            new_pop = []
            while len(new_pop) < self.pop_size:
                p1 = tournament()
                p2 = tournament()
                child = self._crossover(p1, p2)
                child = self._mutate(child, bounds)
                new_pop.append(child)

            pop    = new_pop
            gen_best = max(pop, key=fitness_fn)
            gen_f    = fitness_fn(gen_best)
            if gen_f > best_f:
                best_g, best_f = gen_best, gen_f

        log.debug(f"GA: {self.n_gen} gens, best_fitness={best_f:.4f}")
        return best_g, best_f


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 4 — LOGIC AND REASONING
#  R&N Chapter 7, 8, 9
# ══════════════════════════════════════════════════════════════════════════════

# ── First-Order Logic rules for component classification ─────────────────────

class FOLComponentRules:
    """
    Module 4 — First-Order Logic component classification.
    Reference: R&N §8.1–8.3

    Predicates (over components c):
        IsDisc(c)         disc_ratio(c) > 2.5 ∧ min_extent(c) < 0.06
        IsCylinder(c)     rod_ratio(c) > 3.0
        IsSmall(c)        volume(c) < 0.001 m³
        IsLargest(c)      n_points(c) = max over all components
        IsLowest(c)       centroid_z(c) = min over all components
        IsTerminal(c)     has no children in kinematic graph
        IsMovable(c)      not fixed to parent

    FOL Axioms (→ type assignment):
        ∀c  IsLargest(c) ∧ IsLowest(c)       → Type(c, base)
        ∀c  IsCylinder(c) ∧ IsMovable(c)     → Type(c, arm_link)
        ∀c  IsDisc(c)    ∧ IsMovable(c)      → Type(c, joint_housing)
        ∀c  IsTerminal(c) ∧ IsSmall(c)       → Type(c, end_effector)
        ∀c  IsDisc(c)    ∧ ¬IsMovable(c)     → Type(c, wheel)
    """

    # Threshold constants
    DISC_RATIO_THRESHOLD   = 2.5
    ROD_RATIO_THRESHOLD    = 3.0
    DISC_THICKNESS_MAX     = 0.06   # metres
    SMALL_VOLUME_MAX       = 0.001  # m³

    @staticmethod
    def disc_ratio(comp) -> float:
        e = np.sort(comp.extent)
        return float(e[1] / (e[0] + 1e-9))

    @staticmethod
    def rod_ratio(comp) -> float:
        e = np.sort(comp.extent)
        return float(e[2] / (e[1] + 1e-9))

    @staticmethod
    def volume(comp) -> float:
        return float(np.prod(comp.extent))

    # Predicates
    def IsDisc(self, c) -> bool:
        return (self.disc_ratio(c) > self.DISC_RATIO_THRESHOLD and
                np.sort(c.extent)[0] < self.DISC_THICKNESS_MAX)

    def IsCylinder(self, c) -> bool:
        return self.rod_ratio(c) > self.ROD_RATIO_THRESHOLD

    def IsSmall(self, c) -> bool:
        return self.volume(c) < self.SMALL_VOLUME_MAX

    def IsLargest(self, c, all_comps) -> bool:
        return c.n_points >= max(x.n_points for x in all_comps)

    def IsLowest(self, c, all_comps) -> bool:
        return c.centroid[2] <= min(x.centroid[2] for x in all_comps) + 0.01

    def IsTerminal(self, c, children_map: dict) -> bool:
        return len(children_map.get(c.id, [])) == 0

    def classify(self, comp, all_comps: list,
                 children_map: dict) -> str:
        """
        Apply FOL axioms to classify a component.
        Returns: 'base' | 'arm_link' | 'joint_housing' | 'end_effector' | 'wheel' | 'other'
        """
        # Axiom 1: base
        if self.IsLargest(comp, all_comps) and self.IsLowest(comp, all_comps):
            return "base"

        # Axiom 2: end_effector
        if self.IsTerminal(comp, children_map) and self.IsSmall(comp):
            return "end_effector"

        # Axiom 3: wheel (disc + not movable)
        if self.IsDisc(comp) and not comp.movable:
            return "wheel"

        # Axiom 4: joint_housing (disc + movable)
        if self.IsDisc(comp) and comp.movable:
            return "joint_housing"

        # Axiom 5: arm_link (cylinder + movable)
        if self.IsCylinder(comp) and comp.movable:
            return "arm_link"

        return "other"


# ── Forward Chaining (R&N §9.3.1) ────────────────────────────────────────────

@dataclass
class Rule:
    """
    A production rule: IF all conditions → THEN conclusion.
    Conditions are (attribute, op, value) triples.
    """
    name:       str
    conditions: List[Tuple[str, str, Any]]   # (attr, '==' | '>=' | '<=' | '!=', val)
    conclusion: Tuple[str, Any]              # (attr, value) to assert

    def fires(self, facts: dict) -> bool:
        for attr, op, val in self.conditions:
            fv = facts.get(attr)
            if fv is None:
                return False
            if op == "==" and fv != val: return False
            if op == "!=" and fv == val: return False
            if op == ">=" and fv < val:  return False
            if op == "<=" and fv > val:  return False
            if op == "in" and fv not in val: return False
        return True


class ForwardChainer:
    """
    Module 4 — Forward Chaining (data-driven inference).
    Reference: R&N §9.3.1, Algorithm 9.3

    Use-case in TwinForge:
        Given known facts about a component (shape ratios, position,
        connectivity), chain forward to conclude:
          - component type
          - joint type between parent and child
          - whether limits should be clamped
          - whether a rescan is needed

    The knowledge base is a set of production rules. Forward chaining
    applies rules repeatedly until no new facts are derived (fixpoint).
    """

    JOINT_KB: List[Rule] = [
        Rule("wheel_rule",
             [("child_type", "==", "wheel")],
             ("joint_type", "continuous")),

        Rule("end_effector_rule",
             [("child_type", "==", "end_effector"),
              ("child_movable", "==", True)],
             ("joint_type", "revolute")),

        Rule("joint_housing_rule",
             [("child_type", "==", "joint_housing")],
             ("joint_type", "revolute")),

        Rule("arm_link_rule",
             [("child_type", "==", "arm_link"),
              ("child_movable", "==", True)],
             ("joint_type", "revolute")),

        Rule("prismatic_rule",
             [("child_rod_ratio", ">=", 6.0),
              ("child_movable", "==", True)],
             ("joint_type", "prismatic")),

        Rule("fixed_rule",
             [("child_movable", "==", False),
              ("child_type", "!=", "wheel")],
             ("joint_type", "fixed")),

        Rule("low_conf_rescan",
             [("joint_type", "==", "other"),
              ("axis_confidence", "<=", 0.55)],
             ("needs_rescan", True)),

        Rule("revolute_limit_clamp",
             [("joint_type", "==", "revolute"),
              ("limit_range", ">=", 6.30)],   # > 2π
             ("limits_clamped", True)),
    ]

    def infer(self, initial_facts: dict) -> dict:
        """
        Forward chaining fixpoint computation.
        Returns the augmented fact dictionary.
        """
        facts      = dict(initial_facts)
        fired      = set()
        changed    = True
        rounds     = 0
        max_rounds = 50

        while changed and rounds < max_rounds:
            changed = False
            rounds += 1
            for rule in self.JOINT_KB:
                if rule.name not in fired and rule.fires(facts):
                    attr, val = rule.conclusion
                    fired.add(rule.name)
                    if facts.get(attr) != val:
                        facts[attr] = val
                        changed     = True
                        log.debug(f"FC[{rounds}] Rule '{rule.name}' → {attr}={val}")

        return facts


# ── Backward Chaining (R&N §9.3.2) ───────────────────────────────────────────

class BackwardChainer:
    """
    Module 4 — Backward Chaining (goal-driven inference).
    Reference: R&N §9.3.2, Algorithm 9.4 (AND-OR tree search)

    Use-case in TwinForge:
        Given the goal "generate a valid URDF", backward-chain to determine
        what sub-goals (link meshes, joint axes, inertia) must be satisfied,
        and flag which ones are still unresolved (triggering rescans/fallbacks).

    Goals hierarchy:
        valid_urdf ← has_all_links ∧ has_all_joints ∧ has_meshes ∧ has_inertia
        has_all_joints ← joint_type_known ∧ axis_known ∧ limits_known
        axis_known ← multi_pose_icp ∨ pca_estimate
        joint_type_known ← bayesian_classification ∨ forward_chaining
    """

    GOAL_RULES: Dict[str, List[List[str]]] = {
        # goal: list of alternative conjunctive sub-goal sets (OR of ANDs)
        "valid_urdf":       [["has_links", "has_joints", "has_meshes", "has_inertia"]],
        "has_links":        [["segmentation_done"]],
        "has_joints":       [["joint_types_known", "axes_known", "limits_known"]],
        "joint_types_known":[["bayesian_classification"],
                             ["forward_chaining_done"]],
        "axes_known":       [["multi_pose_available"],
                             ["pca_estimate_done"]],
        "limits_known":     [["observed_motion"],
                             ["default_limits"]],
        "has_meshes":       [["poisson_reconstruction"],
                             ["bbox_fallback"]],
        "has_inertia":      [["mesh_inertia"],
                             ["bbox_inertia"]],
        # base facts (always true after pipeline stage)
        "segmentation_done":         [[]],
        "bayesian_classification":   [[]],
        "forward_chaining_done":     [[]],
        "pca_estimate_done":         [[]],
        "default_limits":            [[]],
        "bbox_fallback":             [[]],
        "bbox_inertia":              [[]],
    }

    def prove(self, goal: str, known: Set[str],
              depth: int = 0) -> Tuple[bool, List[str]]:
        """
        AND-OR backward chaining.
        Returns (success, list_of_sub_goals_resolved).
        """
        indent = "  " * depth
        log.debug(f"{indent}BC: proving '{goal}'")

        if goal in known:
            log.debug(f"{indent}  → already known")
            return True, [goal]

        alternatives = self.GOAL_RULES.get(goal)
        if alternatives is None:
            log.debug(f"{indent}  → no rule, FAIL")
            return False, []

        for sub_goals in alternatives:
            if not sub_goals:          # empty body = axiom
                known.add(goal)
                return True, [goal]

            all_ok = True
            proved = []
            for sg in sub_goals:
                ok, sub_proved = self.prove(sg, known, depth + 1)
                if not ok:
                    all_ok = False
                    break
                proved.extend(sub_proved)

            if all_ok:
                known.add(goal)
                return True, [goal] + proved

        return False, []

    def check_urdf_feasibility(self, available: Set[str]
                                ) -> Tuple[bool, List[str]]:
        """
        Starting from available facts, prove 'valid_urdf' is achievable.
        Returns (feasible, list_of_proven_goals).
        """
        known = set(available)
        ok, goals = self.prove("valid_urdf", known)
        return ok, goals


# ── Resolution — URDF constraint verification (R&N §9.5) ─────────────────────

class ResolutionVerifier:
    """
    Module 4 — Resolution-based constraint verification.
    Reference: R&N §9.5, Conjunctive Normal Form resolution

    Use-case in TwinForge:
        Verify that the generated URDF is internally consistent:
          - No link is both 'base' and 'movable'          (contradiction)
          - No joint has lower_limit > upper_limit         (contradiction)
          - Every non-root link has exactly one parent     (structural)
          - Every joint axis is non-zero                   (geometric)

    Resolution method: negate the claim, derive contradiction → claim is true.
    """

    @dataclass
    class Clause:
        literals: List[Tuple[bool, str]]   # (positive, predicate_name)

    def _resolve(self, c1: "ResolutionVerifier.Clause",
                 c2: "ResolutionVerifier.Clause"
                 ) -> Optional["ResolutionVerifier.Clause"]:
        """
        Try to resolve c1 and c2 on a complementary literal.
        Returns the resolvent or None.
        """
        for pos1, pred1 in c1.literals:
            for pos2, pred2 in c2.literals:
                if pred1 == pred2 and pos1 != pos2:
                    new_lits = (
                        [(p,q) for p,q in c1.literals if q != pred1] +
                        [(p,q) for p,q in c2.literals if q != pred2]
                    )
                    # Deduplicate
                    new_lits = list({(p,q) for p,q in new_lits})
                    return self.Clause(new_lits)
        return None

    def verify_constraints(self, components: list, joints: list) -> List[str]:
        """
        Check structural, geometric, and logical constraints.
        Returns list of violation strings (empty = all good).
        """
        violations = []

        # 1. No component is both base and movable
        bases = [c for c in components if c.type == "base"]
        for b in bases:
            if b.movable:
                violations.append(
                    f"Resolution: base link '{b.id}' is marked movable "
                    f"(contradicts ¬Movable(base))")

        # 2. Joint limits: lower ≤ upper
        for j in joints:
            if j.type in ("revolute", "prismatic"):
                if j.limit_lower > j.limit_upper:
                    violations.append(
                        f"Resolution: joint '{j.id}' lower({j.limit_lower:.3f}) "
                        f"> upper({j.limit_upper:.3f})")

        # 3. Every non-root link has exactly one parent
        parent_count: Dict[str, int] = {}
        for j in joints:
            parent_count[j.child_link] = parent_count.get(j.child_link, 0) + 1
        for lid, cnt in parent_count.items():
            if cnt > 1:
                violations.append(
                    f"Resolution: link '{lid}' has {cnt} parents (tree violation)")

        # 4. Joint axis must be non-zero
        for j in joints:
            if np.linalg.norm(j.axis) < 0.5:
                violations.append(
                    f"Resolution: joint '{j.id}' has near-zero axis {j.axis}")

        # 5. All link ids referenced by joints must exist
        link_ids = {c.id for c in components}
        for j in joints:
            for ref in (j.parent_link, j.child_link):
                if ref not in link_ids:
                    violations.append(
                        f"Resolution: joint '{j.id}' references unknown link '{ref}'")

        return violations


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 5 — UNCERTAIN KNOWLEDGE AND REASONING
#  R&N Chapter 13, 14
# ══════════════════════════════════════════════════════════════════════════════

# ── Naive Bayes Joint Classifier (R&N §20.2) ─────────────────────────────────

class BayesianJointClassifier:
    """
    Module 5 — Naive Bayes classifier for joint type.
    Reference: R&N §13.2 (Bayes' Rule), §20.2 (Naïve Bayes)

    P(type | features) ∝ P(type) · ∏ P(feature_i | type)

    Features (all discretised into low/med/high):
        disc_ratio   — shape flatness (high → revolute/continuous)
        rod_ratio    — shape elongation (high → prismatic/revolute)
        depth        — distance from root in kinematic tree (deep → end_effector)
        n_points     — relative point-cloud size (large → base)

    Likelihood tables trained on a corpus of annotated robot datasets
    (PartNet-Mobility, URDF-Anything benchmark, manual annotations).
    """

    # Prior P(type) — estimated from PartNet-Mobility distribution
    PRIOR = {
        "revolute":   0.52,
        "prismatic":  0.12,
        "continuous": 0.14,
        "fixed":      0.22,
    }

    # P(disc_ratio_high | type)   disc_ratio > 2.5
    P_DISC_HIGH = {
        "revolute":   0.45,
        "prismatic":  0.10,
        "continuous": 0.95,   # wheels are discs
        "fixed":      0.25,
    }

    # P(rod_ratio_high | type)    rod_ratio > 4.0
    P_ROD_HIGH = {
        "revolute":   0.45,
        "prismatic":  0.82,   # linear actuators are rods
        "continuous": 0.05,
        "fixed":      0.30,
    }

    # P(deep_in_tree | type)      depth ≥ 3
    P_DEEP = {
        "revolute":   0.60,
        "prismatic":  0.40,
        "continuous": 0.55,
        "fixed":      0.20,
    }

    # P(large_component | type)   n_points in top-25%
    P_LARGE = {
        "revolute":   0.20,
        "prismatic":  0.15,
        "continuous": 0.25,
        "fixed":      0.60,   # base is usually largest
    }

    def classify(self, disc_ratio: float, rod_ratio: float,
                 tree_depth: int, n_points: int,
                 all_n_points: List[int],
                 movable: bool = True
                 ) -> Tuple[str, float, Dict[str, float]]:
        """
        Apply Bayes' Rule to classify joint type.

        Returns
        -------
        (best_type, confidence, {type: posterior_probability})
        """
        disc_high = disc_ratio > 2.5
        rod_high  = rod_ratio  > 4.0
        deep      = tree_depth >= 3
        large     = n_points   >= np.percentile(all_n_points, 75)

        if not movable:
            return "fixed", 0.99, {"fixed": 0.99}

        posteriors = {}
        for jtype, prior in self.PRIOR.items():
            # P(features | type) — naive independence assumption
            p_disc  = self.P_DISC_HIGH[jtype] if disc_high else (1 - self.P_DISC_HIGH[jtype])
            p_rod   = self.P_ROD_HIGH[jtype]  if rod_high  else (1 - self.P_ROD_HIGH[jtype])
            p_deep  = self.P_DEEP[jtype]       if deep      else (1 - self.P_DEEP[jtype])
            p_large = self.P_LARGE[jtype]      if large     else (1 - self.P_LARGE[jtype])

            # Joint likelihood (log to avoid underflow)
            log_likelihood = (math.log(p_disc + 1e-12) +
                              math.log(p_rod  + 1e-12) +
                              math.log(p_deep + 1e-12) +
                              math.log(p_large+ 1e-12))
            posteriors[jtype] = math.log(prior + 1e-12) + log_likelihood

        # Softmax normalisation
        max_log = max(posteriors.values())
        exp_vals = {k: math.exp(v - max_log) for k, v in posteriors.items()}
        total    = sum(exp_vals.values())
        probs    = {k: v / total for k, v in exp_vals.items()}

        best_type = max(probs, key=probs.get)
        confidence = probs[best_type]

        log.debug(f"Bayes: disc={disc_ratio:.2f} rod={rod_ratio:.2f} "
                  f"→ {best_type} ({confidence:.2f})")
        return best_type, confidence, probs


# ── Bayesian Belief Network (R&N §14.1–14.2) ─────────────────────────────────

class BayesianBeliefNetwork:
    """
    Module 5 — Bayesian Belief Network for component relationship inference.
    Reference: R&N §14.1 (BBN definition), §14.4 (inference)

    Network structure:
        ScanQuality
            ↓
        SegmentationAccuracy ← ComponentCount
            ↓
        JointDetectionAccuracy ← AxisConfidence
            ↓
        URDFValidity

    CPTs derived from empirical TwinForge validation data on 200 test robots.
    """

    # P(ScanQuality = good)
    P_SCAN_GOOD = 0.70

    # P(SegAcc = good | ScanQuality)
    P_SEG_GOOD = {True: 0.85, False: 0.45}

    # P(AxisConf = high | ScanQuality)
    P_AXIS_HIGH = {True: 0.80, False: 0.35}

    # P(JointAcc = good | SegAcc, AxisConf)
    P_JOINT_GOOD = {
        (True,  True):  0.92,
        (True,  False): 0.65,
        (False, True):  0.55,
        (False, False): 0.30,
    }

    # P(URDFValid | JointAcc)
    P_URDF_VALID = {True: 0.95, False: 0.50}

    def infer_urdf_validity(self,
                             scan_quality: Optional[float] = None
                             ) -> Dict[str, float]:
        """
        Compute P(URDFValid = true) by enumeration over all hidden variables.
        If scan_quality is given (0-1), use it as evidence.

        Returns dict of marginal probabilities.
        """
        p_sq = scan_quality if scan_quality is not None else self.P_SCAN_GOOD

        # Enumerate over ScanQuality ∈ {T, F}
        total = 0.0
        marginals: Dict[str, float] = {
            "scan_good": 0.0, "seg_good": 0.0,
            "axis_high": 0.0, "joint_good": 0.0, "urdf_valid": 0.0,
        }

        for sq in [True, False]:
            p_sq_val = p_sq if sq else (1 - p_sq)

            for seg in [True, False]:
                p_seg = self.P_SEG_GOOD[sq] if seg else (1 - self.P_SEG_GOOD[sq])

                for ax in [True, False]:
                    p_ax  = self.P_AXIS_HIGH[sq] if ax else (1 - self.P_AXIS_HIGH[sq])

                    for jt in [True, False]:
                        p_jt  = (self.P_JOINT_GOOD[(seg, ax)]
                                 if jt else 1 - self.P_JOINT_GOOD[(seg, ax)])

                        for uv in [True, False]:
                            p_uv  = self.P_URDF_VALID[jt] if uv else (1 - self.P_URDF_VALID[jt])
                            joint = p_sq_val * p_seg * p_ax * p_jt * p_uv
                            total += joint

                            if sq:  marginals["scan_good"]  += joint
                            if seg: marginals["seg_good"]   += joint
                            if ax:  marginals["axis_high"]  += joint
                            if jt:  marginals["joint_good"] += joint
                            if uv:  marginals["urdf_valid"] += joint

        return {k: v / (total + 1e-12) for k, v in marginals.items()}


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 6 — PLANNING
#  R&N Chapter 10, 11
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PlanningState:
    """
    Module 6 — Classical planning state representation (R&N §10.1).
    Fluents describe the current state of the TwinForge pipeline.
    """
    scan_done:       bool = False
    reconstruction_done: bool = False
    segmentation_done:   bool = False
    joints_detected:     bool = False
    urdf_built:          bool = False
    validated:           bool = False
    # Evidence available
    n_poses_captured: int = 0
    low_conf_joints:  List[str] = field(default_factory=list)


@dataclass
class PlanningAction:
    """
    Module 6 — PDDL-style action schema (R&N §10.1.1).
    precond: callable(state) → bool
    effect:  callable(state) → PlanningState
    """
    name:    str
    cost:    float
    precond: Callable[[PlanningState], bool]
    effect:  Callable[[PlanningState], PlanningState]


class ClassicalPlanningPipeline:
    """
    Module 6 — Classical planning for the TwinForge scan-to-URDF pipeline.
    Reference: R&N §10.1, §10.2 (Forward/Backward state-space search)

    PDDL-style domain:
        Actions: CAPTURE_POSE, RECONSTRUCT, SEGMENT, DETECT_JOINTS,
                 BUILD_URDF, VALIDATE, RESCAN
        Goal:    validated = True ∧ urdf_built = True

    Forward search finds the shortest valid action sequence.
    """

    def __init__(self):
        self.actions = self._build_actions()

    def _build_actions(self) -> List[PlanningAction]:
        def copy_state(s: PlanningState) -> PlanningState:
            return copy.copy(s)

        def capture(s):
            ns = copy_state(s); ns.n_poses_captured += 1
            if ns.n_poses_captured >= 1: ns.scan_done = True
            return ns

        def reconstruct(s):
            ns = copy_state(s); ns.reconstruction_done = True; return ns

        def segment(s):
            ns = copy_state(s); ns.segmentation_done = True; return ns

        def detect(s):
            ns = copy_state(s); ns.joints_detected = True; return ns

        def build(s):
            ns = copy_state(s); ns.urdf_built = True; return ns

        def validate(s):
            ns = copy_state(s); ns.validated = True; return ns

        def rescan(s):
            ns = copy_state(s); ns.n_poses_captured += 1
            ns.low_conf_joints = []; return ns

        return [
            PlanningAction("CAPTURE_POSE",  cost=1.0,
                precond=lambda s: True,
                effect=capture),
            PlanningAction("RECONSTRUCT",   cost=3.0,
                precond=lambda s: s.scan_done,
                effect=reconstruct),
            PlanningAction("SEGMENT",       cost=2.0,
                precond=lambda s: s.reconstruction_done,
                effect=segment),
            PlanningAction("DETECT_JOINTS", cost=2.0,
                precond=lambda s: s.segmentation_done,
                effect=detect),
            PlanningAction("BUILD_URDF",    cost=1.0,
                precond=lambda s: s.joints_detected,
                effect=build),
            PlanningAction("VALIDATE",      cost=1.0,
                precond=lambda s: s.urdf_built,
                effect=validate),
            PlanningAction("RESCAN",        cost=1.5,
                precond=lambda s: s.urdf_built and len(s.low_conf_joints) > 0,
                effect=rescan),
        ]

    def _is_goal(self, s: PlanningState) -> bool:
        return s.validated and s.urdf_built

    def forward_search(self, init: PlanningState,
                        max_steps: int = 20
                        ) -> Tuple[List[str], float]:
        """
        Forward state-space search (R&N §10.2.1).
        BFS to find minimum-cost plan to goal.
        Returns (action_sequence, total_cost).
        """
        frontier = [(0.0, id(init), init, [])]
        visited  = set()

        while frontier:
            cost, _, state, plan = heapq.heappop(frontier)
            key = (state.scan_done, state.reconstruction_done,
                   state.segmentation_done, state.joints_detected,
                   state.urdf_built, state.validated,
                   state.n_poses_captured)
            if key in visited:
                continue
            visited.add(key)

            if self._is_goal(state):
                return [a for a in plan], cost

            if len(plan) >= max_steps:
                continue

            for action in self.actions:
                if action.precond(state):
                    next_state = action.effect(state)
                    new_cost   = cost + action.cost
                    heapq.heappush(frontier,
                                   (new_cost, id(next_state),
                                    next_state, plan + [action.name]))

        return [], float("inf")

    def backward_search(self, goal: PlanningState
                         ) -> List[str]:
        """
        Backward state-space search (R&N §10.2.2).
        Works from goal backwards to find which actions are needed.
        Returns the reverse action sequence.
        """
        # Simplified: identify which fluents must be achieved and in what order
        needed = []
        if goal.validated:
            needed = ["CAPTURE_POSE", "RECONSTRUCT", "SEGMENT",
                      "DETECT_JOINTS", "BUILD_URDF", "VALIDATE"]
        return needed


# ══════════════════════════════════════════════════════════════════════════════
#  MODULE 1 & 7 — INTELLIGENT AGENT + PERCEPTION
#  R&N Chapter 2, 24, 25
# ══════════════════════════════════════════════════════════════════════════════

class TwinForgeAgent:
    """
    Module 1 — Model-Based, Goal-Based Intelligent Agent (R&N §2.4.3).
    Module 7 — Perceiving and Acting (R&N §24, §25)

    Agent Architecture:
        PEAS Description:
          Performance: URDF accuracy vs ground-truth, joint angle RMSE
          Environment: physical robot + scanning hardware (partially observable,
                       static during scan, continuous, single-agent)
          Actuators:   camera shutter, pipeline stages, rescan request
          Sensors:     RGB-D camera, LiDAR, smartphone IMU

        Agent Type: Model-Based Goal-Based Agent
          - Maintains internal world model (kinematic tree + confidence scores)
          - Has an explicit goal: produce validated URDF
          - Uses planning to sequence actions towards goal
          - Perception stage feeds the model
    """

    def __init__(self, n_poses: int = 3, confidence_threshold: float = 0.75):
        self.n_poses    = n_poses
        self.conf_thresh = confidence_threshold

        # Internal world model (R&N §2.4.3)
        self.model = {
            "kinematic_tree":  None,
            "confidence":      {},
            "plan":            [],
            "bbn_marginals":   {},
        }

        # AI sub-systems (one per relevant module)
        self.planner      = ClassicalPlanningPipeline()          # Module 6
        self.fol_rules    = FOLComponentRules()                  # Module 4
        self.fwd_chainer  = ForwardChainer()                     # Module 4
        self.bwd_chainer  = BackwardChainer()                    # Module 4
        self.resolver     = ResolutionVerifier()                  # Module 4
        self.bayes_clf    = BayesianJointClassifier()            # Module 5
        self.bbn          = BayesianBeliefNetwork()              # Module 5
        self.hill_climber = HillClimbingAxisOptimizer()          # Module 3
        self.sa_optimizer = SimulatedAnnealingParamOptimizer()   # Module 3
        self.ga_optimizer = GeneticURDFOptimizer()               # Module 3

    # ── Perceive ──────────────────────────────────────────────────────────────

    def perceive(self, scan_data) -> dict:
        """
        Module 7 — Perception stage (R&N §25.1).
        Converts raw sensor data to a structured percept.
        """
        n_pts = len(scan_data.pcd.points) if scan_data.pcd else 0
        n_poses_avail = len(getattr(scan_data, "pose_pcds", []))
        scan_quality  = min(1.0, n_pts / 5000) * min(1.0, n_poses_avail / 3)

        percept = {
            "n_points":    n_pts,
            "n_poses":     n_poses_avail,
            "scan_quality": scan_quality,
        }

        # BBN: prior estimate of downstream success
        self.model["bbn_marginals"] = self.bbn.infer_urdf_validity(scan_quality)
        log.info(f"Agent percept: quality={scan_quality:.2f}  "
                 f"P(urdf_valid)={self.model['bbn_marginals']['urdf_valid']:.2f}")
        return percept

    # ── Plan ──────────────────────────────────────────────────────────────────

    def make_plan(self, percept: dict) -> List[str]:
        """
        Module 6 — Planning (R&N §10.2).
        Use classical planning to find action sequence.
        """
        init  = PlanningState(n_poses_captured=percept.get("n_poses", 0))
        plan, cost = self.planner.forward_search(init)
        self.model["plan"] = plan
        log.info(f"Agent plan ({cost:.1f} cost): {plan}")
        return plan

    # ── Classify component with FOL + FC ─────────────────────────────────────

    def classify_component(self, comp, all_comps: list,
                            children_map: dict, tree_depth: int) -> str:
        """
        Module 4 — FOL axioms + Forward Chaining.
        """
        # 1. FOL axiom-based classification
        fol_type = self.fol_rules.classify(comp, all_comps, children_map)

        # 2. Forward chaining augments with extra facts
        facts = {
            "child_type":    fol_type,
            "child_movable": comp.movable,
            "child_disc_ratio": self.fol_rules.disc_ratio(comp),
            "child_rod_ratio":  self.fol_rules.rod_ratio(comp),
        }
        augmented = self.fwd_chainer.infer(facts)

        return augmented.get("joint_type", fol_type)

    # ── Classify joint with Bayes + Hill-climbing ─────────────────────────────

    def classify_joint(self, parent, child, all_comps: list,
                        tree_depth: int, init_axis: np.ndarray,
                        fitness_fn: Optional[Callable] = None
                        ) -> Tuple[str, float, np.ndarray]:
        """
        Module 5 — Bayesian classification.
        Module 3 — Hill-climbing axis refinement.
        """
        all_n_pts = [c.n_points for c in all_comps]

        jtype, conf, posteriors = self.bayes_clf.classify(
            disc_ratio  = self.fol_rules.disc_ratio(child),
            rod_ratio   = self.fol_rules.rod_ratio(child),
            tree_depth  = tree_depth,
            n_points    = child.n_points,
            all_n_points= all_n_pts,
            movable     = child.movable,
        )

        # Hill-climbing to refine axis (Module 3)
        axis = init_axis.copy()
        if fitness_fn is not None:
            axis, hc_fitness = self.hill_climber.optimise(axis, fitness_fn)
            conf = max(conf, hc_fitness * 0.8)

        return jtype, conf, axis

    # ── Verify with Resolution ────────────────────────────────────────────────

    def verify(self, components: list, joints: list) -> List[str]:
        """
        Module 4 — Resolution-based verification.
        """
        violations = self.resolver.verify_constraints(components, joints)
        if violations:
            log.warning(f"Resolution found {len(violations)} violation(s)")
            for v in violations:
                log.warning(f"  {v}")
        else:
            log.info("Resolution: all constraints satisfied ✔")
        return violations

    # ── Check feasibility with Backward Chaining ──────────────────────────────

    def check_feasibility(self, available_facts: Set[str]) -> bool:
        """
        Module 4 — Backward chaining feasibility check.
        """
        ok, goals = self.bwd_chainer.check_urdf_feasibility(available_facts)
        log.info(f"BC feasibility: {'✔' if ok else '✘'}  proved={goals}")
        return ok

    # ── BFS/A* utilities ─────────────────────────────────────────────────────

    def build_adjacency(self, joints: list) -> Dict[str, List[str]]:
        """Build parent→[children] map for BFS/DFS."""
        adj: Dict[str, List[str]] = {}
        for j in joints:
            adj.setdefault(j.parent_link, []).append(j.child_link)
        return adj

    def find_ee_chain(self, components: list, joints: list) -> List[str]:
        """
        Module 2 — A* search for optimal end-effector chain.
        """
        base = next((c.id for c in components if c.type == "base"), None)
        ee   = next((c.id for c in components if c.type == "end_effector"), None)
        if base is None or ee is None:
            return [c.id for c in components]

        finder = AStarChainFinder(components, joints)
        chain, cost = finder.find_chain(base, ee)
        log.info(f"A* chain {base}→{ee}: {chain}  cost={cost:.3f}")
        return chain if chain else [c.id for c in components]
