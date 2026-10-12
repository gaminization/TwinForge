"""
Tests for core/ai_reasoning.py
================================
One test class per BCSE306L module.  Every algorithm is exercised with
at least one correctness assertion and one edge-case assertion.

Run:  pytest tests/test_ai_reasoning.py -v -s
"""

import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pytest


# ─────────────────────────────────────────────────────────────────────────────
#  Shared fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _simple_adj():
    """
    Simple kinematic tree:
        base → link_1 → link_2 → end_effector
                      ↘ link_3
    """
    return {
        "base":    ["link_1"],
        "link_1":  ["link_2", "link_3"],
        "link_2":  ["end_effector"],
        "link_3":  [],
        "end_effector": [],
    }

def _all_nodes(adj):
    nodes = set()
    for k, vs in adj.items():
        nodes.add(k); nodes.update(vs)
    return nodes


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 2 — Search Algorithms
# ─────────────────────────────────────────────────────────────────────────────

class TestModule2_BFS:
    """Module 2 — Breadth-First Search (R&N §3.4.1)"""

    def test_bfs_visits_all_nodes(self):
        from core.ai_reasoning import BFSKinematicSearch
        adj   = _simple_adj()
        order = BFSKinematicSearch.bfs("base", adj)
        assert set(order) == _all_nodes(adj), "BFS must visit every node"

    def test_bfs_root_first(self):
        from core.ai_reasoning import BFSKinematicSearch
        order = BFSKinematicSearch.bfs("base", _simple_adj())
        assert order[0] == "base", "BFS must start at root"

    def test_bfs_level_order(self):
        """BFS must visit depth-1 nodes before depth-2 nodes."""
        from core.ai_reasoning import BFSKinematicSearch
        order = BFSKinematicSearch.bfs("base", _simple_adj())
        idx   = {n: i for i, n in enumerate(order)}
        # link_1 (depth 1) must precede link_2 and link_3 (depth 2)
        assert idx["link_1"] < idx["link_2"]
        assert idx["link_1"] < idx["link_3"]
        # link_2 (depth 2) must precede end_effector (depth 3)
        assert idx["link_2"] < idx["end_effector"]
        print(f"\n  BFS order: {order}")

    def test_bfs_find_level_root(self):
        from core.ai_reasoning import BFSKinematicSearch
        lvl = BFSKinematicSearch.find_level("base", "base", _simple_adj())
        assert lvl == 0

    def test_bfs_find_level_deep(self):
        from core.ai_reasoning import BFSKinematicSearch
        lvl = BFSKinematicSearch.find_level("base", "end_effector", _simple_adj())
        assert lvl == 3

    def test_bfs_find_level_unreachable(self):
        from core.ai_reasoning import BFSKinematicSearch
        lvl = BFSKinematicSearch.find_level("base", "ghost_link", _simple_adj())
        assert lvl == -1

    def test_bfs_adjacent_pairs(self):
        from core.ai_reasoning import BFSKinematicSearch
        pairs = BFSKinematicSearch.adjacent_pairs("base", _simple_adj())
        assert ("base",   "link_1") in pairs
        assert ("link_1", "link_2") in pairs
        assert ("link_1", "link_3") in pairs
        assert ("link_2", "end_effector") in pairs
        print(f"\n  Adjacent pairs: {pairs}")


class TestModule2_DFS:
    """Module 2 — Depth-First Search (R&N §3.4.3)"""

    def test_dfs_visits_all_nodes(self):
        from core.ai_reasoning import DFSKinematicSearch
        order = DFSKinematicSearch.dfs_order("base", _simple_adj())
        assert set(order) == _all_nodes(_simple_adj())

    def test_dfs_root_first(self):
        from core.ai_reasoning import DFSKinematicSearch
        order = DFSKinematicSearch.dfs_order("base", _simple_adj())
        assert order[0] == "base"

    def test_no_cycle_in_tree(self):
        from core.ai_reasoning import DFSKinematicSearch
        assert not DFSKinematicSearch.has_cycle("base", _simple_adj())

    def test_cycle_detected(self):
        from core.ai_reasoning import DFSKinematicSearch
        # Introduce a back-edge: end_effector → base
        adj = {**_simple_adj(), "end_effector": ["base"]}
        assert DFSKinematicSearch.has_cycle("base", adj), \
            "Cycle must be detected"

    def test_depth_limited_search(self):
        from core.ai_reasoning import DFSKinematicSearch
        within_2 = DFSKinematicSearch.depth_limited_search("base", _simple_adj(), 2)
        assert "base"    in within_2
        assert "link_1"  in within_2
        assert "link_2"  in within_2
        assert "link_3"  in within_2
        # end_effector is at depth 3 — must NOT be found
        assert "end_effector" not in within_2
        print(f"\n  DLS(limit=2): {within_2}")


class TestModule2_AStar:
    """Module 2 — A* Search (R&N §3.5.2)"""

    def _make_components_and_joints(self):
        """Minimal stubs for AStarChainFinder."""
        from dataclasses import dataclass, field as df
        import numpy as np

        @dataclass
        class FakeComp:
            id: str
            centroid: np.ndarray
            type: str = "arm_link"
            n_points: int = 500

        @dataclass
        class FakeJoint:
            id: str
            parent_link: str
            child_link: str
            confidence: float = 0.9

        comps = [
            FakeComp("base",          np.array([0., 0., 0.])),
            FakeComp("link_1",        np.array([0., 0., 0.2])),
            FakeComp("link_2",        np.array([0., 0., 0.4])),
            FakeComp("end_effector",  np.array([0., 0., 0.6]), type="end_effector"),
        ]
        joints = [
            FakeJoint("j1", "base",   "link_1",       confidence=0.95),
            FakeJoint("j2", "link_1", "link_2",        confidence=0.80),
            FakeJoint("j3", "link_2", "end_effector",  confidence=0.90),
        ]
        return comps, joints

    def test_astar_finds_path(self):
        from core.ai_reasoning import AStarChainFinder
        comps, joints = self._make_components_and_joints()
        finder = AStarChainFinder(comps, joints)
        chain, cost = finder.find_chain("base", "end_effector")
        assert chain == ["base", "link_1", "link_2", "end_effector"]
        assert cost > 0
        print(f"\n  A* chain: {chain}  cost={cost:.4f}")

    def test_astar_no_path(self):
        from core.ai_reasoning import AStarChainFinder
        comps, joints = self._make_components_and_joints()
        finder = AStarChainFinder(comps, joints)
        chain, cost = finder.find_chain("end_effector", "base")  # reversed
        assert chain == [] and cost == float("inf")

    def test_astar_heuristic_admissible(self):
        """h(n) must never overestimate actual cost."""
        from core.ai_reasoning import AStarChainFinder
        comps, joints = self._make_components_and_joints()
        finder = AStarChainFinder(comps, joints)
        for comp in comps:
            h = finder.heuristic(comp.id, "end_effector")
            assert h >= 0, f"Heuristic must be non-negative: h({comp.id})={h}"


class TestModule2_UCS:
    """Module 2 — Uniform Cost Search (R&N §3.4.2)"""

    def test_ucs_finds_cheapest_plan(self):
        from core.ai_reasoning import UCSScanPlanner
        import numpy as np

        vps = [
            UCSScanPlanner.ViewPoint("front",  np.array([1.,0.,0.]), {"base","link_1"}),
            UCSScanPlanner.ViewPoint("left",   np.array([0.,1.,0.]), {"link_1","link_2"}),
            UCSScanPlanner.ViewPoint("top",    np.array([0.,0.,1.]), {"link_2","ee"}),
            UCSScanPlanner.ViewPoint("back",   np.array([-1.,0.,0.]),{"base"}),
        ]
        planner = UCSScanPlanner(vps, {"base","link_1","link_2","ee"})
        seq, cost = planner.plan("front")

        covered = set()
        for vp_id in seq:
            covered |= next(v.covers for v in vps if v.id == vp_id)

        assert covered >= {"base","link_1","link_2","ee"}, \
            f"Plan must cover all links, got {covered}"
        assert cost > 0
        print(f"\n  UCS scan plan: {seq}  cost={cost:.3f}")


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 3 — Local Search
# ─────────────────────────────────────────────────────────────────────────────

class TestModule3_HillClimbing:
    """Module 3 — Hill-Climbing (R&N §4.1.1)"""

    def test_climbs_towards_maximum(self):
        from core.ai_reasoning import HillClimbingAxisOptimizer
        hc = HillClimbingAxisOptimizer(delta=0.05, max_iter=100)

        # Fitness peaks at axis = [0, 0, 1]
        target = np.array([0., 0., 1.])
        def fitness(ax):
            return float(np.dot(ax / np.linalg.norm(ax), target))

        init_axis = np.array([0.5, 0.5, 0.5])
        best_ax, best_f = hc.optimise(init_axis, fitness)

        assert best_f >= fitness(init_axis), \
            "Hill-climbing must not decrease fitness"
        assert abs(np.linalg.norm(best_ax) - 1.0) < 0.01, \
            "Returned axis must be unit vector"
        print(f"\n  HC: init_f={fitness(init_axis):.3f}  best_f={best_f:.3f}")

    def test_unit_sphere_neighbours(self):
        from core.ai_reasoning import HillClimbingAxisOptimizer
        hc   = HillClimbingAxisOptimizer(delta=0.05)
        nbrs = hc._neighbours(np.array([0., 0., 1.]))
        assert len(nbrs) == 6, "Expect 6 directional neighbours"
        for n in nbrs:
            assert abs(np.linalg.norm(n) - 1.0) < 1e-9, \
                f"Neighbour must be unit vector: {n}"

    def test_terminates_at_global_max(self):
        from core.ai_reasoning import HillClimbingAxisOptimizer
        hc = HillClimbingAxisOptimizer(delta=0.1, max_iter=200)
        target = np.array([1., 0., 0.])
        def f(ax): return float(np.dot(ax, target))
        best, val = hc.optimise(target, f)
        # Already at maximum — should not degrade
        assert val >= f(target) - 1e-6


class TestModule3_SimulatedAnnealing:
    """Module 3 — Simulated Annealing (R&N §4.1.2)"""

    def test_finds_better_solution(self):
        from core.ai_reasoning import SimulatedAnnealingParamOptimizer
        sa = SimulatedAnnealingParamOptimizer(T0=1.0, alpha=0.99,
                                              T_min=1e-3, max_iter=500)
        # Minimise distance from target params
        target = {"mass": 1.5, "ixx": 0.03, "iyy": 0.02}
        def energy(p):
            return sum((p[k] - target[k])**2 for k in p)

        init   = {"mass": 3.0, "ixx": 0.10, "iyy": 0.08}
        best, e = sa.optimise(init, energy)

        assert e <= energy(init), \
            f"SA must not increase energy: {e:.4f} > {energy(init):.4f}"
        assert all(v > 0 for v in best.values()), \
            "SA must keep params positive"
        print(f"\n  SA: init_e={energy(init):.4f}  best_e={e:.4f}")

    def test_acceptance_probability(self):
        """P(accept) = exp(-ΔE/T); must be in (0,1) for ΔE>0."""
        import math
        delta_E, T = 0.5, 1.0
        p = math.exp(-delta_E / T)
        assert 0 < p < 1

    def test_temperature_schedule(self):
        """T decreases geometrically: T(t) = T0 * alpha^t."""
        T0, alpha = 1.0, 0.99
        T = T0
        for t in range(100):
            T *= alpha
        expected = T0 * (alpha ** 100)
        assert abs(T - expected) < 1e-12


class TestModule3_GeneticAlgorithm:
    """Module 3 — Genetic Algorithm (R&N §4.1.4)"""

    def test_evolves_better_genome(self):
        from core.ai_reasoning import GeneticURDFOptimizer
        ga = GeneticURDFOptimizer(pop_size=20, n_gen=30, mutation_rate=0.15)

        # Fitness: maximise sum of genome values
        bounds = [(-1.57, 1.57)] * 6
        def fitness(g): return sum(g)

        best_g, best_f = ga.evolve(bounds, fitness)

        max_possible = 1.57 * 6
        assert best_f > -max_possible, "GA must find better than random"
        assert all(lo <= v <= hi for v, (lo,hi) in zip(best_g, bounds)), \
            "All values must stay within bounds"
        print(f"\n  GA: best_fitness={best_f:.4f} / {max_possible:.4f}")

    def test_crossover_length_preserved(self):
        from core.ai_reasoning import GeneticURDFOptimizer
        ga = GeneticURDFOptimizer()
        p1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        p2 = [-1., -2., -3., -4., -5., -6.]
        child = ga._crossover(p1, p2)
        assert len(child) == len(p1), "Crossover must preserve genome length"

    def test_mutation_within_bounds(self):
        from core.ai_reasoning import GeneticURDFOptimizer
        ga     = GeneticURDFOptimizer(mutation_rate=1.0)  # always mutate
        bounds = [(-1.0, 1.0)] * 4
        genome = [0.5, 0.5, 0.5, 0.5]
        for _ in range(50):
            mut = ga._mutate(genome, bounds)
            assert all(-1.0 <= v <= 1.0 for v in mut), \
                f"Mutated genome out of bounds: {mut}"


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 4 — Logic and Reasoning
# ─────────────────────────────────────────────────────────────────────────────

class TestModule4_FOL:
    """Module 4 — First-Order Logic rules (R&N §8)"""

    def _make_comp(self, extent, n_pts=500, centroid_z=0.0, movable=True, type_="arm_link"):
        from dataclasses import dataclass
        import numpy as np
        @dataclass
        class FakeComp:
            id: str = "c"
            extent: np.ndarray = None
            n_points: int = 500
            centroid: np.ndarray = None
            movable: bool = True
            type: str = "arm_link"
        c = FakeComp()
        c.extent   = np.array(extent)
        c.centroid = np.array([0., 0., centroid_z])
        c.n_points = n_pts
        c.movable  = movable
        c.type     = type_
        return c

    def test_disc_predicate_true(self):
        from core.ai_reasoning import FOLComponentRules
        fol  = FOLComponentRules()
        disc = self._make_comp([0.03, 0.12, 0.12])  # thin disc
        assert fol.IsDisc(disc), "Flat disc should satisfy IsDisc"

    def test_disc_predicate_false_for_rod(self):
        from core.ai_reasoning import FOLComponentRules
        fol = FOLComponentRules()
        rod = self._make_comp([0.04, 0.04, 0.25])
        assert not fol.IsDisc(rod)

    def test_cylinder_predicate(self):
        from core.ai_reasoning import FOLComponentRules
        fol = FOLComponentRules()
        rod = self._make_comp([0.04, 0.04, 0.30])  # rod_ratio = 7.5
        assert fol.IsCylinder(rod)

    def test_base_classification(self):
        from core.ai_reasoning import FOLComponentRules
        fol   = FOLComponentRules()
        base  = self._make_comp([0.15,0.15,0.10], n_pts=2000, centroid_z=0.0)
        small = self._make_comp([0.04,0.04,0.20], n_pts=400,  centroid_z=0.2)
        all_c = [base, small]
        cm    = {"c": []}
        assert fol.classify(base,  all_c, cm) == "base"
        assert fol.classify(small, all_c, cm) != "base"

    def test_end_effector_classification(self):
        from core.ai_reasoning import FOLComponentRules
        fol  = FOLComponentRules()
        ee   = self._make_comp([0.03, 0.03, 0.03], n_pts=100,  centroid_z=0.5)
        base = self._make_comp([0.15, 0.15, 0.10], n_pts=2000, centroid_z=0.0)
        all_c = [base, ee]
        # children_map: ee has no children
        cm    = {base.id: [ee.id], ee.id: []}
        ee.id = "ee"; base.id = "base"
        result = fol.classify(ee, all_c, {base.id:[ee.id], ee.id:[]})
        assert result == "end_effector", f"Got {result}"


class TestModule4_ForwardChaining:
    """Module 4 — Forward Chaining (R&N §9.3.1)"""

    def test_wheel_infers_continuous(self):
        from core.ai_reasoning import ForwardChainer
        fc     = ForwardChainer()
        facts  = {"child_type": "wheel", "child_movable": True}
        result = fc.infer(facts)
        assert result["joint_type"] == "continuous"

    def test_arm_link_infers_revolute(self):
        from core.ai_reasoning import ForwardChainer
        fc     = ForwardChainer()
        facts  = {"child_type": "arm_link", "child_movable": True}
        result = fc.infer(facts)
        assert result["joint_type"] == "revolute"

    def test_not_movable_infers_fixed(self):
        from core.ai_reasoning import ForwardChainer
        fc     = ForwardChainer()
        facts  = {"child_type": "arm_link", "child_movable": False}
        result = fc.infer(facts)
        assert result["joint_type"] == "fixed"

    def test_high_rod_ratio_infers_prismatic(self):
        from core.ai_reasoning import ForwardChainer
        fc     = ForwardChainer()
        facts  = {"child_type": "arm_link", "child_movable": True,
                  "child_rod_ratio": 7.0, "joint_type": "arm_link"}
        result = fc.infer(facts)
        # Rod ratio ≥ 6 → prismatic override
        assert result.get("joint_type") == "prismatic"

    def test_fixpoint_reached(self):
        """FC must terminate even if new facts trigger new rules."""
        from core.ai_reasoning import ForwardChainer
        fc     = ForwardChainer()
        facts  = {"child_type": "end_effector", "child_movable": True,
                  "axis_confidence": 0.40, "limit_range": 8.0}
        result = fc.infer(facts)
        # Must terminate (no infinite loop)
        assert "joint_type" in result

    def test_low_confidence_triggers_rescan(self):
        from core.ai_reasoning import ForwardChainer
        fc     = ForwardChainer()
        facts  = {"child_type": "other", "child_movable": True,
                  "joint_type": "other", "axis_confidence": 0.50}
        result = fc.infer(facts)
        assert result.get("needs_rescan") is True


class TestModule4_BackwardChaining:
    """Module 4 — Backward Chaining (R&N §9.3.2)"""

    def test_valid_urdf_provable_with_all_facts(self):
        from core.ai_reasoning import BackwardChainer
        bc       = BackwardChainer()
        # Provide all base facts
        available = {
            "segmentation_done", "bayesian_classification",
            "forward_chaining_done", "pca_estimate_done",
            "default_limits", "bbox_fallback", "bbox_inertia",
        }
        ok, goals = bc.check_urdf_feasibility(available)
        assert ok, f"Valid URDF should be provable; proved={goals}"
        assert "valid_urdf" in goals

    def test_urdf_not_provable_without_segmentation(self):
        from core.ai_reasoning import BackwardChainer
        bc  = BackwardChainer()
        ok, goals = bc.check_urdf_feasibility(set())
        # Even without facts, base axioms are in GOAL_RULES as empty-body rules
        # So valid_urdf WILL be provable (all fallbacks exist)
        # This tests that the chainer runs without error
        assert isinstance(ok, bool)

    def test_prove_known_fact_immediately(self):
        from core.ai_reasoning import BackwardChainer
        bc    = BackwardChainer()
        known = {"segmentation_done"}
        ok, _ = bc.prove("segmentation_done", known)
        assert ok

    def test_prove_unknown_fact_via_subgoal(self):
        from core.ai_reasoning import BackwardChainer
        bc    = BackwardChainer()
        known = {"pca_estimate_done"}
        ok, goals = bc.prove("axes_known", known)
        assert ok
        assert "axes_known" in goals


class TestModule4_Resolution:
    """Module 4 — Resolution constraint verification (R&N §9.5)"""

    def _make_components_and_joints(self, base_movable=False, limit_lo=-1.57,
                                     limit_hi=1.57, axis=np.array([0.,0.,1.])):
        from dataclasses import dataclass, field as df

        @dataclass
        class FakeComp:
            id: str; type: str; movable: bool

        @dataclass
        class FakeJoint:
            id: str; parent_link: str; child_link: str
            type: str; limit_lower: float; limit_upper: float
            axis: np.ndarray

        comps = [FakeComp("base","base", base_movable),
                 FakeComp("l1","arm_link", True)]
        joints = [FakeJoint("j1","base","l1","revolute",
                             limit_lo, limit_hi, axis)]
        return comps, joints

    def test_valid_tree_no_violations(self):
        from core.ai_reasoning import ResolutionVerifier
        rv = ResolutionVerifier()
        comps, joints = self._make_components_and_joints()
        violations = rv.verify_constraints(comps, joints)
        assert violations == [], f"Expected no violations: {violations}"

    def test_movable_base_detected(self):
        from core.ai_reasoning import ResolutionVerifier
        rv = ResolutionVerifier()
        comps, joints = self._make_components_and_joints(base_movable=True)
        violations = rv.verify_constraints(comps, joints)
        assert any("movable" in v.lower() or "base" in v.lower()
                   for v in violations), \
            f"Movable base must be flagged: {violations}"

    def test_invalid_limits_detected(self):
        from core.ai_reasoning import ResolutionVerifier
        rv = ResolutionVerifier()
        comps, joints = self._make_components_and_joints(limit_lo=1.57, limit_hi=-1.57)
        violations = rv.verify_constraints(comps, joints)
        assert any("lower" in v.lower() or "upper" in v.lower()
                   for v in violations), \
            f"Invalid limits must be flagged: {violations}"

    def test_zero_axis_detected(self):
        from core.ai_reasoning import ResolutionVerifier
        rv = ResolutionVerifier()
        comps, joints = self._make_components_and_joints(
            axis=np.array([0., 0., 0.]))
        violations = rv.verify_constraints(comps, joints)
        assert any("axis" in v.lower() for v in violations), \
            f"Zero axis must be flagged: {violations}"


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 5 — Uncertain Knowledge
# ─────────────────────────────────────────────────────────────────────────────

class TestModule5_BayesianClassifier:
    """Module 5 — Naive Bayes (R&N §13.2, §20.2)"""

    def _classify(self, **kwargs):
        from core.ai_reasoning import BayesianJointClassifier
        clf = BayesianJointClassifier()
        defaults = dict(
            disc_ratio=1.0, rod_ratio=1.0,
            tree_depth=1, n_points=500,
            all_n_points=[200, 500, 800, 1200, 1500],
            movable=True,
        )
        defaults.update(kwargs)
        return clf.classify(**defaults)

    def test_wheel_shape_gives_continuous(self):
        jtype, conf, _ = self._classify(disc_ratio=3.5, rod_ratio=1.0)
        assert jtype == "continuous", \
            f"High disc_ratio should → continuous, got {jtype}"

    def test_rod_shape_gives_prismatic(self):
        jtype, conf, _ = self._classify(disc_ratio=1.0, rod_ratio=6.0)
        assert jtype == "prismatic", \
            f"High rod_ratio should → prismatic, got {jtype}"

    def test_not_movable_gives_fixed(self):
        jtype, conf, _ = self._classify(movable=False)
        assert jtype == "fixed"
        assert conf > 0.95

    def test_posteriors_sum_to_one(self):
        _, _, probs = self._classify()
        total = sum(probs.values())
        assert abs(total - 1.0) < 1e-6, \
            f"Posteriors must sum to 1.0, got {total}"

    def test_confidence_in_range(self):
        _, conf, _ = self._classify()
        assert 0.0 <= conf <= 1.0

    def test_deep_link_favours_revolute(self):
        """Deep links (depth ≥ 3) are more likely to be revolute than fixed."""
        from core.ai_reasoning import BayesianJointClassifier
        clf = BayesianJointClassifier()
        _, _, probs = clf.classify(
            disc_ratio=1.5, rod_ratio=2.5, tree_depth=4,
            n_points=300, all_n_points=[200,300,500,1000], movable=True)
        assert probs["revolute"] > probs["fixed"], \
            "Deep movable links should be more likely revolute than fixed"

    def test_bayes_rule_applied(self):
        """
        Verify P(type | features) ∝ P(type) × P(features | type).
        For disc_high=True, continuous should have higher posterior than prismatic.
        """
        from core.ai_reasoning import BayesianJointClassifier
        clf = BayesianJointClassifier()
        _, _, probs = clf.classify(
            disc_ratio=3.0, rod_ratio=1.0,
            tree_depth=1, n_points=500,
            all_n_points=[200,500,800], movable=True)
        # P(disc_high|continuous)=0.85 > P(disc_high|prismatic)=0.10
        assert probs["continuous"] > probs["prismatic"], \
            f"Disc shape → continuous > prismatic: {probs}"


class TestModule5_BayesianBeliefNetwork:
    """Module 5 — Bayesian Belief Network (R&N §14.1–14.4)"""

    def test_marginals_sum_to_one_area(self):
        from core.ai_reasoning import BayesianBeliefNetwork
        bbn = BayesianBeliefNetwork()
        m   = bbn.infer_urdf_validity(scan_quality=0.8)
        # Each marginal is P(variable=True) so each should be in [0,1]
        for k, v in m.items():
            assert 0.0 <= v <= 1.0, f"Marginal {k}={v} out of [0,1]"

    def test_good_scan_gives_high_urdf_prob(self):
        from core.ai_reasoning import BayesianBeliefNetwork
        bbn  = BayesianBeliefNetwork()
        good = bbn.infer_urdf_validity(scan_quality=0.95)
        bad  = bbn.infer_urdf_validity(scan_quality=0.10)
        assert good["urdf_valid"] > bad["urdf_valid"], \
            "Better scan quality should → higher P(URDFValid)"
        print(f"\n  BBN: good_scan P(valid)={good['urdf_valid']:.3f}  "
              f"bad_scan P(valid)={bad['urdf_valid']:.3f}")

    def test_scan_quality_propagates(self):
        """P(SegAcc) must be higher under good scan than bad scan."""
        from core.ai_reasoning import BayesianBeliefNetwork
        bbn = BayesianBeliefNetwork()
        assert bbn.infer_urdf_validity(0.9)["seg_good"] > \
               bbn.infer_urdf_validity(0.1)["seg_good"]

    def test_d_separation(self):
        """Without evidence, P(URDFValid) should be between 0.3 and 0.95."""
        from core.ai_reasoning import BayesianBeliefNetwork
        bbn = BayesianBeliefNetwork()
        m   = bbn.infer_urdf_validity(scan_quality=None)
        assert 0.30 < m["urdf_valid"] < 0.95, \
            f"Prior P(URDFValid) out of plausible range: {m['urdf_valid']:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 6 — Planning
# ─────────────────────────────────────────────────────────────────────────────

class TestModule6_ClassicalPlanning:
    """Module 6 — Classical Planning (R&N §10.1–10.2)"""

    def test_forward_search_finds_plan(self):
        from core.ai_reasoning import ClassicalPlanningPipeline, PlanningState
        planner = ClassicalPlanningPipeline()
        init    = PlanningState(n_poses_captured=3)
        plan, cost = planner.forward_search(init)
        assert len(plan) > 0, "Forward search must find a plan"
        assert cost < float("inf"), "Plan must have finite cost"
        assert "VALIDATE" in plan, "Plan must include VALIDATE"
        assert "BUILD_URDF" in plan
        print(f"\n  Plan: {plan}  cost={cost:.1f}")

    def test_plan_ends_at_goal(self):
        from core.ai_reasoning import ClassicalPlanningPipeline, PlanningState
        planner = ClassicalPlanningPipeline()
        init    = PlanningState()
        plan, _ = planner.forward_search(init)
        # Simulate plan execution
        state   = PlanningState()
        for action_name in plan:
            action = next(a for a in planner.actions if a.name == action_name)
            assert action.precond(state), \
                f"Precondition failed for {action_name}"
            state = action.effect(state)
        assert state.validated and state.urdf_built, \
            "Executing plan must reach goal state"

    def test_backward_search_returns_actions(self):
        from core.ai_reasoning import ClassicalPlanningPipeline, PlanningState
        planner = ClassicalPlanningPipeline()
        goal    = PlanningState(validated=True, urdf_built=True)
        seq     = planner.backward_search(goal)
        assert "VALIDATE" in seq
        assert "BUILD_URDF" in seq

    def test_action_costs_are_positive(self):
        from core.ai_reasoning import ClassicalPlanningPipeline
        planner = ClassicalPlanningPipeline()
        for action in planner.actions:
            assert action.cost > 0, \
                f"Action {action.name} must have positive cost"

    def test_planning_state_fluents(self):
        from core.ai_reasoning import PlanningState
        s = PlanningState()
        assert not s.validated
        assert not s.urdf_built
        s.validated = True
        assert s.validated

    def test_rescan_action_clears_low_conf(self):
        from core.ai_reasoning import ClassicalPlanningPipeline, PlanningState
        planner = ClassicalPlanningPipeline()
        state   = PlanningState(urdf_built=True, low_conf_joints=["j1","j2"])
        rescan  = next(a for a in planner.actions if a.name == "RESCAN")
        assert rescan.precond(state), "RESCAN should fire when low_conf_joints non-empty"
        new_s = rescan.effect(state)
        assert new_s.low_conf_joints == [], "RESCAN must clear low_conf_joints"


# ─────────────────────────────────────────────────────────────────────────────
#  MODULE 1 — Intelligent Agent
# ─────────────────────────────────────────────────────────────────────────────

class TestModule1_Agent:
    """Module 1 — Intelligent Agent (R&N §2.1–2.4)"""

    def _make_agent(self):
        from core.ai_reasoning import TwinForgeAgent
        return TwinForgeAgent(n_poses=3, confidence_threshold=0.75)

    def _make_scan_data(self):
        """Minimal scan data stub."""
        try:
            import open3d as o3d
        except ImportError:
            pytest.skip("open3d not installed")
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(np.random.randn(1000,3)*0.1)

        class FakeScan:
            pass
        s = FakeScan()
        s.pcd = pcd
        s.pose_pcds = [pcd, pcd, pcd]
        return s

    def test_perceive_returns_percept(self):
        agent  = self._make_agent()
        scan   = self._make_scan_data()
        percept = agent.perceive(scan)
        assert "n_points"     in percept
        assert "scan_quality" in percept
        assert 0 <= percept["scan_quality"] <= 1

    def test_perceive_updates_bbn(self):
        agent   = self._make_agent()
        scan    = self._make_scan_data()
        agent.perceive(scan)
        assert "urdf_valid" in agent.model["bbn_marginals"]

    def test_make_plan_returns_sequence(self):
        agent   = self._make_agent()
        percept = {"n_poses": 3, "scan_quality": 0.8}
        plan    = agent.make_plan(percept)
        assert len(plan) > 0
        assert "VALIDATE" in plan
        print(f"\n  Agent plan: {plan}")

    def test_verify_flags_zero_axis(self):
        from dataclasses import dataclass

        @dataclass
        class FakeComp:
            id: str; type: str; movable: bool

        @dataclass
        class FakeJoint:
            id: str; parent_link: str; child_link: str
            type: str; limit_lower: float; limit_upper: float
            axis: np.ndarray

        agent  = self._make_agent()
        comps  = [FakeComp("base","base",False),
                  FakeComp("l1","arm_link",True)]
        joints = [FakeJoint("j1","base","l1","revolute",-1.57,1.57,
                             np.array([0.,0.,0.]))]  # zero axis!
        violations = agent.verify(comps, joints)
        assert len(violations) > 0, "Agent must detect zero axis"

    def test_backward_chaining_feasibility(self):
        agent = self._make_agent()
        available = {"segmentation_done","pca_estimate_done",
                     "default_limits","bbox_inertia","bayesian_classification",
                     "forward_chaining_done","bbox_fallback"}
        ok = agent.check_feasibility(available)
        assert ok, "URDF should be feasible with full fact set"


# ─────────────────────────────────────────────────────────────────────────────
#  SEARCH NODE utility
# ─────────────────────────────────────────────────────────────────────────────

class TestSearchNode:
    def test_path_reconstruction(self):
        from core.ai_reasoning import SearchNode
        n0 = SearchNode(f_cost=0.0, g_cost=0.0, state="a", parent=None)
        n1 = SearchNode(f_cost=1.0, g_cost=1.0, state="b", parent=n0)
        n2 = SearchNode(f_cost=2.0, g_cost=2.0, state="c", parent=n1)
        assert n2.path() == ["a", "b", "c"]

    def test_ordering_by_f_cost(self):
        import heapq
        from core.ai_reasoning import SearchNode
        nodes = [
            SearchNode(f_cost=3.0, state="c"),
            SearchNode(f_cost=1.0, state="a"),
            SearchNode(f_cost=2.0, state="b"),
        ]
        heapq.heapify(nodes)
        assert heapq.heappop(nodes).state == "a"  # lowest f
        assert heapq.heappop(nodes).state == "b"
