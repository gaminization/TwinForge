# TwinForge 3D
### AI-Powered Generation of Digital Twins from Physical Robots
**Course: BCSE306L Artificial Intelligence — VIT University**
*Team: Garv Arora, Arnav Sinha | Faculty: Prof. Akila Victor*

---

## Quick Start

```bash
# Install
pip install -r requirements.txt

# Run demo (no hardware)
python twinforge.py --demo

# From a real scan
python twinforge.py --input scan.ply --name my_robot

# Mobile app (phone + PC over WiFi)
python start_server.py
# Scan the QR code on your phone, walk around the robot, get URDF
```

---

## BCSE306L Algorithm Map

Every syllabus module is implemented in `core/ai_reasoning.py`.

| Module | Algorithm | Where Used |
|--------|-----------|------------|
| **M1** Intelligent Agents | PEAS model-based agent | `TwinForgeAgent` |
| **M2** BFS | Level-order kinematic traversal | `BFSKinematicSearch` |
| **M2** DFS + Cycle Detection | FK computation, URDF validation | `DFSKinematicSearch` |
| **M2** Depth-Limited Search | Collision-relevant link search | `DFSKinematicSearch.depth_limited_search` |
| **M2** UCS | Optimal scan sequence planning | `UCSScanPlanner` |
| **M2** A\* | Best end-effector chain discovery | `AStarChainFinder` |
| **M3** Hill-Climbing | Joint axis refinement on S² | `HillClimbingAxisOptimizer` |
| **M3** Simulated Annealing | Physical parameter optimisation | `SimulatedAnnealingParamOptimizer` |
| **M3** Genetic Algorithm | Joint limit evolution | `GeneticURDFOptimizer` |
| **M4** FOL | Component classification axioms | `FOLComponentRules` |
| **M4** Forward Chaining | Joint type inference (8 rules) | `ForwardChainer` |
| **M4** Backward Chaining | URDF feasibility goal tree | `BackwardChainer` |
| **M4** Resolution | Structural constraint verification | `ResolutionVerifier` |
| **M5** Naïve Bayes | Joint type classification | `BayesianJointClassifier` |
| **M5** Bayesian Belief Network | Pipeline success probability | `BayesianBeliefNetwork` |
| **M6** Classical Planning | PDDL scan-to-URDF pipeline | `ClassicalPlanningPipeline` |
| **M6** Forward/Backward Search | Action sequence optimisation | `forward_search / backward_search` |
| **M7** Perception | RGB-D → point cloud → components | `acquisition.py + segmenter.py` |
| **M7** Object Recognition | DBSCAN + FOL component labelling | `segmenter.py` |

Full documentation with R&N citations → **`DOCUMENTATION.md`**

---

## Architecture

```
Physical Robot
    │
    ▼ Module 7: Perception
Camera / Phone Scan ──────────► Point Cloud
    │
    ▼ Module 2: BFS/A* Search
DBSCAN Segmentation ──────────► Components + depths
    │
    ▼ Module 4: FOL + Forward Chaining
Component Classification ─────► Types (base/arm/wheel/ee)
    │
    ▼ Module 5: Naïve Bayes + BBN
Joint Type Inference ─────────► revolute/prismatic/continuous/fixed
    │
    ▼ Module 3: Hill-Climbing + ICP
Axis Optimisation ────────────► Refined rotation axes
    │
    ▼ Module 4: Resolution
Constraint Verification ──────► Structural validity
    │
    ▼ Module 6: Classical Planning
Pipeline Execution ───────────► Ordered action sequence
    │
    ▼
URDF + STL Meshes + 3D Viewer
```

---

## Project Structure

```
twinforge3d/
├── twinforge.py            CLI entry point
├── start_server.py         Mobile app server launcher (prints QR code)
│
├── core/
│   ├── ai_reasoning.py     ALL BCSE306L algorithms (M2–M7)
│   ├── acquisition.py      Scan input (PLY/PCD/video/demo)
│   ├── segmenter.py        DBSCAN + FOL component detection
│   ├── joint_detector.py   Bayes + HC + FC joint detection
│   ├── kinematic_tree.py   BFS/DFS tree with cycle detection
│   ├── urdf_builder.py     URDF + STL mesh generation
│   └── visualizer.py       FK animation (Open3D + matplotlib)
│
├── server/
│   ├── app.py              FastAPI server (WebSocket progress)
│   └── static/
│       ├── index.html      Mobile PWA (camera scan UI)
│       └── viewer.html     3D URDF viewer (Three.js FK)
│
├── tests/
│   ├── test_e2e.py         16 pipeline integration tests
│   └── test_ai_reasoning.py 63 algorithm unit tests (one per M2–M7 algo)
│
└── DOCUMENTATION.md        Complete syllabus → algorithm mapping
```

---

## Datasets & Models

| Resource | Purpose | Reference |
|----------|---------|-----------|
| PartNet-Mobility | Bayesian prior P(joint_type) | Chang et al., ICCV 2019 |
| URDF-Anything benchmark | Evaluation metric | Liu et al., 2025 |
| SAM 2 (optional) | 2D segmentation masks | Meta AI, 2024 |
| GPT-4o / LLaVA (optional) | VLM semantic labelling | OpenAI / Haotian Liu |

The core pipeline runs with **zero pre-trained models** — all algorithms are classical AI.

---

## Test Results

```
tests/test_e2e.py          16/16 passed   (full pipeline: scan → URDF → FK animation)
tests/test_ai_reasoning.py 68/68 passed   (every BCSE306L algorithm individually tested)
                           ─────────────
Total                      84/84 passed
```

---

## References

1. Russell, S. & Norvig, P. (2015). *AI: A Modern Approach*, 3rd Ed. Prentice Hall.
2. Ester et al. (1996). DBSCAN. *KDD-96*.
3. Chang et al. (2019). PartNet-Mobility. *ICCV*.
4. Liu et al. (2025). URDF-Anything. *arXiv:2511.00940*.
5. Kavraki et al. (1996). PRM. *IEEE TRA*.
