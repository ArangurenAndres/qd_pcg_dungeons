# QD-PCG-Dungeons  
**Quality–Diversity Procedural Dungeon Generation with MAP-Elites and Self-Adaptive Evolution Strategies**

This repository implements a **quality–diversity (QD)** approach to procedural dungeon generation using **MAP-Elites** and **self-adaptive Evolution Strategies (ES)**. Instead of producing a single optimal level, the system discovers and maintains a diverse archive of *playable, high-quality dungeon layouts* that differ along meaningful structural dimensions.

The project is designed as a **research-oriented, modular PCG framework** with explicit playability guarantees, interpretable evaluation, and strong exploratory behavior.

---

## Overview

Each dungeon is represented as a 2D grid composed of walls, floors, a start tile, and a goal tile. The generation process is evolutionary and archive-based: candidate levels are mutated, evaluated, and inserted into a MAP-Elites archive according to their behavioral characteristics and fitness.

The core objective is to maximize **behavioral coverage** while maintaining **high-quality elites** in each region of the design space.

---

## Playability Constraint: Breadth-First Search (BFS)

Playability is enforced as a **hard constraint** using **Breadth-First Search (BFS)**. Each walkable tile is treated as a node in a graph, with edges connecting adjacent tiles. Because all moves have equal cost, BFS guarantees that the first time the goal is reached, the path found is the **shortest possible path**.

If BFS cannot find a path from the start tile `S` to the goal tile `G`, the level is discarded immediately. This ensures that all archived content is playable by construction.

---

## Evaluation and Fitness

Playable levels are further evaluated using a simple **noisy navigation agent** that probabilistically follows the BFS shortest path while occasionally taking random actions. Multiple rollouts provide soft behavioral signals describing robustness and navigational structure.

The fitness function is a weighted combination of **normalized components**, including:
- Noisy agent success rate  
- Normalized shortest-path length  
- Fraction of the level explored  
- Penalty for deviation from a target wall density  

Fitness is used **locally within each behavior bin**, not as a global ranking.

---

## Quality–Diversity via MAP-Elites

The MAP-Elites archive discretizes a two-dimensional behavior space:

- **X-axis:** interior wall density  
- **Y-axis:** normalized shortest-path length  

Each bin stores the highest-fitness level discovered for that region. Coverage is defined as the fraction of occupied bins:

\[
\text{coverage} = \frac{\text{number of filled bins}}{\text{total number of bins}}
\]

This descriptor choice provides smooth, interpretable coverage of the design space and avoids sparse or unstable binning.

---

## Evolutionary Dynamics

Each individual carries a self-adaptive mutation rate \( \sigma \), updated using a log-normal rule:

\[
\sigma' = \sigma \cdot \exp(\tau \cdot \mathcal{N}(0,1))
\]

Mutation operates at two scales:
- **Fine-grained tile flips** controlled by \( \sigma \)
- **Block mutations** that overwrite square regions to induce large structural changes

Exploration is further encouraged through an **early high-rate random injection schedule** and **coverage-biased parent selection**, which favors underrepresented behavioral regions.

---

## Visualization and Interface

The repository includes:
- ASCII and PNG exports of elite levels
- Coverage and fitness heatmaps over the MAP-Elites grid
- Progress plots tracking archive growth and coverage
- An interactive **Flask web application** for live parameter tuning, progress monitoring, and sample inspection

---



