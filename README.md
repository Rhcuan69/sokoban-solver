# Sokoban Solver

[![make.fun · sokoban-solver-01-run-20261121](https://make.fun/badge/sokoban-solver-01-run-20261121.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![rank](https://make.fun/badge/sokoban-solver-01-run-20261121/Rhcuan69/sokoban-solver/rank.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)
[![reward](https://make.fun/badge/sokoban-solver-01-run-20261121/Rhcuan69/sokoban-solver/reward.svg)](https://make.fun/bounty/sokoban-solver-01-run-20261121?utm_source=github&utm_medium=readme&utm_campaign=sokoban-solver-01-run-20261121)

A push-optimal Sokoban solver for the make.fun bounty
[sokoban-solver-01-run-20261121](https://make.fun/bounty/sokoban-solver-01-run-20261121)
("Build the best Sokoban solver, 21 November 2026"). One program answers all
5 instances of a set in a single run: it writes the player's moves that push
every box onto a goal, in as few pushes as possible.

## How it works

- **Decomposition:** each level is a one-wide hallway with 1–4 rooms off it.
  A box pushed out of a room can never return, so every room is an
  independent problem: rooms are solved separately and stitched together
  with hallway walks.
- **Per-room push-optimal A\*** over box configurations with a canonical
  player region, an admissible heuristic from exact single-box push-distance
  tables (Dijkstra over box/player states, reversed from each goal) combined
  by cheapest box-to-goal assignment, and 2×2 freeze-deadlock pruning.
- **Deterministic work budget:** a run-wide expansion cap plus weighted and
  greedy fallback passes, so one hard room can never time out the whole run.
  No clock reads, no randomness — two runs write byte-identical outputs.

## Practice score

`makefun grade-local --driver process` on the kit's practice set:
**1,000,000 / 1,000,000** — every instance solved at push-optimal P\*
(i001 13, i002 16, i003 29, i004 47, i005 60 pushes), in ~0.4 s for the whole
set. The house baseline plain-astar 1.0.0 scores 200,000 on the same set.

## Run

`main.ts` is plain Node 22 TypeScript (stdlib only: `node:fs`, `node:path`),
no dependencies, no network. It reads `/input/<id>/level.txt` and writes
`/output/<id>/solution.txt` (LURD moves) for each instance.
