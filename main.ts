// Sokoban solver: decompose each level into its hallway and rooms, solve every
// room push-optimally on its own, and stitch the walks and pushes together.
//
// One run answers all 5 instances (i001..i005) within a fixed amount of work:
// no wall clock is ever read, so two executions write identical bytes.
//
// Method, per room:
//   * Push-based A*: a state is the sorted box cells plus the player's reachable
//     region (canonicalised to its lowest cell). A move is one push; the cost is
//     the push count, which is exactly what the score counts.
//   * Heuristic: for every goal, a 0-1 BFS over (box, player) states gives the
//     fewest pushes a lone box needs to reach that goal with a free player.
//     The heuristic is the cheapest assignment of boxes to goals under that
//     table (brute-force permutations, at most 4 boxes), which lower-bounds any
//     real solution, so A* returns a push-optimal room solution.
//   * Dead squares fall out of the same table (a cell with no finite entry for
//     any goal can never hold a box that still solves the room) and 2x2 blocks
//     of wall-or-box with a box off its goal are pruned as freeze deadlocks.
//   * A fixed expansion budget per room plus one run-wide work budget keep
//     the work bounded and deterministic; if optimal A* exhausts its budget,
//     weighted (x2) and greedy (x6) passes try to at least solve the room (a
//     solved room still scores, an unsolved one scores 0), and a level whose
//     hallway/room split is not found is tried whole as one room instead.
//
// Rooms are found from the level alone: the hallway is the row with the most
// floor, and each room is a connected component of the floor once that row is
// removed. A box pushed out of a room's door can never come back, so no
// optimal solution ever moves one there, and the rooms are independent.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Expansion budgets, counted in states, never in seconds: deterministic.
// The whole run shares one work budget, so a single pathological room can
// never eat the 60-second wall and zero the other four instances: observed
// sets need under 2,000 expansions in total, this allows a hundred times that.
const ASTAR_BUDGET = 150000;
const WEIGHTED_BUDGET = 150000;
const GREEDY_BUDGET = 150000;
let workLeft = 200000;

const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const CH = ['u', 'd', 'l', 'r'];
const INF = 1e9;

// ---------------------------------------------------------------- parsing ---

/** A parsed level: walls, goals, boxes and the player, on a w-by-h grid. */
function parseLevel(text) {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const w = Math.max(...rows.map((r) => r.length));
  const h = rows.length;
  const wall = new Uint8Array(w * h).fill(1);
  const goals = [];
  const boxes = [];
  let player = -1;
  rows.forEach((row, y) => {
    for (let x = 0; x < row.length; x++) {
      const ch = row[x];
      const i = y * w + x;
      if (ch === '#') continue;
      wall[i] = 0;
      if (ch === '.' || ch === '*' || ch === '+') goals.push(i);
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  goals.sort((a, b) => a - b);
  return { w, h, wall, goals, boxes, player };
}

// ------------------------------------------------------------- BFS helpers ---

/** Shortest walk between two cells on `floor` avoiding `blocked`, as a move string, or null. */
function walkPath(w, h, floor, blocked, from, to) {
  if (from === to) return '';
  const prev = new Int32Array(floor.length).fill(-1);
  const prevDir = new Int8Array(floor.length).fill(-1);
  const q = [from];
  prev[from] = from;
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi];
    const cx = c % w;
    const cy = (c / w) | 0;
    for (let d = 0; d < 4; d++) {
      const nx = cx + DX[d];
      const ny = cy + DY[d];
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
      const n = ny * w + nx;
      if (!floor[n] || blocked[n] || prev[n] !== -1) continue;
      prev[n] = c;
      prevDir[n] = d;
      if (n === to) {
        let s = '';
        let cur = n;
        while (cur !== from) {
          s = CH[prevDir[cur]] + s;
          cur = prev[cur];
        }
        return s;
      }
      q.push(n);
    }
  }
  return null;
}

/** All cells reachable from `from` on `floor` avoiding `blocked`. */
function flood(w, h, floor, blocked, from) {
  const seen = new Uint8Array(floor.length);
  const out = [from];
  seen[from] = 1;
  for (let i = 0; i < out.length; i++) {
    const c = out[i];
    const cx = c % w;
    const cy = (c / w) | 0;
    for (let d = 0; d < 4; d++) {
      const nx = cx + DX[d];
      const ny = cy + DY[d];
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
      const n = ny * w + nx;
      if (floor[n] && !blocked[n] && !seen[n]) {
        seen[n] = 1;
        out.push(n);
      }
    }
  }
  return out;
}

// ------------------------------------------------------------ decomposition ---

/**
 * Split a level into hallway row + rooms. A room is a connected component of
 * the floor with the hallway row removed; its door is the component cell next
 * to the hallway, and `outside` is the hallway cell by that door. The hallway
 * row is not always unique by floor count (a room row can tie it), so every
 * candidate row is tried, most floor first, and the first one whose split
 * validates is used.
 */
function decompose(level) {
  const { w, h, wall } = level;
  const counts = [];
  for (let y = 0; y < h; y++) {
    let c = 0;
    for (let x = 0; x < w; x++) if (!wall[y * w + x]) c++;
    counts.push(c);
  }
  const candidates = counts
    .map((c, y) => [c, y])
    .filter(([c]) => c > 0)
    .sort((a, b) => b[0] - a[0] || a[1] - b[1])
    .map(([, y]) => y);
  for (const hallRow of candidates) {
    const dec = tryDecompose(level, hallRow);
    if (dec) return dec;
  }
  return null;
}

function tryDecompose(level, hallRow) {
  const { w, h, wall, goals, boxes } = level;
  // Any box sitting in the hallway itself breaks the room model.
  for (const b of boxes) if (((b / w) | 0) === hallRow) return null;
  const floor = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) floor[i] = wall[i] ? 0 : 1;
  const blocked = new Uint8Array(w * h);
  for (let x = 0; x < w; x++) blocked[hallRow * w + x] = 1;
  const seen = new Uint8Array(w * h);
  const rooms = [];
  const inRoom = new Set();
  for (let s = 0; s < w * h; s++) {
    if (!floor[s] || blocked[s] || seen[s]) continue;
    const comp = flood(w, h, floor, blocked, s);
    for (const c of comp) seen[c] = 1;
    const cellSet = new Set(comp);
    let door = -1;
    for (const c of comp) {
      const y = (c / w) | 0;
      if (Math.abs(y - hallRow) === 1 && floor[hallRow * w + (c % w)]) {
        door = c;
        break;
      }
    }
    if (door === -1) return null;
    const outside = hallRow * w + (door % w);
    const rBoxes = boxes.filter((b) => cellSet.has(b));
    const rGoals = goals.filter((g) => cellSet.has(g));
    if (rBoxes.length === 0 || rBoxes.length !== rGoals.length) return null;
    for (const b of rBoxes) inRoom.add(b);
    rooms.push({ cells: comp, cellSet, door, outside, boxes: rBoxes, goals: rGoals });
  }
  if (inRoom.size !== boxes.length) return null;
  rooms.sort((a, b) => a.door - b.door);
  return { hallRow, rooms };
}

// --------------------------------------------------------------- room solver ---

/**
 * Push-optimal A* for one room. `room` holds its cells (global indices),
 * boxes, goals, door cell and the hallway cell outside the door. `weight` 1
 * is optimal A*; a larger weight trades optimality for speed. Returns
 * {moves, endPlayerGlobal} (player starts on `outside`, ends inside the
 * room), or null when the budget runs out.
 */
function solveRoom(level, room, weight) {
  const { w, h } = level;
  const floor = new Uint8Array(w * h);
  for (const c of room.cells) floor[c] = 1;
  floor[room.outside] = 1;
  const cells = [...room.cells, room.outside].sort((a, b) => a - b);
  const n = cells.length;
  const idx = new Map(cells.map((c, i) => [c, i]));
  const outsideIdx = idx.get(room.outside);
  const nb = cells.map((c) => {
    const out = [-1, -1, -1, -1];
    const cx = c % w;
    const cy = (c / w) | 0;
    for (let d = 0; d < 4; d++) {
      const nx = cx + DX[d];
      const ny = cy + DY[d];
      if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
      const m = ny * w + nx;
      if (idx.has(m)) out[d] = idx.get(m);
    }
    return out;
  });
  const startBoxes = room.boxes.map((b) => idx.get(b)).sort((a, b) => a - b);
  const goalSet = new Set(room.goals.map((g) => idx.get(g)));
  const goalList = [...goalSet];
  const k = startBoxes.length;

  // --- push-distance table: table[g][b] = fewest pushes a lone box needs to
  // go from cell b to goal g with a free player. Dijkstra over (box, player)
  // states, run backwards from the goal: player steps cost 0, pushes cost 1.
  // A cell with no finite entry for any goal is a dead square.
  const table = goalList.map((g) => {
    const dist = new Int32Array(n * n).fill(INF);
    // Binary heap of [d, state]; ties break by state index, deterministic.
    const heap = [];
    const less = (a, b) => a[0] < b[0] || (a[0] === b[0] && a[1] < b[1]);
    const push = (e) => {
      let i = heap.length;
      heap.push(e);
      while (i > 0) {
        const p = (i - 1) >> 1;
        if (!less(heap[i], heap[p])) break;
        const t = heap[i];
        heap[i] = heap[p];
        heap[p] = t;
        i = p;
      }
    };
    const pop = () => {
      const top = heap[0];
      const last = heap.pop();
      if (heap.length > 0) {
        heap[0] = last;
        let i = 0;
        for (;;) {
          const l = i * 2 + 1;
          const r = l + 1;
          let m = i;
          if (l < heap.length && less(heap[l], heap[m])) m = l;
          if (r < heap.length && less(heap[r], heap[m])) m = r;
          if (m === i) break;
          const t = heap[i];
          heap[i] = heap[m];
          heap[m] = t;
          i = m;
        }
      }
      return top;
    };
    for (let p = 0; p < n; p++) {
      if (p === g) continue;
      dist[g * n + p] = 0;
      push([0, g * n + p]);
    }
    while (heap.length > 0) {
      const [d0, s] = pop();
      if (d0 !== dist[s]) continue;
      const b = (s / n) | 0;
      const p = s % n;
      // Reverse of a player step (cost 0): predecessor (b, q), q next to p.
      for (let dd = 0; dd < 4; dd++) {
        const q = nb[p][dd];
        if (q < 0 || q === b) continue;
        const t = b * n + q;
        if (dist[t] > d0) {
          dist[t] = d0;
          push([d0, t]);
        }
      }
      // Reverse of a push (cost 1): the push that made (b, p) came from the
      // box at p pushed along the direction p -> b; the player stood at p
      // pushed one step further back.
      for (let dd = 0; dd < 4; dd++) {
        if (nb[p][dd] !== b) continue;
        const pp = nb[p][dd ^ 1];
        if (pp < 0) continue;
        const t = p * n + pp;
        if (dist[t] > d0 + 1) {
          dist[t] = d0 + 1;
          push([d0 + 1, t]);
        }
      }
    }
    const row = new Int32Array(n).fill(INF);
    for (let b = 0; b < n; b++) {
      let best = INF;
      for (let p = 0; p < n; p++) {
        if (p !== b && dist[b * n + p] < best) best = dist[b * n + p];
      }
      row[b] = best;
    }
    return row;
  });
  const dead = new Uint8Array(n);
  for (let b = 0; b < n; b++) {
    if (goalSet.has(b)) continue;
    let live = false;
    for (let g = 0; g < goalList.length; g++) if (table[g][b] < INF) live = true;
    if (!live) dead[b] = 1;
  }

  // --- heuristic: cheapest box->goal assignment under the push table.
  // Up to 5 boxes the assignment is brute-forced over all permutations; for
  // more boxes (only the whole-level fallback) the sum of each box's cheapest
  // goal is used instead, which is still a lower bound on the pushes left.
  // Either way the heuristic never overestimates, so weight 1 stays optimal.
  const perms = [];
  if (k <= 5) {
    const permRec = (used, cur) => {
      if (cur.length === k) {
        perms.push(cur.slice());
        return;
      }
      for (let g = 0; g < k; g++) {
        if (used[g]) continue;
        used[g] = true;
        cur.push(g);
        permRec(used, cur);
        cur.pop();
        used[g] = false;
      }
    };
    permRec(new Array(k).fill(false), []);
  }
  const heuristic = (bs) => {
    if (k <= 5) {
      let best = INF;
      for (const pm of perms) {
        let s = 0;
        for (let i = 0; i < k; i++) {
          const c = table[pm[i]][bs[i]];
          if (c >= INF) {
            s = INF;
            break;
          }
          s += c;
        }
        if (s < best) best = s;
      }
      return best;
    }
    let s = 0;
    for (let i = 0; i < k; i++) {
      let m = INF;
      for (let g = 0; g < k; g++) if (table[g][bs[i]] < m) m = table[g][bs[i]];
      if (m >= INF) return INF;
      s += m;
    }
    return s;
  };

  // --- A* over pushes. State: sorted box indices + canonical player cell
  // (lowest cell of the player's reachable region). The key is a string, so
  // it stays exact for any number of boxes.
  const isGoal = (bs) => {
    for (let i = 0; i < k; i++) if (!goalSet.has(bs[i])) return false;
    return true;
  };
  const keyOf = (bs, pc) => bs.join(',') + '|' + pc;
  const regionOf = (bs, from) => {
    const blocked = new Uint8Array(n);
    for (const b of bs) blocked[b] = 1;
    const seen = new Uint8Array(n);
    const out = [from];
    seen[from] = 1;
    for (let i = 0; i < out.length; i++) {
      const c = out[i];
      for (let d = 0; d < 4; d++) {
        const m = nb[c][d];
        if (m >= 0 && !blocked[m] && !seen[m]) {
          seen[m] = 1;
          out.push(m);
        }
      }
    }
    return out;
  };
  // 2x2 freeze check around cell `at` (local index): every cell of a 2x2
  // block wall-or-box, with a box off its goal inside, means no push saves it.
  const frozenAt = (bs, at) => {
    const boxSet = new Set(bs);
    const cx = cells[at] % w;
    const cy = (cells[at] / w) | 0;
    for (let ox = -1; ox <= 0; ox++) {
      for (let oy = -1; oy <= 0; oy++) {
        let allBlocked = true;
        let hasOffGoalBox = false;
        for (let a = 0; a < 2; a++) {
          for (let b = 0; b < 2; b++) {
            const gx = cx + ox + a;
            const gy = cy + oy + b;
            if (gx < 0 || gx >= w || gy < 0 || gy >= h) continue; // out of board: wall
            const gi = gy * w + gx;
            if (!idx.has(gi)) continue; // wall
            const li = idx.get(gi);
            if (boxSet.has(li)) {
              if (!goalSet.has(li)) hasOffGoalBox = true;
            } else {
              allBlocked = false;
            }
          }
        }
        if (allBlocked && hasOffGoalBox) return true;
      }
    }
    return false;
  };

  const startPlayer = idx.get(room.outside);
  const startCanon = Math.min(...regionOf(startBoxes, startPlayer));
  const sBoxes = [startBoxes.slice()];
  const sPlayer = [startPlayer]; // actual player cell after the push that made the state
  const sParent = [-1];
  const sPushBox = [-1];
  const sPushDir = [-1];
  const best = new Map([[keyOf(startBoxes, startCanon), 0]]);
  // Binary heap of [f, -g, stateIndex]; ties go to the deeper state.
  const heap = [];
  const hLess = (a, b) => a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const hPush = (e) => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!hLess(heap[i], heap[p])) break;
      const t = heap[i];
      heap[i] = heap[p];
      heap[p] = t;
      i = p;
    }
  };
  const hPop = () => {
    const top = heap[0];
    const last = heap.pop();
    if (heap.length > 0) {
      heap[0] = last;
      let i = 0;
      for (;;) {
        const l = i * 2 + 1;
        const r = l + 1;
        let m = i;
        if (l < heap.length && hLess(heap[l], heap[m])) m = l;
        if (r < heap.length && hLess(heap[r], heap[m])) m = r;
        if (m === i) break;
        const t = heap[i];
        heap[i] = heap[m];
        heap[m] = t;
        i = m;
      }
    }
    return top;
  };
  hPush([heuristic(startBoxes) * weight, 0, 0]);
  let goalState = -1;
  let expansions = 0;
  const budget = weight === 1 ? ASTAR_BUDGET : weight === 2 ? WEIGHTED_BUDGET : GREEDY_BUDGET;
  while (heap.length > 0) {
    const [, negG, si] = hPop();
    const g = -negG;
    const canon = Math.min(...regionOf(sBoxes[si], sPlayer[si]));
    if (g > (best.get(keyOf(sBoxes[si], canon)) ?? INF)) continue; // stale entry
    if (isGoal(sBoxes[si])) {
      goalState = si;
      break;
    }
    // One expansion costs one unit of the run-wide work budget as well as
    // this pass's own budget, so the whole run stays inside its wall budget.
    if (++expansions > budget || --workLeft < 0) return null;
    const bs = sBoxes[si];
    const region = regionOf(bs, sPlayer[si]);
    const regionSet = new Set(region);
    const boxSet = new Set(bs);
    for (let bi = 0; bi < k; bi++) {
      const b = bs[bi];
      for (let d = 0; d < 4; d++) {
        const stance = nb[b][d ^ 1];
        if (stance < 0 || !regionSet.has(stance)) continue;
        const target = nb[b][d];
        // A box may never be pushed out of a real room's door into the
        // hallway (it could never come back); a whole-level room has no door.
        if (target < 0 || (target === outsideIdx && !room.whole) || boxSet.has(target)) continue;
        if (dead[target]) continue;
        const nbs = bs.slice();
        nbs[bi] = target;
        nbs.sort((a, b2) => a - b2);
        if (frozenAt(nbs, target)) continue;
        const h = heuristic(nbs);
        if (h >= INF) continue;
        const nRegion = regionOf(nbs, b);
        const nCanon = Math.min(...nRegion);
        const key = keyOf(nbs, nCanon);
        const ng = g + 1;
        if ((best.get(key) ?? INF) <= ng) continue;
        best.set(key, ng);
        sBoxes.push(nbs);
        sPlayer.push(b);
        sParent.push(si);
        sPushBox.push(b);
        sPushDir.push(d);
        hPush([ng + h * weight, -ng, sBoxes.length - 1]);
      }
    }
  }
  if (goalState < 0) return null;

  // Reconstruct: replay the pushes from the start, BFS-walking the player to
  // each stance square.
  const pushes = [];
  let cur = goalState;
  while (sParent[cur] >= 0) {
    pushes.push([sPushBox[cur], sPushDir[cur]]);
    cur = sParent[cur];
  }
  pushes.reverse();
  const blocked = new Uint8Array(n);
  const curBoxes = startBoxes.slice();
  let player = startPlayer;
  let moves = '';
  for (const [box, d] of pushes) {
    for (let i = 0; i < n; i++) blocked[i] = 0;
    for (const b of curBoxes) blocked[b] = 1;
    const stance = nb[box][d ^ 1];
    const path = walkLocal(nb, blocked, player, stance);
    if (path === null) return null;
    moves += path + CH[d];
    curBoxes[curBoxes.indexOf(box)] = nb[box][d];
    curBoxes.sort((a, b) => a - b);
    player = box;
  }
  return { moves, endPlayerGlobal: cells[player] };

  // BFS walk on local indices, deterministic neighbour order u,d,l,r.
  function walkLocal(nbs, blockedArr, from, to) {
    if (from === to) return '';
    const prev = new Int32Array(n).fill(-1);
    const prevDir = new Int8Array(n).fill(-1);
    const q = [from];
    prev[from] = from;
    for (let qi = 0; qi < q.length; qi++) {
      const c = q[qi];
      for (let d = 0; d < 4; d++) {
        const m = nbs[c][d];
        if (m < 0 || blockedArr[m] || prev[m] !== -1) continue;
        prev[m] = c;
        prevDir[m] = d;
        if (m === to) {
          let s = '';
          let cur2 = m;
          while (cur2 !== from) {
            s = CH[prevDir[cur2]] + s;
            cur2 = prev[cur2];
          }
          return s;
        }
        q.push(m);
      }
    }
    return null;
  }
}

// ----------------------------------------------------------------- top level ---

/** Solve one level text, returning the full move string (or '' when stuck). */
function solve(text) {
  const level = parseLevel(text);
  const dec = decompose(level);
  if (!dec) {
    // Fallback: no hallway/room split was found, so try the level as one
    // room entered at the player's start, with the same pass ladder.
    const cells = [];
    for (let i = 0; i < level.w * level.h; i++) if (!level.wall[i]) cells.push(i);
    const whole = {
      cells,
      cellSet: new Set(cells),
      door: level.player,
      outside: level.player,
      boxes: level.boxes,
      goals: level.goals,
      whole: true,
    };
    for (const weight of [1, 2, 6]) {
      const res = solveRoom(level, whole, weight);
      if (res) return res.moves;
    }
    return '';
  }
  const { w, h, wall, boxes } = level;
  const floorAll = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) floorAll[i] = wall[i] ? 0 : 1;
  const boxAt = new Uint8Array(w * h);
  for (const b of boxes) boxAt[b] = 1;
  let player = level.player;
  let moves = '';
  for (const room of dec.rooms) {
    // Walk to the hallway cell outside this room's door.
    const toOutside = walkPath(w, h, floorAll, boxAt, player, room.outside);
    if (toOutside === null) return '';
    moves += toOutside;
    player = room.outside;
    // Solve the room; retry weighted, then greedy, when the optimal pass
    // runs out of budget, so a hard room still gets solved for some points.
    let res = solveRoom(level, room, 1);
    if (!res) res = solveRoom(level, room, 2);
    if (!res) res = solveRoom(level, room, 6);
    if (!res) return '';
    moves += res.moves;
    player = res.endPlayerGlobal;
    for (const b of room.boxes) boxAt[b] = 0;
    for (const g of room.goals) boxAt[g] = 1;
    // Walk out to the hallway (any hallway cell) for the next room.
    let outPath = null;
    let outEnd = -1;
    for (let x = 0; x < w; x++) {
      const hc = dec.hallRow * w + x;
      if (!floorAll[hc] || boxAt[hc]) continue;
      const p = walkPath(w, h, floorAll, boxAt, player, hc);
      if (p !== null) {
        outPath = p;
        outEnd = hc;
        break;
      }
    }
    if (outPath === null) return '';
    moves += outPath;
    player = outEnd;
  }
  return moves;
}

for (const id of readdirSync(I).sort()) {
  const moves = solve(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
