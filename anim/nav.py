from collections import defaultdict
from dataclasses import dataclass
import heapq
from typing import List, Set, Tuple

import numpy as np

from .utils import back_dijkstra, in_bounds, NEIGHBORS


def extend_path(path: List[Tuple[int, int]], max_turns: int = 100):
    for _ in range(len(path), max_turns):
        path.append(path[-1])


# existing paths should each be max_turns long
# starts[i], ends[i] gives the desired start and end for agent i
# ends should not conflict
# starts should be in order of priority.
def multi_astar(
    starts: List[Tuple[int, int]],
    ends: List[Tuple[int, int]],
    costs: List[np.ndarray],
    existing_paths: List[List[Tuple[int, int]]],
    max_turns: int = 100,
):

    # time, loc -> occupied or not
    occupied = set()
    for path in existing_paths:
        for t, loc in enumerate(path):
            occupied.add((t, loc))

    paths = []
    for start, end, cost in zip(starts, ends, costs):
        h = back_dijkstra(end, cost)
        # TODO handle the case where end can be made unreachable
        # because of previous ends, e.g. previous units surround your
        # destination by some turn, and you can't get there in time.
        path = single_astar(start, end, cost, h, occupied, max_turns)
        paths.append(path)
        if path is None:
            continue
        for t, u in enumerate(path):
            occupied.add((t, u))
        for t in range(len(path), max_turns):
            occupied.add((t, path[-1]))
    return paths


def single_astar_path(came_from, end):
    total_path = [end[1]]
    current = end
    while current in came_from:
        current = came_from[current]
        total_path.append(current[1])
    total_path.reverse()
    return total_path


def single_astar(
    start: Tuple[int, int],
    end: Tuple[int, int],
    cost: np.ndarray,
    h: np.ndarray,
    occupied: Set[Tuple[int, Tuple[int, int]]],
    max_turns: int = 100,
):
    came_from = {}
    seen = set()
    g = defaultdict(lambda: float("inf"))
    g[(0, start)] = 0
    f = defaultdict(lambda: float("inf"))
    f[(0, start)] = h[start]

    q = []
    heapq.heappush(q, (h[start], (0, start)))
    expanded = 0
    while len(q) > 0:
        d, tu = heapq.heappop(q)
        t, u = tu
        if u == end:
            return single_astar_path(came_from, tu)
        if u in seen:
            continue
        seen.add(u)
        if t + 1 >= max_turns:
            continue

        expanded += 1

        # TODO consider moving center
        for n in NEIGHBORS:
            v = (u[0] + n[0], u[1] + n[1])
            tv = (t + 1, v)
            if not in_bounds(v, cost.shape):
                continue
            if tv in occupied:
                continue
            next_g = g[tu] + cost[v]
            if next_g < g[tv]:
                came_from[tv] = tu
                g[tv] = next_g
                next_f = next_g + h[v]
                f[tv] = next_f
                heapq.heappush(q, (next_f, tv))

    return None
