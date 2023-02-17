from dataclasses import dataclass
from typing import Dict, List, Tuple
import heapq

import numpy as np

from .lux import Unit, MOVE_DELTAS, TO_DIRECTION, ActionType, Factory
from .state import AgentState

import sys

def in_bounds(u, size):
    return u[0] >= 0 and u[1] >= 0 and u[0] < size[0] and u[1] < size[1]


NEIGHBORS = [
    (a, b) for a in range(-1, 2) for b in range(-1, 2) if 0 < abs(a) + abs(b) <= 1
]


def dijkstra(start: Tuple[int, int], cost: np.ndarray):
    dist = np.full_like(cost, np.inf, dtype=np.float32)
    seen = set()
    dist[start] = 0
    q = []
    heapq.heappush(q, (dist[start], start))
    while len(q) > 0:
        d, u = heapq.heappop(q)
        if u in seen:
            continue
        seen.add(u)
        for n in NEIGHBORS:
            v = (u[0] + n[0], u[1] + n[1])
            if not in_bounds(v, cost.shape):
                continue
            e = cost[v]
            next_d = dist[u] + e
            if next_d < dist[v]:
                dist[v] = next_d
                heapq.heappush(q, (dist[v], v))

    return dist


def back_dijkstra(end: Tuple[int, int], cost: np.ndarray):
    dist = np.full_like(cost, np.inf, dtype=np.float32)
    seen = set()
    dist[end] = 0
    q = []
    heapq.heappush(q, (dist[end], end))
    while len(q) > 0:
        d, u = heapq.heappop(q)
        if u in seen:
            continue
        seen.add(u)
        for n in NEIGHBORS:
            v = (u[0] + n[0], u[1] + n[1])
            if not in_bounds(v, cost.shape):
                continue
            e = cost[u]
            next_d = dist[u] + e
            if next_d < dist[v]:
                dist[v] = next_d
                heapq.heappush(q, (dist[v], v))

    return dist


# only works for own units (cause recharge makes things complicated)
def action_queue_to_path(loc: np.ndarray, action_queue: np.ndarray):
    loc = np.copy(loc)
    path = [tuple(loc)]
    for action in action_queue:
        action = tuple(action)
        move = 0
        if action[0] == ActionType.MOVE.value:
            move = action[1]
        move_delta = MOVE_DELTAS[move]
        for i in range(action[5]):
            loc += move_delta
            path.append(tuple(loc))
    return path


def path_to_action_queue(unit: Unit, path: List[Tuple[int, int]]):
    deltas = [(a[0] - b[0], a[1] - b[1]) for a, b in zip(path[1:], path[:-1])]
    dirs = [TO_DIRECTION[delta] for delta in deltas]
    cmp_dirs = []
    for d in dirs:
        if not cmp_dirs:
            cmp_dirs.append([d, 1])
            continue
        if cmp_dirs[-1][0] == d:
            cmp_dirs[-1][1] += 1
        else:
            cmp_dirs.append([d, 1])

    return [unit.move(d, n=n) for d, n in cmp_dirs]


@dataclass
class Zones:
    dist: np.ndarray
    zone: np.ndarray
    to_id: Dict[int, str]
    from_id: Dict[str, int]


def factory_zones(rubble: np.ndarray, factories: Dict[str, Factory]):
    dists = []
    to_id = {}
    from_id = {}
    cost = np.floor(1 + rubble * 0.05)
    for idx, (fid, factory) in enumerate(factories.items()):
        dists.append(dijkstra(tuple(factory.pos), cost))
        to_id[idx] = fid
        from_id[fid] = idx

    dists = np.stack(dists)
    dist = np.min(dists, axis=0)
    zone = np.argmin(dists, axis=0)
    return Zones(dist, zone, to_id, from_id)


def _calculate_factory_neighbors():
    factory = set([(a, b) for a in range(-1, 2) for b in range(-1, 2)])
    factory_neighbors = set(
        [(f[0] + n[0], f[1] + n[1]) for f in factory for n in NEIGHBORS]
    )
    return factory_neighbors.difference(factory)

FACTORY_NEIGHBORS = list(_calculate_factory_neighbors())
FACTORY_SPOTS = [(a, b) for a in range(-1, 2) for b in range(-1, 2) if 0 < abs(a) + abs(b)]

def factory_outer(factories: Dict[str, Factory]):
    outer = set()
    for fid, factory in factories.items():
        for n in FACTORY_NEIGHBORS:
            loc = factory.pos + n
            outer.add(tuple(loc))
    return list(outer)

def factory_spots(factories: Dict[str, Factory]):
    spots = set()
    for fid, factory in factories.items():
        for n in FACTORY_SPOTS:
            loc = factory.pos + n
            spots.add(tuple(loc))
    return list(spots)