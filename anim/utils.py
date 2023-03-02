from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple
import heapq

import numpy as np

from .lux import Unit, MOVE_DELTAS, TO_DIRECTION, ActionType, Factory, UnitConfig, GameState

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


@dataclass
class Zones:
    dist: np.ndarray
    zone: np.ndarray
    to_loc: Dict[int, Tuple[int, int]]
    from_loc: Dict[Tuple[int, int], int]
    to_id: Dict[int, str]


class DijkstraCache:
    def __init__(self):
        self.f_dists: Dict[Tuple[Tuple[int, int], str], np.ndarray] = {}
        self.b_dists: Dict[Tuple[Tuple[int, int], str], np.ndarray] = {}
        self.costs: Dict[str, np.ndarray] = {}

    def add_cost(self, name: str, cost: np.ndarray):
        self.costs[name] = cost

    def forward(self, start: Tuple[int, int], cost_name: str):
        if (start, cost_name) not in self.f_dists:
            cost = self.costs[cost_name]
            self.f_dists[start, cost_name] = dijkstra(start, cost)

        return self.f_dists[start, cost_name]

    def backward(self, end: Tuple[int, int], cost_name: str):
        if (end, cost_name) not in self.b_dists:
            cost = self.costs[cost_name]
            self.b_dists[end, cost_name] = back_dijkstra(end, cost)

        return self.b_dists[end, cost_name]


class ZonesCache:
    def __init__(self, dcache: DijkstraCache):
        self.dcache = dcache
        self.zones: Dict[Tuple[str, str], Zones] = {}

    def get_zone(self, zone_type: str, cost_name: str):
        return self.zones[zone_type, cost_name]

    def make_zones(
        self, zone_type: str, cost_name: str, locs: List[Tuple[Tuple[int, int], str]]
    ):
        dists = []
        to_loc = {}
        from_loc = {}
        to_id = {}
        for idx, (loc, name) in enumerate(locs):
            dists.append(self.dcache.backward(loc, cost_name))
            to_id[idx] = name
            to_loc[idx] = loc
            from_loc[loc] = idx

        dists = np.stack(dists)
        dist = np.min(dists, axis=0)
        zone = np.argmin(dists, axis=0)

        self.zones[zone_type, cost_name] = Zones(dist, zone, to_loc, from_loc, to_id)


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


def _calculate_factory_neighbors():
    factory = set([(a, b) for a in range(-1, 2) for b in range(-1, 2)])
    factory_neighbors = set(
        [(f[0] + n[0], f[1] + n[1]) for f in factory for n in NEIGHBORS]
    )
    return factory_neighbors.difference(factory)


FACTORY_NEIGHBORS = list(_calculate_factory_neighbors())
FACTORY_SPOTS = [
    (a, b) for a in range(-1, 2) for b in range(-1, 2) if 0 < abs(a) + abs(b)
]


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


def named_factory_spots(factories: Dict[str, Factory]):
    spots = list()
    for fid, factory in factories.items():
        for n in FACTORY_SPOTS:
            loc = factory.pos + n
            spots.append((tuple(loc), fid))
    return spots


def add_locs(u: Tuple[int, int], v: Tuple[int, int]):
    return (u[0] + v[0], u[1] + v[1])


def sub_locs(u: Tuple[int, int], v: Tuple[int, int]):
    return (u[0] - v[0], u[1] - v[1])
