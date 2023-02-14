# Module to handle the bidding and factory placement stage.

from typing import List, Tuple
import heapq

import numpy as np

from lux.kit import obs_to_game_state, GameState
from lux.config import EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory

from .state import AgentState

neighbors = [
    (a, b) for a in range(-1, 2) for b in range(-1, 2) if 0 < abs(a) + abs(b) <= 1
]


def in_bounds(u, size):
    return u[0] >= 0 and u[1] >= 0 and u[0] < size[0] and u[1] < size[1]


def back_dijkstra(start: Tuple[int, int], rubble: np.ndarray):
    dist = np.full_like(rubble, np.inf, dtype=np.float32)
    seen = set()
    dist[start] = 0
    q = []
    heapq.heappush(q, (dist[start], start))
    while len(q) > 0:
        d, u = heapq.heappop(q)
        if u in seen:
            continue
        seen.add(u)
        for n in neighbors:
            v = (u[0] + n[0], u[1] + n[1])
            if not in_bounds(v, rubble.shape):
                continue
            e = 1 + 0.05 * rubble[u]
            next_d = dist[u] + e
            if next_d < dist[v]:
                dist[v] = next_d
                heapq.heappush(q, (dist[v], v))

    return dist


def get_factory_dist(dist: np.ndarray):
    factory_dist = np.full_like(dist, np.inf)
    m, n = dist.shape
    for i in range(0, 3):
        for j in range(0, 3):
            k = abs(i - 1) + abs(j - 1)
            d = dist[i : i + m - 2, j : j + n - 2]
            factory_dist[1 : m - 1, 1 : n - 1] = np.minimum(
                factory_dist[1 : m - 1, 1 : n - 1], d + k
            )

    return factory_dist


def get_score(rubble: np.ndarray, resource: np.ndarray, eps=1e-6):
    locs = np.argwhere(resource)
    score = np.zeros_like(rubble, dtype=np.float32)
    for loc in locs:
        dist = get_factory_dist(back_dijkstra(tuple(loc), rubble))
        score += 1 / (dist + eps)
    return score


def get_scores(rubble: np.ndarray, ice: np.ndarray, ore: np.ndarray):
    return get_score(rubble, ice), get_score(rubble, ore)


def get_sorted_scores(valid_spawns_mask: np.ndarray, factory_overall: np.ndarray):
    locs = np.argwhere(valid_spawns_mask)
    sorted_scores = []
    for loc in locs:
        loc = tuple(loc)
        overall = factory_overall[loc].item()
        sorted_scores.append((overall, loc))
    sorted_scores.sort(reverse=True)
    return sorted_scores


def make_sorted_scores(game_state: GameState):
    rubble = game_state.board.rubble
    ice = game_state.board.ice
    ore = game_state.board.ore
    spawns = game_state.board.valid_spawns_mask
    ice_score, ore_score = get_scores(rubble, ice, ore)
    overall = ice_score * ore_score
    sorted_scores = get_sorted_scores(spawns, overall)
    return sorted_scores


SortedScoresType = List[Tuple[float, Tuple[int, int]]]


def make_bid(state: AgentState):
    return dict(faction="AlphaStrike", bid=0)


def place_factory(state: AgentState):
    factories_to_place = state.game.teams[state.player].factories_to_place
    my_turn_to_place = my_turn_to_place_factory(
        state.game.teams[state.player].place_first, state.step
    )
    if not (factories_to_place > 0 and my_turn_to_place):
        return dict()

    valid_spawns_mask = state.game.board.valid_spawns_mask
    for score, loc in state.sorted_scores:
        if valid_spawns_mask[loc].item():
            return dict(spawn=loc, metal=150, water=150)
    return dict()


def agent_early_setup(
    state: AgentState, step: int, obs, remainingOverageTime: int = 60
):
    state.game = obs_to_game_state(step, state.env_cfg, obs)
    state.step = step
    if step == 0:
        state.sorted_scores = make_sorted_scores(state.game)
        return make_bid(state)
    else:
        return place_factory(state)
