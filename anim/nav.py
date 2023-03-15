from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
import copy
import heapq
from typing import Any, Dict, List, Set, Tuple

import numpy as np

from anim.state import AgentState

from .lux import MOVE_DELTAS, ResourceType, UnitCargo, GameState, ActionType, UnitConfig

from .utils import back_dijkstra, in_bounds, NEIGHBORS, add_locs, sub_locs


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
        # BUG should be tu?
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


class NavActionType(Enum):
    BASIC = 0
    GOTO = 1
    TARGET = 2
    HIT = 3
    BASIC_OR = 4
    AVOID = 5


@dataclass
class UnitState:
    step: int
    unit_id: str
    unit_type: int
    pos: Tuple[int, int]
    power: int
    cargo: np.ndarray


@dataclass
class CostTable:
    move_power: np.ndarray
    transfer_power: np.ndarray
    transfer_cargo: np.ndarray

    @staticmethod
    def from_unit_cfg(unit_cfg: UnitConfig, rubble: np.ndarray):
        move_power = np.zeros((len(ActionType), *rubble.shape), dtype=np.int64)
        move_power[ActionType.MOVE.value] = np.floor(
            rubble * unit_cfg.RUBBLE_MOVEMENT_COST + unit_cfg.MOVE_COST
        )
        move_power[ActionType.DIG.value] = unit_cfg.DIG_COST
        move_power[ActionType.SELF_DESTRUCT.value] = unit_cfg.SELF_DESTRUCT_COST
        transfer_power = np.zeros((len(ActionType), len(ResourceType)), dtype=np.int64)
        transfer_power[ActionType.TRANSFER.value][ResourceType.POWER.value] = 1
        transfer_power[ActionType.PICKUP.value][ResourceType.POWER.value] = -1

        transfer_cargo = np.zeros((len(ActionType), len(ResourceType)), dtype=np.int64)
        transfer_cargo[ActionType.TRANSFER.value] = 1
        transfer_cargo[ActionType.PICKUP.value] = -1
        transfer_cargo[ActionType.TRANSFER.value][ResourceType.POWER.value] = 0
        transfer_cargo[ActionType.PICKUP.value][ResourceType.POWER.value] = 0
        return CostTable(move_power, transfer_power, transfer_cargo)

    def get_power_cost(self, loc: Tuple[int, int], action: np.ndarray):
        a = action[0].item()
        m = action[1].item()
        t = action[2].item()
        r = action[3].item()
        delta = tuple(MOVE_DELTAS[m])
        loc = add_locs(loc, delta)
        lc = self.move_power[(a, *loc)].item()
        tc = t * self.transfer_power[a, r].item()
        return lc + tc

    def get_cargo_cost(self, action: np.ndarray):
        a = action[0].item()
        t = action[2].item()
        r = action[3].item()
        return t * self.transfer_cargo[a, r].item()


@dataclass
class NavState:
    step: int
    cycle_length: int
    day_length: int
    cost_table: List[CostTable]
    units: Dict[str, UnitState]
    max_turns: int = 100
    search_limit: int = 10000
    occupied: Dict[Tuple[int, Tuple[int, int]], UnitState] = field(default_factory=dict)
    danger: Dict[Tuple[int, Tuple[int, int]], int] = field(default_factory=dict)
    actions: Dict[str, List[np.ndarray]] = field(
        default_factory=lambda: defaultdict(list)
    )

    @staticmethod
    def from_agent_state(state: AgentState):
        step = state.game.real_env_steps
        env_cfg = state.game.env_cfg
        units = {}
        for unit_id, unit in state.game.units[state.player].items():
            unit_type = 0 if unit.unit_type == "LIGHT" else 1
            power = unit.power - unit.unit_cfg.ACTION_QUEUE_POWER_COST
            cargo = np.zeros(len(ResourceType), dtype=np.int64)
            units[unit_id] = UnitState(
                step, unit_id, unit_type, tuple(unit.pos), power, cargo
            )
        cost_table = [
            CostTable.from_unit_cfg(env_cfg.ROBOTS["LIGHT"], state.game.board.rubble),
            CostTable.from_unit_cfg(env_cfg.ROBOTS["HEAVY"], state.game.board.rubble),
        ]

        return NavState(
            step,
            env_cfg.CYCLE_LENGTH,
            env_cfg.DAY_LENGTH,
            units,
            cost_table,
        )

    def update(self, unit_id: str, action: np.ndarray):
        unit = self.units[unit_id]
        a = action[0].item()
        m = action[1].item()
        t = action[2].item()
        r = action[3].item()
        power_cost = self.cost_table[unit.unit_type].get_power_cost(unit.pos, action)
        unit.power -= power_cost
        unit.step += 1
        if a == ActionType.MOVE.value:
            unit.pos = add_locs(unit.pos, tuple(MOVE_DELTAS[m]))
        unit.cargo[r] -= self.cost_table[unit.unit_type].get_cargo_cost(action)
        ts = unit.step, unit.pos
        self.occupied[ts] = copy.deepcopy(unit)
        self.actions[unit_id].append(action)


def get_power_cycle(
    power: int, start_step: int, max_step: int, cycle_length: int, day_length: int
):
    cycle = {}
    for step in range(start_step, max_step):
        cycle[step] = power
        power += (step % cycle_length) < day_length
    return cycle


def basic(state: NavState, unit_id: str, action: np.ndarray):
    state.update(unit_id, action)
    return True


def goto(
    state: NavState,
    unit_id: str,
    end: Tuple[int, int],
    cost: np.ndarray,
    h: np.ndarray,
):
    unit = state.units[unit_id]
    start = unit.pos
    start_step = unit.step
    min_power = unit.unit_type * 1000
    power = min_power + unit.power
    power_cycle = get_power_cycle(
        power, start_step, max_step, state.cycle_length, state.day_length
    )
    max_step = start_step + state.max_turns
    search_limit = state.search_limit
    occupied = state.occupied
    danger = state.danger
    came_from = {}
    seen = set()
    g = defaultdict(lambda: float("inf"))
    g[(start_step, start)] = 0
    f = defaultdict(lambda: float("inf"))
    f[(start_step, start)] = h[start]

    q = []
    heapq.heappush(q, (h[start], (start_step, start)))
    expanded = 0
    while len(q) > 0:
        if expanded >= search_limit:
            return None
        d, tu = heapq.heappop(q)
        t, u = tu
        if u == end:
            return single_astar_path(came_from, tu)
        # BUG should be tu?
        if u in seen:
            continue
        seen.add(u)
        if t + 1 >= max_step:
            continue

        expanded += 1

        for n, delta in enumerate(MOVE_DELTAS):
            v = add_locs(u, tuple(delta))
            tv = (t + 1, v)
            if not in_bounds(v, cost.shape):
                continue
            if tv in occupied:
                continue
            next_g = g[tu] + cost[(*u, n)]
            if next_g >= power_cycle[t + 1] - danger[tv] + 1:
                continue
            # TODO Handle move center
            if next_g < g[tv]:
                came_from[tv] = tu
                g[tv] = next_g
                next_f = next_g + h[v]
                f[tv] = next_f
                heapq.heappush(q, (next_f, tv))

    return None
