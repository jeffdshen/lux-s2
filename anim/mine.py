from dataclasses import dataclass
from enum import Enum
import sys
from typing import Dict, List, Tuple

import numpy as np

from anim.state import AgentState
from .lux import (
    Unit,
    Factory,
    MOVE_DELTAS,
    TO_DIRECTION,
    ActionType,
    Board,
    ResourceType,
)

from .nav import extend_path, multi_astar
from .utils import (
    Zones,
    dijkstra,
    action_queue_to_path,
    factory_spots,
    path_to_action_queue,
    factory_zones,
)


def get_mine_locations(zones: Zones, board: Board):
    locs = []
    ices = np.argwhere(board.ice)
    for ice in ices:
        dist = zones.dist[tuple(ice)]
        fid = zones.to_id[zones.zone[tuple(ice)]]
        locs.append((dist, fid, 0, tuple(ice)))

    ores = np.argwhere(board.ore)
    for ore in ores:
        dist = zones.dist[tuple(ore)]
        fid = zones.to_id[zones.zone[tuple(ore)]]
        locs.append((dist, fid, 1, tuple(ore)))

    locs.sort()
    return locs


class ObjectiveType(Enum):
    MINE = 0
    DROPOFF = 1


@dataclass
class Objective:
    unit: Unit
    loc: Tuple[int, int]
    obj_type: int


def make_mine(state: AgentState):
    units = state.game.units[state.player]
    factories = state.game.factories[state.player]
    board = state.game.board
    zones = factory_zones(board.rubble, factories)
    mines = get_mine_locations(zones, board)
    spots = factory_spots(factories)
    unit_paths = get_paths(units)
    objective_types = {}
    for mine in mines:
        objective_types[mine[-1]] = ObjectiveType.MINE.value
    for spot in spots:
        objective_types[spot] = ObjectiveType.DROPOFF.value
    spots.reverse()
    mines.reverse()

    # TODO run a matching algorithm between bots and mines/spots
    # could create a distance function estimating the value for each (bot, mine)
    # and then match to maximize value.
    existing_paths = []
    objs: List[Objective] = []
    for unit_id, unit in units.items():
        unit_path = unit_paths[unit_id]
        if len(unit_path) > 1:
            loc = unit_path[-1]
            # print(loc, unit, file=sys.stderr)
            objective_types.pop(loc, None)
            existing_paths.append(unit_path)
            continue
        objective_types.pop(tuple(unit.pos), None)

    for unit_id, unit in units.items():
        unit_path = unit_paths[unit_id]
        if len(unit_path) > 1:
            continue
        elif unit.cargo.ice + unit.cargo.metal >= 5:
            while spots and spots[-1] not in objective_types:
                spots.pop()
            if spots:
                loc = spots.pop()
                objs.append(Objective(unit, loc, objective_types.pop(loc)))
            else:
                existing_paths.append(unit_path)
        else:
            if unit.power < unit.unit_cfg.DIG_COST * 5:
                existing_paths.append(unit_paths[unit_id])
                state.actions[unit_id] = [
                    unit.pickup(
                        ResourceType.POWER.value,
                        unit.unit_cfg.DIG_COST * 6 - unit.power,
                    )
                ]
                continue
            while mines and mines[-1][-1] not in objective_types:
                mines.pop()

            if mines:
                loc = mines[-1][-1]
                dist = state.dijkstra.backward(loc, unit.unit_type)
                if unit.power >= 2 * dist[tuple(unit.pos)] + unit.unit_cfg.DIG_COST * 6:
                    loc = mines.pop()[-1]
                    objs.append(Objective(unit, loc, objective_types.pop(loc)))
                    continue

            existing_paths.append(unit_path)

    # print(objs, file=sys.stderr)
    for path in existing_paths:
        extend_path(path)

    # print("OBJS: ", objs, file=sys.stderr)
    starts = [tuple(obj.unit.pos) for obj in objs]
    ends = [obj.loc for obj in objs]
    cost_dict = {
        "LIGHT": np.floor(1 + board.rubble * 0.05),
        "HEAVY": np.floor(20 + board.rubble),
    }
    costs = [cost_dict[obj.unit.unit_type] for obj in objs]
    paths = multi_astar(starts, ends, costs, existing_paths)
    # print("PATHS: ", paths, file=sys.stderr)
    # print("ALL_PATHS", existing_paths, file=sys.stderr)
    for obj, path in zip(objs, paths):
        if path is None:
            continue
        q = path_to_action_queue(obj.unit, path)
        if obj.obj_type == ObjectiveType.MINE.value:
            q.append(obj.unit.dig(n=5))
        else:
            if obj.unit.cargo.ice > obj.unit.cargo.ore:
                transfer_type = ResourceType.ICE.value
                transfer_amount = obj.unit.cargo.ice
            else:
                transfer_type = ResourceType.ORE.value
                transfer_amount = obj.unit.cargo.ore
            q.append(
                obj.unit.transfer(TO_DIRECTION[0, 0], transfer_type, transfer_amount)
            )
        state.actions[obj.unit.unit_id] = q[:20]
    # print("ACTIONS: ", state.actions, file=sys.stderr)


def get_paths(units: Dict[str, Unit]):
    paths = {}
    for unit_id, unit in units.items():
        path = []
        path = action_queue_to_path(unit.pos, unit.action_queue)
        paths[unit_id] = path
    return paths
