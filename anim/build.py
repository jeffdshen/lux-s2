from typing import Dict, Tuple
from collections import Counter

import numpy as np

from .lux import Factory, Unit
from .state import AgentState

def can_build(factory: Factory, robot_type: str, occupied: np.ndarray, state: AgentState):
    if factory.power < state.env_cfg.ROBOTS[robot_type].POWER_COST:
        return False
    if factory.cargo.metal < state.env_cfg.ROBOTS[robot_type].METAL_COST:
        return False
    if occupied[tuple(factory.pos)]:
        return False

    return True

def get_occupied_board(shape: Tuple[int, int], units: Dict[str, Unit]):
    occupied = np.zeros(shape, dtype=np.int64)
    for unit_id, unit in units.items():
        occupied[tuple(unit.pos)] = 1
    return occupied

def count_units(units: Dict[str, Unit]): 
    return Counter([unit.unit_type for _, unit in units.items()])


def build(state: AgentState):
    factories = state.game.factories[state.player]
    map_shape = state.game.board.rubble.shape
    occupied = get_occupied_board(map_shape, state.game.units[state.player])
    unit_counts = count_units(state.game.units[state.player])
    for unit_id, factory in factories.items():
        if can_build(factory, "HEAVY", occupied, state):
            state.actions[unit_id] = factory.build_heavy()
            continue
        if unit_counts["HEAVY"] > 0.2 * unit_counts["LIGHT"]:
            if can_build(factory, "LIGHT", occupied, state):
                state.actions[unit_id] = factory.build_light()
                continue
        if factory.water_cost(state.game) <= factory.cargo.water / 2 - 200:
            state.actions[unit_id] = factory.water()
        