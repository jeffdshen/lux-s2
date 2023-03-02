import numpy as np

from .lux import obs_to_game_state, UnitConfig

from anim.utils import ZonesCache, named_factory_spots
from .state import AgentState, DijkstraCache
from .build import build
from .mine import make_mine


def make_cost(name: str, state: AgentState):
    rubble = state.game.board.rubble
    unit_config = state.game.env_cfg.ROBOTS[name]
    cost = np.floor(unit_config.MOVE_COST + rubble * unit_config.RUBBLE_MOVEMENT_COST)
    enemy_factories = state.game.factories[state.opp_player]
    for _, factory in enemy_factories.items():
        cost[factory.pos_slice] = float("inf")

    return cost


def agent_act(state: AgentState, step: int, obs, remainingOverageTime: int = 60):
    state.game = obs_to_game_state(step, state.env_cfg, obs)
    state.step = step
    state.actions = {}
    state.dcache = DijkstraCache()
    state.dcache.add_cost("LIGHT", make_cost("LIGHT", state))
    state.dcache.add_cost("HEAVY", make_cost("HEAVY", state))
    factory_spots = named_factory_spots(state.game.factories[state.player])
    state.zcache = ZonesCache(state.dcache)
    state.zcache.make_zones("FACTORY_SPOT", "LIGHT", factory_spots)
    state.zcache.make_zones("FACTORY_SPOT", "HEAVY", factory_spots)
    build(state)
    make_mine(state)
