import numpy as np

from lux.utils import direction_to

from .lux import obs_to_game_state

from .state import AgentState, DijkstraCache
from .build import build
from .mine import make_mine


def agent_act(state: AgentState, step: int, obs, remainingOverageTime: int = 60):
    state.game = obs_to_game_state(step, state.env_cfg, obs)
    state.step = step
    state.actions = {}
    state.dijkstra = DijkstraCache()
    state.dijkstra.add_cost("LIGHT", np.floor(1 + state.game.board.rubble * 0.05))
    state.dijkstra.add_cost("HEAVY", np.floor(20 + state.game.board.rubble * 1))
    build(state)
    make_mine(state)
