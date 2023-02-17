import numpy as np

from lux.utils import direction_to

from .lux import obs_to_game_state

from .state import AgentState
from .build import build
from .mine import make_mine


def agent_act(state: AgentState, step: int, obs, remainingOverageTime: int = 60):
    state.game = obs_to_game_state(step, state.env_cfg, obs)
    state.step = step
    state.actions = {}
    build(state)
    make_mine(state)
