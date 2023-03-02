from anim.actor import agent_act
from anim.state import AgentState
from lux.kit import obs_to_game_state
from lux.config import EnvConfig
from lux.utils import direction_to
import numpy as np
import sys

from anim.bid import agent_early_setup


class Agent:
    def __init__(self, player: str, env_cfg: EnvConfig) -> None:
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        self.env_cfg: EnvConfig = env_cfg
        self.state = AgentState(
            env_cfg=env_cfg, player=player, opp_player=self.opp_player
        )
        np.random.seed(0)
        np.set_printoptions(threshold=np.inf)

    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        return agent_early_setup(self.state, step, obs, remainingOverageTime)

    def act(self, step: int, obs, remainingOverageTime: int = 60):
        agent_act(self.state, step, obs, remainingOverageTime)
        return self.state.actions
