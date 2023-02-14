from dataclasses import dataclass
from typing import List, Optional

from lux.config import EnvConfig
from lux.kit import GameState

@dataclass
class AgentState:
    env_cfg: EnvConfig
    player: str
    opp_player: str
    game: Optional[GameState] = None
    step: int = 0
    sorted_scores: Optional[List] = None
