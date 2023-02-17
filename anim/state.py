from dataclasses import dataclass
from typing import Dict, List, Optional

from .lux import EnvConfig, GameState


@dataclass
class AgentState:
    env_cfg: EnvConfig
    player: str
    opp_player: str
    game: Optional[GameState] = None
    step: int = 0
    sorted_scores: Optional[List] = None
    actions: Dict = None
