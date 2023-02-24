from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .lux import EnvConfig, GameState
from .utils import DijkstraCache


@dataclass
class AgentState:
    env_cfg: EnvConfig
    player: str
    opp_player: str
    game: Optional[GameState] = None
    step: int = 0
    sorted_scores: Optional[List] = None
    actions: Dict = None
    dijkstra: DijkstraCache = None
