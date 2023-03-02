from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .lux import EnvConfig, GameState
from .utils import DijkstraCache, ZonesCache


@dataclass
class AgentState:
    env_cfg: EnvConfig
    player: str
    opp_player: str
    game: Optional[GameState] = None
    step: int = 0
    sorted_scores: Optional[List] = None
    actions: Optional[Dict] = None
    dcache: Optional[DijkstraCache] = None
    zcache: Optional[ZonesCache] = None
