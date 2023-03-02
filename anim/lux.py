from enum import Enum

from lux.cargo import UnitCargo
from lux.config import UnitConfig, EnvConfig
from lux.factory import Factory
from lux.forward_sim import forward_sim
from lux.kit import obs_to_game_state, Board, GameState
from lux.team import Team
from lux.unit import Unit
import lux.unit

from lux.utils import my_turn_to_place_factory

MOVE_DELTAS = lux.unit.move_deltas
TO_DIRECTION = {tuple(delta): i for i, delta in enumerate(MOVE_DELTAS)}


class ActionType(Enum):
    MOVE = 0
    TRANSFER = 1
    PICKUP = 2
    DIG = 3
    SELF_DESTRUCT = 4
    RECHARGE = 5


class UnitType(Enum):
    LIGHT = 0
    HEAVY = 1


class ResourceType(Enum):
    ICE = 0
    ORE = 1
    WATER = 2
    METAL = 3
    POWER = 4
