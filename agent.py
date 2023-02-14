from anim.state import AgentState
from lux.kit import obs_to_game_state, GameState
from lux.config import EnvConfig
from lux.utils import direction_to, my_turn_to_place_factory
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
        actions = dict()

        """
        optionally do forward simulation to simulate positions of units, lichen, etc. in the future
        from lux.forward_sim import forward_sim
        forward_obs = forward_sim(obs, self.env_cfg, n=2)
        forward_game_states = [obs_to_game_state(step + i, self.env_cfg, f_obs) for i, f_obs in enumerate(forward_obs)]
        """

        game_state = obs_to_game_state(step, self.env_cfg, obs)
        factories = game_state.factories[self.player]
        game_state.teams[self.player].place_first
        factory_tiles, factory_units = [], []
        for unit_id, factory in factories.items():
            if (
                factory.power >= self.env_cfg.ROBOTS["HEAVY"].POWER_COST
                and factory.cargo.metal >= self.env_cfg.ROBOTS["HEAVY"].METAL_COST
            ):
                actions[unit_id] = factory.build_heavy()
            if factory.water_cost(game_state) <= factory.cargo.water / 5 - 200:
                actions[unit_id] = factory.water()
            factory_tiles += [factory.pos]
            factory_units += [factory]
        factory_tiles = np.array(factory_tiles)

        units = game_state.units[self.player]
        ice_map = game_state.board.ice
        ice_tile_locations = np.argwhere(ice_map == 1)
        for unit_id, unit in units.items():

            # track the closest factory
            closest_factory = None
            adjacent_to_factory = False
            if len(factory_tiles) > 0:
                factory_distances = np.mean((factory_tiles - unit.pos) ** 2, 1)
                closest_factory_tile = factory_tiles[np.argmin(factory_distances)]
                closest_factory = factory_units[np.argmin(factory_distances)]
                adjacent_to_factory = (
                    np.mean((closest_factory_tile - unit.pos) ** 2) == 0
                )

                # previous ice mining code
                if unit.cargo.ice < 40:
                    ice_tile_distances = np.mean(
                        (ice_tile_locations - unit.pos) ** 2, 1
                    )
                    closest_ice_tile = ice_tile_locations[np.argmin(ice_tile_distances)]
                    if np.all(closest_ice_tile == unit.pos):
                        if unit.power >= unit.dig_cost(
                            game_state
                        ) + unit.action_queue_cost(game_state):
                            actions[unit_id] = [unit.dig(repeat=0, n=1)]
                    else:
                        direction = direction_to(unit.pos, closest_ice_tile)
                        move_cost = unit.move_cost(game_state, direction)
                        if (
                            move_cost is not None
                            and unit.power
                            >= move_cost + unit.action_queue_cost(game_state)
                        ):
                            actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
                # else if we have enough ice, we go back to the factory and dump it.
                elif unit.cargo.ice >= 40:
                    direction = direction_to(unit.pos, closest_factory_tile)
                    if adjacent_to_factory:
                        if unit.power >= unit.action_queue_cost(game_state):
                            actions[unit_id] = [
                                unit.transfer(direction, 0, unit.cargo.ice, repeat=0)
                            ]
                    else:
                        move_cost = unit.move_cost(game_state, direction)
                        if (
                            move_cost is not None
                            and unit.power
                            >= move_cost + unit.action_queue_cost(game_state)
                        ):
                            actions[unit_id] = [unit.move(direction, repeat=0, n=1)]
        return actions
