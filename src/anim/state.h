#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "anim/lux.h"
#include "anim/utils.h"

namespace anim {
struct Actions {
  std::map<size_t, lux::FactoryAction> factories;
  std::map<size_t, std::vector<lux::UnitAction>> units;
};

struct BoardState {
  Eigen::ArrayXXd ice;
  Eigen::ArrayXXd lichen;
  Eigen::ArrayXXd lichen_strains;
  Eigen::ArrayXXd ore;
  Eigen::ArrayXXd rubble;
  Eigen::ArrayXXd valid_spawns_mask;
  Eigen::ArrayXXd factory_occupancy;
  int64_t factories_per_team;
};

struct GameState {
  BoardState board;
  std::vector<std::vector<lux::Unit>> units;
  std::vector<lux::Team> teams;
  std::vector<std::vector<lux::Factory>> factories;
  int64_t real_env_steps;
  lux::EnvConfig config;

  void reset(const lux::Observation& obs) {
    board.ice = to_eigen(obs.board.ice);
    board.lichen = to_eigen(obs.board.lichen);
    board.lichen_strains = to_eigen(obs.board.lichen_strains);
    board.ore = to_eigen(obs.board.ore);
    board.rubble = to_eigen(obs.board.rubble);
    board.valid_spawns_mask = to_eigen(obs.board.valid_spawns_mask);
    board.factory_occupancy = to_eigen(obs.board.factory_occupancy);

    units.clear();
    teams.clear();
    factories.clear();

    for (size_t i = 0; i < 2; i++) {
      auto player = "player_" + std::to_string(i);

      {
        units.emplace_back();
        auto& all_units = obs.units.at(player);
        units[i].reserve(all_units.size());
        for (auto& [_, unit] : all_units) {
          units[i].emplace_back(unit);
        }
      }

      // teams doesn't exist at bidding.
      if (!obs.teams.contains(player)) {
        teams.emplace_back();
      } else {
        teams.emplace_back(obs.teams.at(player));
      }

      {
        factories.emplace_back();
        auto& all_factories = obs.factories.at(player);
        factories[i].reserve(all_factories.size());
        for (auto& [_, factory] : all_factories) {
          factories[i].emplace_back(factory);
        }
      }
    }
    real_env_steps = obs.real_env_steps;
    config = obs.config;
  }
};

struct AgentState {
  lux::EnvConfig env_cfg;
  GameState game;
  std::size_t player;
  std::size_t opp_player;
  int64_t step = 0;
  int64_t overage_time = 0;
  std::vector<std::pair<double, Loc>> sorted_scores;
  Actions actions;
  DijkstraCache dcache;
  ZonesCache zcache;
  std::vector<std::vector<Loc>> factory_spots;

  json get_actions_json() {
    json res = json::object();
    for (auto& [idx, action] : actions.factories) {
      std::string unit_id = game.factories[player][idx].unit_id;
      res[unit_id] = action;
    }
    for (auto& [idx, action] : actions.units) {
      std::string unit_id = game.units[player][idx].unit_id;
      res[unit_id] = action;
    }
    return res;
  }
};

inline Eigen::ArrayXXd make_cost(
    const std::string& name, const AgentState& state) {
  auto& rubble = state.game.board.rubble;
  auto& unit_cfg = state.env_cfg.ROBOTS[name];
  Eigen::ArrayXXd cost =
      (unit_cfg.MOVE_COST + rubble * unit_cfg.RUBBLE_MOVEMENT_COST).floor();
  auto& enemy_factories = state.game.factories.at(state.opp_player);
  for (auto& factory : enemy_factories) {
    auto& pos = factory.pos;
    cost.block(pos.x - 1, pos.y - 1, 3, 3) = INF;
  }
  return cost;
}

inline void state_reset(
    AgentState& state,
    const std::string& player,
    int64_t step,
    const lux::Observation& obs,
    int64_t remainingOverageTime) {
  state.env_cfg = obs.config;
  state.game.reset(obs);
  state.player = player == "player_1";
  state.opp_player = 1 - state.player;
  state.step = step;
  state.overage_time = remainingOverageTime;

  // Don't reset sorted_scores

  state.actions = {};

  state.dcache = DijkstraCache{};
  auto p0 = make_cost("LIGHT", state);
  auto p1 = make_cost("HEAVY", state);
  state.dcache.add_cost("P0", p0 + EPS);
  state.dcache.add_cost("P1", p1 + EPS);
  state.dcache.add_cost("T0", p0 * EPS + 1);
  state.dcache.add_cost("T1", p1 * EPS + 1);

  state.factory_spots = get_factory_spots(state.game.factories[state.player]);
  state.zcache = ZonesCache{};
  state.zcache.dcache = &state.dcache;
  state.zcache.make_zones("FACTORY_SPOT", "P0", state.factory_spots);
  state.zcache.make_zones("FACTORY_SPOT", "P1", state.factory_spots);
}

} // namespace anim
