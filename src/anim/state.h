#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include <Eigen/Dense>

#include "anim/lux.h"
#include "anim/utils.h"

namespace anim {
struct Actions {
  std::map<std::string, lux::FactoryAction> factories;
  std::map<std::string, std::vector<lux::UnitAction>> units;

  json to_json() {
    json actions = json::object();
    for (auto& [unit_id, action] : factories) {
      actions[unit_id] = action;
    }
    for (auto& [unit_id, action] : units) {
      actions[unit_id] = action;
    }
    return actions;
  }
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
  std::map<std::string, std::map<std::string, lux::Unit>> units;
  std::map<std::string, lux::Team> teams;
  std::map<std::string, std::map<std::string, lux::Factory>> factories;
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
    units = obs.units;
    teams = obs.teams;
    factories = obs.factories;
    real_env_steps = obs.real_env_steps;
    config = obs.config;
  }
};

struct AgentState {
  lux::EnvConfig env_cfg;
  GameState game;
  std::string player;
  std::string opp_player;
  int64_t step = 0;
  int64_t overage_time = 0;
  std::vector<std::pair<double, Loc>> sorted_scores;
  Actions actions;
  DijkstraCache dcache;
  ZonesCache zcache;
};

inline Eigen::ArrayXXd make_cost(
    const std::string& name, const AgentState& state) {
  auto& rubble = state.game.board.rubble;
  auto& unit_cfg = state.env_cfg.ROBOTS[name];
  Eigen::ArrayXXd cost =
      (unit_cfg.MOVE_COST + rubble * unit_cfg.RUBBLE_MOVEMENT_COST).floor();
  auto& enemy_factories = state.game.factories.at(state.opp_player);
  for (auto& [_, factory] : enemy_factories) {
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
  state.player = player;
  state.opp_player = player == "player_0" ? "player_1" : "player_0";
  state.step = step;
  state.overage_time = remainingOverageTime;

  // Don't reset sorted_scores

  state.actions = {};

  state.dcache = DijkstraCache{};
  state.dcache.add_cost("LIGHT", make_cost("LIGHT", state));
  state.dcache.add_cost("HEAVY", make_cost("HEAVY", state));

  auto factory_spots = named_factory_spots(state.game.factories[state.player]);
  state.zcache = ZonesCache{};
  state.zcache.dcache = &state.dcache;
  state.zcache.make_zones("FACTORY_SPOT", "LIGHT", factory_spots);
  state.zcache.make_zones("FACTORY_SPOT", "HEAVY", factory_spots);
}

} // namespace anim
