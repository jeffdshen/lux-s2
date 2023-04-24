#pragma once

#include <cstdint>
#include <memory>
#include <unordered_map>
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
  std::vector<std::vector<Loc>> enemy_adj;
  std::vector<int64_t> water_costs;
  std::vector<double> free_factory_power;
  RubbleScores rubble_scores;
  RubbleScores lichen_scores;
  std::vector<TeamLichenTable> team_lichen;
  std::vector<double> prev_value;
  std::vector<double> next_value;
  std::unordered_map<std::string, double> mapped_value;

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

  std::vector<lux::Unit>& my_units() { return game.units[player]; }

  const std::vector<lux::Unit>& my_units() const { return game.units[player]; }

  lux::Unit& my_unit(size_t i) { return my_units()[i]; }

  const lux::Unit& my_unit(size_t i) const { return my_units()[i]; }

  void add_mapped_value() {
    for (size_t i = 0; i < next_value.size(); i++) {
      mapped_value[my_unit(i).unit_id] = next_value[i];
    }
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

inline std::vector<TeamLichenTable> make_team_lichen(const AgentState& state) {
  std::vector<TeamLichenTable> team_lichen;
  auto& lichen = state.game.board.lichen;
  auto& strains = state.game.board.lichen_strains;
  for (size_t i = 0; i < 2; i++) {
    auto zeros = Eigen::ArrayXXd::Zero(lichen.rows(), lichen.cols());
    team_lichen.emplace_back();
    team_lichen.back().lichen = zeros;
    team_lichen.back().factory_id = zeros;
  }
  std::unordered_map<int32_t, size_t> to_team;
  std::unordered_map<int32_t, size_t> to_factory;
  for (size_t i = 0; i < 2; i++) {
    for (size_t j = 0; j < state.game.factories[i].size(); j++) {
      auto& factory = state.game.factories[i][j];
      to_team[factory.strain_id] = i;
      to_factory[factory.strain_id] = j;
    }
  }
  for (Eigen::Index i = 0; i < lichen.rows(); i++) {
    for (Eigen::Index j = 0; j < lichen.cols(); j++) {
      if (strains(i, j) < 0) {
        continue;
      }

      auto strain = static_cast<int32_t>(strains(i, j));
      if (!to_team.contains(strain)) {
        // weird, should never happen
        continue;
      }

      auto& table = team_lichen[to_team.at(strain)];
      table.lichen(i, j) = lichen(i, j);
      table.factory_id(i, j) = to_factory[strain];
    }
  }
  return team_lichen;
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
  auto R = state.game.board.rubble + state.game.board.ice * 1000 +
      state.game.board.ore * 1000 + EPS;
  state.dcache.add_cost("R", R);

  state.factory_spots = get_factory_spots(state.game.factories[state.player]);
  state.enemy_adj = get_factory_adjacent(
      state.game.factories[state.opp_player], shape(state.game.board.ice));
  state.zcache = ZonesCache{};
  state.zcache.dcache = &state.dcache;
  state.zcache.make_zones("FACTORY_SPOT", "P0", state.factory_spots);
  state.zcache.make_zones("FACTORY_SPOT", "P1", state.factory_spots);
  state.zcache.make_zones("ENEMY_ADJ", "P0", state.enemy_adj);
  state.zcache.make_zones("ENEMY_ADJ", "P1", state.enemy_adj);

  state.water_costs = {};
  for (auto& factory : state.game.factories[state.player]) {
    state.water_costs.emplace_back(factory.waterCost(obs));
  }
  state.free_factory_power = {};
  state.rubble_scores = {};
  state.lichen_scores = {};
  state.team_lichen = make_team_lichen(state);

  state.prev_value = {};
  state.next_value = {};
  auto& units = state.game.units[state.player];
  for (size_t i = 0; i < units.size(); i++) {
    state.prev_value.emplace_back(state.mapped_value[units[i].unit_id]);
    state.next_value.emplace_back(0.0);
  }
  state.mapped_value = {};
}

} // namespace anim
