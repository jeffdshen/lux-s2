
#include <cstdint>
#include <string>

#include <Eigen/Dense>

#include "anim/lux.h"
#include "anim/nav.h"
#include "anim/state.h"
#include "anim/utils.h"

namespace anim {
const double DISCOUNT = 0.99;

struct MineObj {
  size_t unit_id;
  int32_t step;
  Loc end;
  lux::Resource resource_type;
  bool go_home;

  // cached values
  size_t factory_id = 0;
  double move_power = 0;
  double move_turns = 0;
  double dig_turns = 0;
  double resources = 0;
  double future_value = 0;
  double value = 0;
  double turns = 0;

  void estimate(AgentState& state, const NavState& nav) {
    auto& unit = nav.units[unit_id];
    auto& unit_cfg = nav.cost_table.unit_cfgs[unit.unit_type];
    auto cost_name = "P" + std::to_string(unit.unit_type);

    double cost = 0.0;

    // dcache to mine location (end)
    if (!go_home) {
      auto& loc = unit.loc;
      cost += state.dcache.backward(end, cost_name)(loc.first, loc.second);
    }

    // dcache to home base
    auto& zone = state.zcache.get_zone("FACTORY_SPOT", cost_name);
    factory_id = static_cast<size_t>(zone.zone(end.first, end.second));
    cost += zone.dist(end.first, end.second);

    // pickup/dropoff
    {
      auto [power, turns] = unzipd(cost);
      move_power = power;
      move_turns = turns;
    }

    double max_power = unit.r_at(lux::Resource::POWER);
    auto buffer = unit_cfg.MOVE_COST * 5.0;
    auto dig_cost = unit_cfg.DIG_COST;
    auto dig_gain = unit_cfg.DIG_RESOURCE_GAIN;

    if (max_power - move_power - buffer < 0.0) {
      future_value = -1.0;
      value = -1.0;
      return;
    }
    if (!go_home) {
      dig_turns = std::floor((max_power - move_power - buffer) / dig_cost);
    }

    resources = unit.r_at(resource_type) + dig_turns * dig_gain;
    double power_left = max_power - (dig_turns * dig_cost + move_power);

    if (resources <= 0.0) {
      future_value = -1.0;
      value = -1.0;
      return;
    }

    // TODO add power cycle for daytime, different values for ice/ore
    turns = move_turns + dig_turns + 1;
    future_value = resources * 5.0 + power_left;
    value = future_value * std::pow(DISCOUNT, turns);
  }

  bool execute(AgentState& state, NavState& nav) {
    auto& unit = nav.units[unit_id];
    auto cost_name = "P" + std::to_string(unit.unit_type);
    auto& cost = state.dcache.costs[cost_name];
    if (!go_home) {
      auto& h = state.dcache.backward(end, cost_name);
      auto [_, locs] = go_to(nav, unit_id, end, cost, h);
      nav.apply_moves(unit_id, locs);

      // TODO calculate real dig turns
      for (size_t i = 0; i < dig_turns; i++) {
        auto action = lux::UnitAction::Dig(0, 1);
        if (!nav.update(unit_id, action)) {
          break;
        }
      }
    }
    {
      auto factory_spot = state.factory_spots[factory_id][0];
      auto& h = state.dcache.backward(factory_spot, cost_name);
      auto [_, locs] = go_to(nav, unit_id, factory_spot, cost, h);
      nav.apply_moves(unit_id, locs);
    }
    {
      auto action = lux::UnitAction::Transfer(
          lux::Direction::CENTER, resource_type, resources, 0, 1);
      nav.update(unit_id, action);
    }
    return true;
  }
};

struct ObjState {
  std::vector<MineObj> mine;
};

// TODO how to calcuate mine score?
// (reward - power cost) / turns?
// if you have finished mining, how to know to turn back?
// roughly forward simulate up to point with definitive reward
// this gives you a location + reward + ending power
// and turn #
// you can calculate a value function of all end states
// assume value function is separable into components per unit

inline void make_mine(AgentState& state) {
  auto& game = state.game;
  auto ice = argwhere(game.board.ice, nonzero_f);
  auto ore = argwhere(game.board.ore, nonzero_f);
  auto factories = state.game.factories[state.player];
  ObjState obj;
  auto nav = NavState::from_agent_state(state);
  auto& units = nav.units;
  std::priority_queue<std::pair<double, size_t>> q;

  for (auto& unit : units) {
    for (auto& loc : ice) {
      MineObj mine{unit.unit_id, unit.step, loc, lux::Resource::ICE, false};
      mine.estimate(state, nav);
      if (mine.value <= 0) {
        continue;
      }
      q.push({mine.value, obj.mine.size()});
      obj.mine.emplace_back(std::move(mine));
    }

    for (auto& loc : ore) {
      MineObj mine{unit.unit_id, unit.step, loc, lux::Resource::ORE, false};
      mine.estimate(state, nav);
      if (mine.value <= 0) {
        continue;
      }
      q.push({mine.value, obj.mine.size()});
      obj.mine.emplace_back(std::move(mine));
    }
    {
      auto loc = unit.loc;
      {
        MineObj mine{unit.unit_id, unit.step, loc, lux::Resource::ICE, true};
        mine.estimate(state, nav);
        if (mine.value <= 0) {
          continue;
        }
        q.push({mine.value, obj.mine.size()});
        obj.mine.emplace_back(std::move(mine));
      }
      {
        MineObj mine{unit.unit_id, unit.step, loc, lux::Resource::ORE, true};
        mine.estimate(state, nav);
        if (mine.value <= 0) {
          continue;
        }
        q.push({mine.value, obj.mine.size()});
        obj.mine.emplace_back(std::move(mine));
      }
    }
  }

  std::unordered_set<size_t> unit_seen;
  std::unordered_map<Loc, size_t, LocHash> loc_count{MAX_SIZE * MAX_SIZE};
  while (!q.empty()) {
    auto [value, obj_id] = q.top();
    q.pop();
    auto& mine = obj.mine[obj_id];
    if (unit_seen.contains(mine.unit_id)) {
      continue;
    }
    if (loc_count[mine.end] >= 2) {
      continue;
    }
    unit_seen.emplace(mine.unit_id);
    loc_count[mine.end]++;

    mine.execute(state, nav);
  }
  // TODO go back home behavior when done mining
  // TODO run avoid
  set_action_queue(state, nav);
}
} // namespace anim
