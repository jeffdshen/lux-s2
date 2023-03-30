#pragma once

#include <cstdint>
#include <string>
#include <variant>

#include <Eigen/Dense>

#include "anim/lux.h"
#include "anim/nav.h"
#include "anim/state.h"
#include "anim/utils.h"

namespace anim {
const double DISCOUNT = 0.99;

enum class MineStage { PICKUP, MINE, RETURN };

enum class ObjStatus { INVALID, RETRY, VALID };

struct MineObj {
  size_t unit_id;
  int32_t step;
  Loc end;
  lux::Resource resource_type;
  MineStage stage;

  // cached values
  size_t factory_id = 0;
  double move_power = 0;
  double move_turns = 0;
  double max_power = 0;
  double pickup_power = 0;
  double dig_turns = 0;
  double resources = 0;
  double turns = 0;
  double value = 0;

  bool estimate_dig_turns(const lux::UnitConfig& unit_cfg) {
    if (stage > MineStage::MINE) {
      return true;
    }

    auto buffer = unit_cfg.MOVE_COST * 5.0;
    auto dig_cost = unit_cfg.DIG_COST;

    dig_turns = std::floor((max_power - move_power - buffer) / dig_cost);
    if (dig_turns <= 0.0) {
      return false;
    }
    return true;
  }

  bool estimate_value(const UnitState& unit, const lux::UnitConfig& unit_cfg) {
    auto dig_gain = unit_cfg.DIG_RESOURCE_GAIN;
    auto dig_cost = unit_cfg.DIG_COST;

    resources = unit.r_at(resource_type) + dig_turns * dig_gain;
    if (resources <= 0.0) {
      return false;
    }

    double power_left = max_power - (dig_turns * dig_cost + move_power);

    // TODO add power cycle for daytime, different values for ice/ore
    turns = move_turns + dig_turns;
    double future_value = resources * 5.0 + power_left - max_power;
    value = future_value * std::pow(DISCOUNT, turns);
    return true;
  }

  bool estimate_turns(AgentState& state, const NavState& nav) {
    auto& unit = nav.units[unit_id];
    auto& loc = unit.loc;
    auto& unit_cfg = nav.cost_table.unit_cfgs[unit.unit_type];
    auto cost_name = "P" + std::to_string(unit.unit_type);
    auto& zone = state.zcache.get_zone("FACTORY_SPOT", cost_name);
    double cost = 0.0;
    factory_id = static_cast<size_t>(zone.zone(end.first, end.second));

    // The RETURN stage.
    cost += zone.dist(end.first, end.second);
    max_power = unit.r_at(lux::Resource::POWER);
    double add_turns = 1.0;

    switch (stage) {
      case MineStage::PICKUP: {
        Loc f_loc = to_loc(state.game.factories[state.player][factory_id].pos);
        cost += state.dcache.backward(
            state.factory_spots[factory_id], cost_name)(loc.first, loc.second);
        cost +=
            state.dcache.backward(end, cost_name)(f_loc.first, f_loc.second);
        add_turns += 1.0;

        max_power += nav.factories[factory_id].power;
        max_power =
            std::min(max_power, static_cast<double>(unit_cfg.BATTERY_CAPACITY));

        pickup_power = max_power - unit.r_at(lux::Resource::POWER);
        if (pickup_power <= 0.0) {
          return false;
        }
        break;
      }
      case MineStage::MINE: {
        cost += state.dcache.backward(end, cost_name)(loc.first, loc.second);
        break;
      }
      case MineStage::RETURN: {
        break;
      }
    }

    unzipd(cost, move_power, move_turns);
    move_turns += add_turns;

    return true;
  }

  bool estimate(AgentState& state, const NavState& nav) {
    if (!estimate_turns(state, nav)) {
      return false;
    }

    auto& unit = nav.units[unit_id];
    auto& unit_cfg = nav.cost_table.unit_cfgs[unit.unit_type];

    if (!estimate_dig_turns(unit_cfg)) {
      return false;
    }

    if (!estimate_value(unit, unit_cfg)) {
      return false;
    }

    return true;
  }

  bool execute(AgentState& state, NavState& nav) {
    auto& unit = nav.units[unit_id];
    auto cost_name = "P" + std::to_string(unit.unit_type);
    auto& cost = state.dcache.costs[cost_name];
    switch (stage) {
      case MineStage::PICKUP: {
        {
          auto& factory_spots = state.factory_spots[factory_id];
          auto& h = state.dcache.backward(factory_spots, cost_name);
          auto [_, locs] = go_any(nav, unit_id, factory_spots, cost, h);
          size_t err = nav.apply_moves(unit_id, locs);
          if (err > 0) {
            LUX_INFO("err pickup: " << err);
          }
        }
        {
          auto action =
              lux::UnitAction::Pickup(lux::Resource::POWER, pickup_power, 0, 1);
          nav.update(unit_id, action);
        }
        [[fallthrough]];
      }
      case MineStage::MINE: {
        auto& h = state.dcache.backward(end, cost_name);
        auto [_, locs] = go_to(nav, unit_id, end, cost, h);
        size_t err = nav.apply_moves(unit_id, locs);
        if (err > 0) {
          LUX_LOG("err mine: " << err);
        }

        // TODO calculate real dig turns
        for (size_t i = 0; i < dig_turns; i++) {
          auto action = lux::UnitAction::Dig(0, 1);
          if (!nav.update(unit_id, action)) {
            break;
          }
        }
        [[fallthrough]];
      }
      case MineStage::RETURN: {
        {
          auto& factory_spots = state.factory_spots[factory_id];
          auto& h = state.dcache.backward(factory_spots, cost_name);
          auto [_, locs] = go_any(nav, unit_id, factory_spots, cost, h);
          size_t err = nav.apply_moves(unit_id, locs);
          if (err > 0) {
            LUX_INFO("err return: " << err);
          }
        }
        {
          auto action = lux::UnitAction::Transfer(
              lux::Direction::CENTER, resource_type, resources, 0, 1);
          nav.update(unit_id, action);
        }
      }
    }
    return true;
  }

  ObjStatus get_status(const AgentState&, const NavState& nav) const {
    if (step != nav.units[unit_id].step) {
      return ObjStatus::INVALID;
    }

    if (stage != MineStage::RETURN) {
      for (int32_t i = 0; i < nav.max_turns; i++) {
        if (nav.occupied.contains({i, end})) {
          return ObjStatus::INVALID;
        }
      }
    }

    if (pickup_power > nav.factories[factory_id].power) {
      // LUX_INFO(
      //     "status " << unit_id << ", " << pickup_power << ", "
      //               << nav.factories[factory_id].power);
      return ObjStatus::RETRY;
    }

    return ObjStatus::VALID;
  }
};

struct AvoidObj {
  size_t unit_id;
  int32_t step;
  int32_t turns;
  double value = 0;

  bool estimate(AgentState&, const NavState&) { return true; }

  bool execute(AgentState& state, NavState& nav) {
    auto& unit = nav.units[unit_id];
    auto cost_name = "P" + std::to_string(unit.unit_type);
    auto& cost = state.dcache.costs[cost_name];
    {
      int32_t max_turns =
          std::min(nav.max_turns - 1, turns + unit.step - nav.step);
      auto [_, locs] = avoid(nav, unit_id, max_turns, cost);
      nav.apply_moves(unit_id, locs);
    }
    return true;
  }

  ObjStatus get_status(const AgentState&, const NavState&) {
    return ObjStatus::VALID;
  }
};

using ObjItem = std::variant<MineObj, AvoidObj>;
struct ObjMatch {
  std::vector<ObjItem> items;
  std::priority_queue<std::pair<double, size_t>> q;

  void add(MineObj mine, AgentState& state, const NavState& nav) {
    bool success = mine.estimate(state, nav);
    if (!success) {
      return;
    }
    q.push({mine.value, items.size()});
    items.emplace_back(std::move(mine));
  }
};

inline void make_mine(AgentState& state) {
  auto& game = state.game;
  auto ice = argwhere(game.board.ice, nonzero_f);
  auto ore = argwhere(game.board.ore, nonzero_f);
  auto factories = state.game.factories[state.player];
  ObjMatch match;
  auto nav = NavState::from_agent_state(state);
  auto& units = nav.units;

  for (auto& unit : units) {
    size_t unit_id = unit.unit_id;
    int32_t step = unit.step;
    for (auto& loc : ice) {
      {
        MineObj mine{unit_id, step, loc, lux::Resource::ICE, MineStage::MINE};
        match.add(mine, state, nav);
      }
      {
        MineObj mine{unit_id, step, loc, lux::Resource::ICE, MineStage::PICKUP};
        match.add(mine, state, nav);
      }
    }

    for (auto& loc : ore) {
      {
        MineObj mine{unit_id, step, loc, lux::Resource::ORE, MineStage::MINE};
        match.add(mine, state, nav);
      }
      {
        MineObj mine{unit_id, step, loc, lux::Resource::ORE, MineStage::PICKUP};
        match.add(mine, state, nav);
      }
    }
    {
      auto loc = unit.loc;
      {
        MineObj mine{unit_id, step, loc, lux::Resource::ICE, MineStage::RETURN};
        match.add(mine, state, nav);
      }
      {
        MineObj mine{unit_id, step, loc, lux::Resource::ORE, MineStage::RETURN};
        match.add(mine, state, nav);
      }
    }
  }

  std::unordered_map<Loc, size_t, LocHash> loc_count{MAX_SIZE * MAX_SIZE};
  while (!match.q.empty()) {
    auto [value, obj_id] = match.q.top();
    match.q.pop();
    auto& item = match.items[obj_id];
    ObjStatus status =
        std::visit([&](auto&& x) { return x.get_status(state, nav); }, item);
    if (status == ObjStatus::INVALID) {
      continue;
    }
    if (status == ObjStatus::RETRY) {
      std::visit(
          [&, obj_id = obj_id](auto&& x) {
            bool success = x.estimate(state, nav);
            if (success) {
              match.q.push({x.value, obj_id});
            }
          },
          item);
      continue;
    }
    std::visit([&](auto&& x) { return x.execute(state, nav); }, item);
  }

  for (auto& unit : units) {
    AvoidObj o{unit.unit_id, unit.step, nav.max_turns};
    o.execute(state, nav);
  }
  set_action_queue(state, nav);
}
} // namespace anim
