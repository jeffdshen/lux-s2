#pragma once

#include <cstdint>
#include <string>
#include <variant>

#include <Eigen/Dense>

#include "anim/lux.h"
#include "anim/nav.h"
#include "anim/rubble.h"
#include "anim/state.h"
#include "anim/utils.h"

namespace anim {
const double DISCOUNT = 0.99;

enum class RubbleStage { PICKUP, MINE };

enum class MineStage { PICKUP, MINE, RETURN };

enum class ObjStatus { INVALID, RETRY, VALID };

std::string get_cost_name(const std::string& s, const UnitState& unit) {
  return s + std::to_string(unit.unit_type);
}

// TODO water value for factories, also adjust values depending on
// lichen space
constexpr std::array<double, MAX_RESOURCE_TYPE> DEFAULT_RESOURCE_VALUE = {
    5.0, 5.0, 5.0, 5.0, 0.0};

struct ObjEstimate {
  size_t unit_id;
  Loc end;
  size_t factory_id = 0;
  double move_power = 0;
  double move_turns = 0;
  double max_power = 0;
  double pickup_power = 0;
  double dig_turns = 0;
  double reward = 0;
  std::array<double, MAX_RESOURCE_TYPE> resources = {0};

  void add_unzipd(double cost) {
    auto [power, turns] = unzipd(cost);
    move_power += power;
    move_turns += turns;
  }

  bool init_unit(AgentState&, const NavState& nav) {
    auto& unit = nav.units[unit_id];
    max_power = unit.r_at(lux::Resource::POWER);
    resources = unit.resources;
    return true;
  }

  bool set_factory_id(AgentState& state, const NavState& nav) {
    auto& unit = nav.units[unit_id];
    auto P = get_cost_name("P", unit);
    auto& zone = state.zcache.get_zone("FACTORY_SPOT", P);
    factory_id = static_cast<size_t>(zone.zone(end.first, end.second));
    return true;
  }

  bool add_pickup(AgentState& state, const NavState& nav) {
    auto& unit = nav.units[unit_id];
    auto& unit_cfg = nav.cost_table.unit_cfgs[unit.unit_type];
    auto& loc = unit.loc;
    auto P = get_cost_name("P", unit);
    Loc f_loc = to_loc(state.game.factories[state.player][factory_id].pos);

    double cost = 0.0;
    auto& factory_spots = state.factory_spots[factory_id];
    cost += state.dcache.backward(factory_spots, P)(loc.first, loc.second);
    cost += state.dcache.backward(end, P)(f_loc.first, f_loc.second);

    max_power += nav.factories[factory_id].power;
    max_power =
        std::min(max_power, static_cast<double>(unit_cfg.BATTERY_CAPACITY));

    pickup_power = max_power - unit.r_at(lux::Resource::POWER);
    if (pickup_power <= 2 * unit_cfg.ACTION_QUEUE_POWER_COST) {
      return false;
    }
    add_unzipd(cost);
    move_turns += 1.0;
    return true;
  }

  bool add_goto(AgentState& state, const NavState& nav) {
    auto& unit = nav.units[unit_id];
    auto& loc = unit.loc;
    auto P = get_cost_name("P", unit);
    double cost = state.dcache.backward(end, P)(loc.first, loc.second);
    add_unzipd(cost);
    return true;
  }

  bool add_return(AgentState& state, const NavState& nav) {
    auto& unit = nav.units[unit_id];
    auto P = get_cost_name("P", unit);
    auto& zone = state.zcache.get_zone("FACTORY_SPOT", P);
    double cost = zone.dist(end.first, end.second);
    add_unzipd(cost);
    return true;
  }

  bool add_dropoff(AgentState&, const NavState&, lux::Resource resource_type) {
    size_t i = static_cast<size_t>(resource_type);
    move_turns += 1;
    reward += resources[i] * DEFAULT_RESOURCE_VALUE[i];
    return true;
  }

  bool add_dig_turns(AgentState& state, const NavState& nav) {
    auto& unit = nav.units[unit_id];
    auto& unit_cfg = nav.cost_table.unit_cfgs[unit.unit_type];
    auto buffer = unit_cfg.MOVE_COST * 5.0;
    auto dig_cost = unit_cfg.DIG_COST;

    dig_turns = std::floor((max_power - move_power - buffer) / dig_cost);
    if (dig_turns <= 0.0) {
      return false;
    }

    double rubble = state.game.board.rubble(end.first, end.second);
    double rubble_turns = std::ceil(rubble / unit_cfg.DIG_RUBBLE_REMOVED);
    bool is_ice = state.game.board.ice(end.first, end.second) == 1.0;
    bool is_ore = state.game.board.ore(end.first, end.second) == 1.0;
    if (!is_ice && !is_ore) {
      dig_turns = std::min(dig_turns, rubble_turns);
    }
    rubble_turns = std::min(dig_turns, rubble_turns);
    double rubble_cleared =
        std::min(rubble, rubble_turns * unit_cfg.DIG_RUBBLE_REMOVED);
    if (rubble > 0.0) {
      reward += rubble_cleared / rubble *
          state.rubble_scores.value(end.first, end.second);
    }

    if (is_ice) {
      size_t i = static_cast<size_t>(lux::Resource::ICE);
      resources[i] += (dig_turns - rubble_turns) * unit_cfg.DIG_RESOURCE_GAIN;
    }
    if (is_ore) {
      size_t i = static_cast<size_t>(lux::Resource::ORE);
      resources[i] += (dig_turns - rubble_turns) * unit_cfg.DIG_RESOURCE_GAIN;
    }

    return true;
  }

  double get_value(AgentState&, const NavState& nav) {
    auto& unit = nav.units[unit_id];
    auto& unit_cfg = nav.cost_table.unit_cfgs[unit.unit_type];
    auto dig_cost = unit_cfg.DIG_COST;
    double power_left = max_power - (dig_turns * dig_cost + move_power);

    // TODO add power cycle for daytime, different values for ice/ore
    double turns = move_turns + dig_turns;
    double future_value = power_left - max_power;
    future_value += reward;
    return future_value * std::pow(DISCOUNT, turns);
  }

  void reset() { *this = ObjEstimate{unit_id, end}; }
};

struct MineObj {
  size_t unit_id;
  int32_t step;
  Loc end;
  lux::Resource resource_type;
  MineStage stage;

  // cached values
  ObjEstimate obj{unit_id, end};
  double value = 0.0;

  bool estimate(AgentState& state, const NavState& nav) {
    obj.reset();
    obj.init_unit(state, nav);
    obj.set_factory_id(state, nav);

    if (!obj.add_return(state, nav)) {
      return false;
    }
    switch (stage) {
      case MineStage::PICKUP: {
        if (!obj.add_pickup(state, nav)) {
          return false;
        }
        break;
      }
      case MineStage::MINE: {
        if (!obj.add_goto(state, nav)) {
          return false;
        }
        break;
      }
      case MineStage::RETURN: {
        break;
      }
    }
    if (stage <= MineStage::MINE && !obj.add_dig_turns(state, nav)) {
      return false;
    }

    if (!obj.add_dropoff(state, nav, resource_type)) {
      return false;
    }
    value = obj.get_value(state, nav);
    if (value <= 0) {
      return false;
    }
    return true;
  }

  bool execute(AgentState& state, NavState& nav) {
    auto& unit = nav.units[unit_id];
    auto P = get_cost_name("P", unit);
    auto& cost = state.dcache.costs[P];
    switch (stage) {
      case MineStage::PICKUP: {
        {
          auto& factory_spots = state.factory_spots[obj.factory_id];
          auto& h = state.dcache.backward(factory_spots, P);
          auto [_, locs] = go_any(nav, unit_id, factory_spots, cost, h);
          size_t err = nav.apply_moves(unit_id, locs);
          if (err > 0) {
            LUX_INFO("err pickup: " << err);
          }
        }
        {
          auto action = lux::UnitAction::Pickup(
              lux::Resource::POWER, obj.pickup_power, 0, 1);
          nav.update(unit_id, action);
        }
        [[fallthrough]];
      }
      case MineStage::MINE: {
        auto& h = state.dcache.backward(end, P);
        auto [_, locs] = go_to(nav, unit_id, end, cost, h);
        size_t err = nav.apply_moves(unit_id, locs);
        if (err > 0) {
          LUX_LOG("err mine: " << err);
        }

        // TODO calculate real dig turns
        for (size_t i = 0; i < obj.dig_turns; i++) {
          auto action = lux::UnitAction::Dig(0, 1);
          if (!nav.update(unit_id, action)) {
            break;
          }
        }
        [[fallthrough]];
      }
      case MineStage::RETURN: {
        {
          auto& factory_spots = state.factory_spots[obj.factory_id];
          auto& h = state.dcache.backward(factory_spots, P);
          auto [_, locs] = go_any(nav, unit_id, factory_spots, cost, h);
          size_t err = nav.apply_moves(unit_id, locs);
          if (err > 0) {
            LUX_INFO("err return: " << err);
          }
        }
        {
          double resources = obj.resources[static_cast<size_t>(resource_type)];
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

    if (stage == MineStage::PICKUP &&
        obj.pickup_power > nav.factories[obj.factory_id].power) {
      return ObjStatus::RETRY;
    }

    return ObjStatus::VALID;
  }
};

struct RubbleObj {
  size_t unit_id;
  int32_t step;
  Loc end;
  RubbleStage stage;

  // cached values
  ObjEstimate obj{unit_id, end};
  double value = 0.0;

  bool estimate(AgentState& state, const NavState& nav) {
    obj.reset();
    obj.init_unit(state, nav);
    obj.set_factory_id(state, nav);

    switch (stage) {
      case RubbleStage::PICKUP: {
        if (!obj.add_pickup(state, nav)) {
          return false;
        }
        break;
      }
      case RubbleStage::MINE: {
        if (!obj.add_goto(state, nav)) {
          return false;
        }
        break;
      }
    }
    if (!obj.add_dig_turns(state, nav)) {
      return false;
    }

    value = obj.get_value(state, nav);
    if (value <= 0) {
      return false;
    }
    return true;
  }

  bool execute(AgentState& state, NavState& nav) {
    auto& unit = nav.units[unit_id];
    auto P = get_cost_name("P", unit);
    auto& cost = state.dcache.costs[P];
    switch (stage) {
      case RubbleStage::PICKUP: {
        {
          auto& factory_spots = state.factory_spots[obj.factory_id];
          auto& h = state.dcache.backward(factory_spots, P);
          auto [_, locs] = go_any(nav, unit_id, factory_spots, cost, h);
          size_t err = nav.apply_moves(unit_id, locs);
          if (err > 0) {
            LUX_INFO("err pickup: " << err);
          }
        }
        {
          auto action = lux::UnitAction::Pickup(
              lux::Resource::POWER, obj.pickup_power, 0, 1);
          nav.update(unit_id, action);
        }
        [[fallthrough]];
      }
      case RubbleStage::MINE: {
        auto& h = state.dcache.backward(end, P);
        auto [_, locs] = go_to(nav, unit_id, end, cost, h);
        size_t err = nav.apply_moves(unit_id, locs);
        if (err > 0) {
          LUX_LOG("err mine: " << err);
        }

        // TODO calculate real dig turns
        for (size_t i = 0; i < obj.dig_turns; i++) {
          auto action = lux::UnitAction::Dig(0, 1);
          if (!nav.update(unit_id, action)) {
            break;
          }
        }
      }
    }
    return true;
  }

  ObjStatus get_status(const AgentState&, const NavState& nav) const {
    if (step != nav.units[unit_id].step) {
      return ObjStatus::INVALID;
    }

    for (int32_t i = 0; i < nav.max_turns; i++) {
      if (nav.occupied.contains({i, end})) {
        return ObjStatus::INVALID;
      }
    }

    if (stage == RubbleStage::PICKUP &&
        obj.pickup_power > nav.factories[obj.factory_id].power) {
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

using ObjItem = std::variant<MineObj, RubbleObj, AvoidObj>;
struct ObjMatch {
  std::vector<ObjItem> items;
  std::priority_queue<std::pair<double, size_t>> q;

  void add(MineObj obj, AgentState& state, const NavState& nav) {
    bool success = obj.estimate(state, nav);
    if (!success) {
      return;
    }
    q.push({obj.value, items.size()});
    items.emplace_back(std::move(obj));
  }

  void add(RubbleObj obj, AgentState& state, const NavState& nav) {
    bool success = obj.estimate(state, nav);
    if (!success) {
      return;
    }
    q.push({obj.value, items.size()});
    items.emplace_back(std::move(obj));
  }
};

inline void add_mine_objs(
    ObjMatch& match, AgentState& state, const NavState& nav) {
  auto& board = state.game.board;
  auto ice = argwhere(board.ice, nonzero_f);
  auto ore = argwhere(board.ore, nonzero_f);
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
}

inline void add_rubble_objs(
    ObjMatch& match, AgentState& state, const NavState& nav) {
  auto& locs = state.rubble_scores.locs;
  auto& ice = state.game.board.ice;
  auto& ore = state.game.board.ore;
  auto& units = nav.units;
  for (size_t i = 0; i < std::min<size_t>(60, locs.size()); i++) {
    auto& loc = locs[i];
    if (ice(loc.first, loc.second) > 0 || ore(loc.first, loc.second) > 0) {
      continue;
    }

    for (auto& unit : units) {
      size_t unit_id = unit.unit_id;
      int32_t step = unit.step;
      {
        RubbleObj obj{unit_id, step, loc, RubbleStage::MINE};
        match.add(std::move(obj), state, nav);
      }
      {
        RubbleObj obj{unit_id, step, loc, RubbleStage::PICKUP};
        match.add(std::move(obj), state, nav);
      }
    }
  }
}

inline void make_mine(AgentState& state) {
  ObjMatch match;
  auto nav = NavState::from_agent_state(state);
  add_rubble_scores(state);
  add_mine_objs(match, state, nav);
  add_rubble_objs(match, state, nav);
  auto factories = state.game.factories[state.player];

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

  auto& units = nav.units;
  for (auto& unit : units) {
    AvoidObj o{unit.unit_id, unit.step, nav.max_turns};
    o.execute(state, nav);
  }
  set_action_queue(state, nav);
}
} // namespace anim
