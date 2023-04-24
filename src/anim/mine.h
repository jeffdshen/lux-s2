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
const double WATER_DISCOUNT = 0.996;

enum class RubbleStage { PICKUP, MINE };

enum class AttackStage { PICKUP, ATTACK };

enum class MineStage { EARLY_RETURN, PICKUP, MINE, RETURN };

enum class ObjStatus { INVALID, RETRY, VALID };

std::string get_cost_name(const std::string& s, const UnitState& unit) {
  return s + std::to_string(unit.unit_type);
}

constexpr std::array<double, MAX_RESOURCE_TYPE> DEFAULT_RESOURCE_VALUE = {
    5.0, 8, 5.0 * 4.0, 8 * 5.0, 0.0};
constexpr std::array<double, MAX_RESOURCE_TYPE> LOW_WATER_BONUS = {
    12.5, 0.0, 12.5 * 4.0, 0.0, 0.0};
constexpr double POWER_PENALTY = 0.998;
// TODO adjust based on lichen space
// constexpr std::array<double, MAX_RESOURCE_TYPE> LOW_SQUARE_PENALTY={};

struct ObjEstimate {
  size_t unit_id;
  Loc end;
  std::vector<Loc> ends{};
  size_t factory_id = 0;
  double move_power = 0;
  double move_turns = 0;
  double max_power = 0;
  double pickup_power = 0;
  double dig_turns = 0;
  double reward = 0;
  double value = 0;
  std::array<double, MAX_RESOURCE_TYPE> resources = {0};

  bool add(
      double power,
      double turns,
      const NavState& nav,
      bool out_of_order = false) {
    move_power += power;
    bool res = move_power + power <= max_power;
    if (!out_of_order) {
      max_power += nav.get_power_gain(unit_id, move_turns, move_turns + turns);
    }
    move_turns += turns;
    return res;
  }

  bool add_unzipd(double cost, const NavState& nav, bool out_of_order = false) {
    auto [power, turns] = unzipd(cost);
    return add(power, turns, nav, out_of_order);
  }

  bool init_unit(AgentState&, const NavState& nav) {
    auto& unit = nav.units[unit_id];
    max_power = unit.r_at(lux::Resource::POWER);
    resources = unit.resources;
    return true;
  }

  bool set_factory_id(
      AgentState& state, const NavState& nav, bool self = false) {
    auto& unit = nav.units[unit_id];
    auto P = get_cost_name("P", unit);
    auto& zone = state.zcache.get_zone("FACTORY_SPOT", P);
    if (self) {
      auto& loc = unit.loc;
      factory_id = static_cast<size_t>(zone.zone(loc.first, loc.second));
    } else {
      factory_id = static_cast<size_t>(zone.zone(end.first, end.second));
    }
    return true;
  }

  bool set_enemy_ends(AgentState& state, const NavState& nav) {
    auto& unit = nav.units[unit_id];
    auto P = get_cost_name("P", unit);
    auto& zone = state.zcache.get_zone("ENEMY_ADJ", P);
    size_t enemy_factory_id =
        static_cast<size_t>(zone.zone(end.first, end.second));
    ends = state.enemy_adj[enemy_factory_id];
    return true;
  }

  bool add_pickup(AgentState& state, const NavState& nav) {
    auto& unit = nav.units[unit_id];
    auto& unit_cfg = nav.get_unit_cfg(unit);
    auto& loc = unit.loc;
    auto P = get_cost_name("P", unit);
    Loc f_loc = to_loc(state.game.factories[state.player][factory_id].pos);

    {
      auto& factory_spots = state.factory_spots[factory_id];
      double cost =
          state.dcache.backward(factory_spots, P)(loc.first, loc.second);
      if (!add_unzipd(cost, nav)) {
        return false;
      }
    }

    {
      max_power += nav.factories[factory_id].power;
      max_power = std::min(
          max_power,
          static_cast<double>(unit_cfg.BATTERY_CAPACITY) + move_power);
    }
    pickup_power = max_power - unit.r_at(lux::Resource::POWER) - move_power;
    if (pickup_power <= 10 * unit_cfg.ACTION_QUEUE_POWER_COST) {
      return false;
    }

    add(0, 1, nav);
    {
      double cost = 0.0;
      if (ends.empty()) {
        cost += state.dcache.backward(end, P)(f_loc.first, f_loc.second);
      } else {
        cost += state.dcache.backward(ends, P)(f_loc.first, f_loc.second);
      }
      if (!add_unzipd(cost, nav)) {
        return false;
      }
    }

    return true;
  }

  bool add_goto(AgentState& state, const NavState& nav) {
    auto& unit = nav.units[unit_id];
    auto& loc = unit.loc;
    auto P = get_cost_name("P", unit);
    double cost = state.dcache.backward(end, P)(loc.first, loc.second);
    if (!add_unzipd(cost, nav)) {
      return false;
    }
    return true;
  }

  bool add_return(AgentState& state, const NavState& nav) {
    auto& unit = nav.units[unit_id];
    auto P = get_cost_name("P", unit);
    auto& zone = state.zcache.get_zone("FACTORY_SPOT", P);
    double cost = zone.dist(end.first, end.second);
    add_unzipd(cost, nav, true);
    return true;
  }

  bool add_dropoff(
      AgentState&, const NavState& nav, lux::Resource resource_type) {
    size_t i = static_cast<size_t>(resource_type);
    add(0, 1, nav);
    double r = resources[i] * DEFAULT_RESOURCE_VALUE[i];
    {
      double water = nav.factories[factory_id].water;
      r = std::max(
          r,
          resources[i] * LOW_WATER_BONUS[i] * std::pow(WATER_DISCOUNT, water));
    }
    if (resource_type == lux::Resource::ORE ||
        resource_type == lux::Resource::METAL) {
      double power = nav.factories[factory_id].power;
      r = r * (1 - std::pow(POWER_PENALTY, power));
    }

    reward += r;
    return true;
  }

  bool add_early_dropoff_pickup(
      AgentState& state, const NavState& nav, lux::Resource resource_type) {
    if (!add_pickup(state, nav)) {
      return false;
    }
    if (!add_dropoff(state, nav, resource_type)) {
      return false;
    }
    double turns = move_turns + dig_turns - 1.0;
    value += reward * std::pow(DISCOUNT, turns);
    reward = 0.0;
    return true;
  }

  bool add_dig_turns(AgentState& state, const NavState& nav) {
    auto& unit = nav.units[unit_id];
    auto& unit_cfg = nav.get_unit_cfg(unit);
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
          state.rubble_scores.reward(end.first, end.second);
    }

    if (is_ice || is_ore) {
      double max_resource = unit_cfg.CARGO_SPACE;
      for (size_t i = 0; i < 4; i++) {
        max_resource -= resources[i];
      }
      max_resource = std::max(max_resource, 0.0);

      auto resource_type = is_ice ? lux::Resource::ICE : lux::Resource::ORE;
      size_t i = static_cast<size_t>(resource_type);
      double gain = (dig_turns - rubble_turns) * unit_cfg.DIG_RESOURCE_GAIN;
      gain = std::min(max_resource, gain);
      resources[i] += gain;
    }

    return true;
  }

  double get_value(AgentState&, const NavState& nav) {
    auto& unit = nav.units[unit_id];
    auto& unit_cfg = nav.get_unit_cfg(unit);
    auto dig_cost = unit_cfg.DIG_COST;
    double power_left = max_power - (dig_turns * dig_cost + move_power);

    // TODO add power cycle for daytime, different values for ice/ore
    double turns = move_turns + dig_turns;
    double future_value = power_left - max_power;
    future_value += reward;
    return future_value * std::pow(DISCOUNT, turns) + value;
  }

  void reset() { *this = ObjEstimate{unit_id, end}; }
};

struct PrevObj {
  size_t unit_id;
  int32_t step;

  double value = 0.0;
  bool estimate(AgentState& state, const NavState&) {
    value = state.prev_value[unit_id];
    if (value <= 0) {
      return false;
    }
    return true;
  }

  bool execute(AgentState& state, NavState& nav) {
    auto& unit = state.my_unit(unit_id);
    std::deque<lux::UnitAction> actions{
        unit.action_queue.begin(), unit.action_queue.end()};

    if (actions.empty()) {
      return false;
    }
    for (size_t i = 0; i < 100 && !actions.empty(); i++) {
      auto action = pop_action(actions);
      if (!nav.update(unit_id, action)) {
        nav.revert(unit_id, step - nav.step);
        return false;
      }
    }

    return true;
  }

  ObjStatus get_status(const AgentState&, const NavState& nav) const {
    if (step != nav.units[unit_id].step) {
      return ObjStatus::INVALID;
    }

    return ObjStatus::VALID;
  }
};

// TODO
// struct AttackObj {
//   size_t unit_id;
//   int32_t step;
//   Loc end;

//   // cached values
//   ObjEstimate obj{unit_id, end};
//   double value = 0.0;
//   bool estimate(AgentState& state, const NavState& nav) {
//     obj.reset();
//     obj.init_unit(state, nav);
//     obj.set_factory_id(state, nav, true);

//     switch (stage) {
//       case AttackStage::PICKUP: {
//         if (!obj.add_pickup(state, nav)) {
//           return false;
//         }
//         break;
//       }
//       case AttackStage::ATTACK: {
//         if (!obj.add_goto(state, nav)) {
//           return false;
//         }
//         break;
//       }
//     }
//     // TODO
//     if (!obj.add_dig_turns(state, nav)) {
//       return false;
//     }

//     value = obj.get_value(state, nav);
//     if (value <= 0) {
//       return false;
//     }
//     return true;
//   }
// };

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
      case MineStage::EARLY_RETURN: {
        if (!obj.add_early_dropoff_pickup(state, nav, resource_type)) {
          return false;
        }
        break;
      }
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
          auto [_, locs] = go_any(nav, unit_id, factory_spots, cost, h, 1);
          size_t err = nav.apply_moves(unit_id, locs);
          if (err > 0) {
            LUX_INFO(
                "rev pick: " << err << ", " << state.my_unit(unit_id).unit_id);
            nav.revert(unit_id, step - nav.step);
            return false;
          }
        }
        {
          auto action = lux::UnitAction::Pickup(
              lux::Resource::POWER, obj.pickup_power, 0, 1);
          if (!nav.update(unit_id, action)) {
            LUX_INFO("rev pick: " << state.my_unit(unit_id).unit_id);
            nav.revert(unit_id, step - nav.step);
            return false;
          }
        }
        [[fallthrough]];
      }
      case MineStage::MINE: {
        {
          auto& h = state.dcache.backward(end, P);
          auto [_, locs] = go_to(nav, unit_id, end, cost, h);
          size_t err = nav.apply_moves(unit_id, locs);
          if (err > 0) {
            LUX_INFO(
                "rev mine: " << err << ", " << state.my_unit(unit_id).unit_id);
            nav.revert(unit_id, step - nav.step);
            return false;
          }
        }
        {
          auto action = lux::UnitAction::Dig(0, 1);
          size_t err = nav.repeat(unit_id, action, obj.dig_turns);
          if (err > 3) {
            LUX_INFO(
                "rev dig: " << err << ", " << state.my_unit(unit_id).unit_id);
            nav.revert(unit_id, step - nav.step);
            return false;
          }
        }
        [[fallthrough]];
      }
      case MineStage::EARLY_RETURN:
      case MineStage::RETURN: {
        {
          auto& factory_spots = state.factory_spots[obj.factory_id];
          auto& h = state.dcache.backward(factory_spots, P);
          auto [_, locs] = go_any(nav, unit_id, factory_spots, cost, h, 1);
          size_t err = nav.apply_moves(unit_id, locs);
          if (err > 0) {
            LUX_INFO(
                "rev return: " << err << ", "
                               << state.my_unit(unit_id).unit_id);
            nav.revert(unit_id, step - nav.step);
            return false;
          }
        }
        {
          double resources = obj.resources[static_cast<size_t>(resource_type)];
          auto action = lux::UnitAction::Transfer(
              lux::Direction::CENTER, resource_type, resources, 0, 1);
          if (!nav.update(unit_id, action)) {
            LUX_INFO(
                "rev tx: " << resources << ", "
                           << state.my_unit(unit_id).unit_id);
            nav.revert(unit_id, step - nav.step);
            return false;
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

    if (stage != MineStage::RETURN) {
      for (int32_t i = 0; i < nav.max_turns; i++) {
        if (nav.occupied.contains({i, end})) {
          return ObjStatus::INVALID;
        }
      }
    }

    if (stage <= MineStage::PICKUP &&
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
          auto [_, locs] = go_any(nav, unit_id, factory_spots, cost, h, 1);
          size_t err = nav.apply_moves(unit_id, locs);
          if (err > 0) {
            LUX_INFO(
                "rev pick: " << err << ", " << state.my_unit(unit_id).unit_id);
            nav.revert(unit_id, step - nav.step);
            return false;
          }
        }
        {
          auto action = lux::UnitAction::Pickup(
              lux::Resource::POWER, obj.pickup_power, 0, 1);
          if (!nav.update(unit_id, action)) {
            LUX_INFO("rev pick: " << state.my_unit(unit_id).unit_id);
            nav.revert(unit_id, step - nav.step);
            return false;
          }
        }
        [[fallthrough]];
      }
      case RubbleStage::MINE: {
        {
          auto& h = state.dcache.backward(end, P);
          auto [_, locs] = go_to(nav, unit_id, end, cost, h);
          size_t err = nav.apply_moves(unit_id, locs);
          if (err > 0) {
            LUX_INFO(
                "rev mine: " << err << ", " << state.my_unit(unit_id).unit_id);
            nav.revert(unit_id, step - nav.step);
            return false;
          }
        }
        {
          auto action = lux::UnitAction::Dig(0, 1);
          size_t err = nav.repeat(unit_id, action, obj.dig_turns);
          if (err > 3) {
            LUX_INFO(
                "rev dig: " << err << ", " << state.my_unit(unit_id).unit_id);
            nav.revert(unit_id, step - nav.step);
            return false;
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
      size_t err = nav.apply_moves(unit_id, locs);
      if (err > 0) {
        LUX_INFO(
            "err avoid: " << err << ", " << state.my_unit(unit_id).unit_id);
      }
    }
    return true;
  }

  ObjStatus get_status(const AgentState&, const NavState& nav) {
    if (step != nav.units[unit_id].step) {
      return ObjStatus::INVALID;
    }
    return ObjStatus::VALID;
  }
};

using ObjItem = std::variant<MineObj, RubbleObj, AvoidObj, PrevObj>;
struct ObjMatch {
  std::vector<ObjItem> items;
  std::priority_queue<std::pair<double, size_t>> q;

  template <class Obj>
  void add(Obj obj, AgentState& state, const NavState& nav) {
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
  for (size_t i = 0; i < std::min<size_t>(40, locs.size()); i++) {
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

inline void add_prev_objs(
    ObjMatch& match, AgentState& state, const NavState& nav) {
  auto& units = nav.units;
  for (auto& unit : units) {
    size_t unit_id = unit.unit_id;
    int32_t step = unit.step;
    {
      PrevObj obj{unit_id, step};
      match.add(std::move(obj), state, nav);
    }
  }
}

inline void add_avoid_objs(
    ObjMatch& match, AgentState& state, const NavState& nav) {
  auto& units = nav.units;
  for (auto& unit : units) {
    size_t unit_id = unit.unit_id;
    int32_t step = unit.step;
    {
      AvoidObj obj{unit_id, step, 10};
      match.add(std::move(obj), state, nav);
    }
  }
}

inline void make_mine(AgentState& state) {
  ObjMatch match;
  auto nav = NavState::from_agent_state(state);

  {
    add_rubble_scores(state);
    add_mine_objs(match, state, nav);
    add_rubble_objs(match, state, nav);
    // add_prev_objs(match, state, nav);
    add_avoid_objs(match, state, nav);
  }
  auto factories = state.game.factories[state.player];

  std::unordered_map<Loc, size_t, LocHash> loc_count{MAX_SIZE * MAX_SIZE};

  {
    size_t match_count = 0;
    size_t retry_count = 0;
    size_t execute_count = 0;
    size_t fail_count = 0;
    while (!match.q.empty()) {
      match_count++;
      auto [value, obj_id] = match.q.top();
      match.q.pop();
      auto& item = match.items[obj_id];
      ObjStatus status =
          std::visit([&](auto&& x) { return x.get_status(state, nav); }, item);
      if (status == ObjStatus::INVALID) {
        continue;
      }
      if (status == ObjStatus::RETRY) {
        retry_count++;
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
      std::visit(
          [&](auto&& x) {
            bool success = x.execute(state, nav);
            execute_count++;
            if (success) {
              state.next_value[x.unit_id] += x.value / DISCOUNT;
            } else {
              fail_count++;
            }
          },
          item);
    }
    LUX_INFO(
        "match counts: " << match_count << ", " << retry_count << ", "
                         << execute_count << ", " << fail_count);
  }

  // {
  //   auto& units = nav.units;
  //   for (auto& unit : units) {
  //     AvoidObj o{unit.unit_id, unit.step, 10};
  //     o.execute(state, nav);
  //   }
  // }
  set_action_queue(state, nav);
  state.add_mapped_value();
}
} // namespace anim
