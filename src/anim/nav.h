#pragma once

#include <cstdint>
#include <string>

#include <Eigen/Dense>

#include "anim/lux.h"
#include "anim/state.h"
#include "anim/utils.h"

namespace anim {
struct UnitState {
  int32_t step;
  size_t unit_id;
  size_t unit_type;

  Loc loc;
  std::array<double, MAX_RESOURCE_TYPE> resources;

  const double& r_at(const lux::Resource& r) const {
    return resources[static_cast<size_t>(r)];
  };

  double& r_at(const lux::Resource& r) {
    return resources[static_cast<size_t>(r)];
  };
};

struct FactoryState {
  size_t unit_id;
  Loc loc;
  double power;
};

struct CostTable {
  std::array<Eigen::ArrayXXd, MAX_UNIT_TYPE> move_power;
  std::array<lux::UnitConfig, MAX_UNIT_TYPE> unit_cfgs;

  static CostTable from(
      const lux::UnitConfigs& unit_cfgs, const Eigen::ArrayXXd& rubble) {
    CostTable table;
    table.unit_cfgs[0] = unit_cfgs.LIGHT;
    table.unit_cfgs[1] = unit_cfgs.HEAVY;
    for (size_t i = 0; i < 2; i++) {
      auto& unit = table.unit_cfgs[i];
      table.move_power[i] =
          (unit.MOVE_COST + rubble * unit.RUBBLE_MOVEMENT_COST).floor();
    }
    return table;
  }

  double get_power_cost(const UnitState& unit, const lux::UnitAction& action) {
    switch (action.type) {
      case lux::UnitAction::Type::MOVE: {
        auto loc = add(unit.loc, action.direction);
        return move_power[unit.unit_type](loc.first, loc.second);
      }
      case lux::UnitAction::Type::TRANSFER: {
        return action.amount * (action.resource == lux::Resource::POWER);
      }
      case lux::UnitAction::Type::PICKUP: {
        return -action.amount * (action.resource == lux::Resource::POWER);
      }
      case lux::UnitAction::Type::DIG: {
        return unit_cfgs[unit.unit_type].DIG_COST;
      }
      case lux::UnitAction::Type::SELF_DESTRUCT: {
        return unit_cfgs[unit.unit_type].SELF_DESTRUCT_COST;
      }
      case lux::UnitAction::Type::RECHARGE: {
        return 0.0;
      }
    }
  }

  double get_cargo_cost(const lux::UnitAction& action) {
    switch (action.type) {
      case lux::UnitAction::Type::MOVE:
        return 0.0;
      case lux::UnitAction::Type::TRANSFER:
        return action.amount * (action.resource != lux::Resource::POWER);
      case lux::UnitAction::Type::PICKUP:
        return -action.amount * (action.resource != lux::Resource::POWER);
      case lux::UnitAction::Type::DIG:
        return 0.0;
      case lux::UnitAction::Type::SELF_DESTRUCT:
        return 0.0;
      case lux::UnitAction::Type::RECHARGE:
        return 0.0;
    }
  }
};

using OccupiedMap = std::unordered_map<TimeLoc, UnitState, TimeLocHash>;
using DangerMap = std::unordered_map<TimeLoc, double, TimeLocHash>;
using TimeLocCosts = std::unordered_map<TimeLoc, double, TimeLocHash>;

struct NavState {
  int32_t step;
  int32_t cycle_length;
  int32_t day_length;
  CostTable cost_table;
  std::vector<UnitState> units;
  std::vector<FactoryState> factories;
  std::vector<std::vector<double>> power_cycles;
  std::vector<std::vector<double>> day_powers;
  int32_t max_turns = 100;
  int32_t search_limit = 10000;
  OccupiedMap occupied{MAX_SIZE * MAX_SIZE * 100};
  DangerMap danger{MAX_SIZE * MAX_SIZE * 100};
  std::vector<std::vector<lux::UnitAction>> actions;

  static NavState from_agent_state(const AgentState& state) {
    NavState nav;
    nav.step = state.game.real_env_steps;
    nav.cycle_length = state.env_cfg.CYCLE_LENGTH;
    nav.day_length = state.env_cfg.DAY_LENGTH;
    nav.cost_table =
        CostTable::from(state.game.config.ROBOTS, state.game.board.rubble);

    auto& my_units = state.game.units[state.player];
    nav.units.reserve(my_units.size());
    for (size_t i = 0; i < my_units.size(); i++) {
      auto& my_unit = my_units[i];
      auto& cargo = my_unit.cargo;
      UnitState unit;
      unit.step = nav.step;
      unit.unit_id = i;
      unit.unit_type = my_unit.unit_type == "LIGHT" ? 0 : 1;
      unit.loc = to_loc(my_unit.pos);
      double power = my_unit.power - my_unit.unitConfig.ACTION_QUEUE_POWER_COST;
      // Relys on order of types being ICE, ORE, WATER, METAL, POWER
      unit.resources = {
          static_cast<double>(cargo.ice),
          static_cast<double>(cargo.ore),
          static_cast<double>(cargo.water),
          static_cast<double>(cargo.metal),
          static_cast<double>(power)};
      nav.units.emplace_back(std::move(unit));
    }

    auto& my_factories = state.game.factories[state.player];
    nav.factories.reserve(my_factories.size());
    for (size_t i = 0; i < my_factories.size(); i++) {
      FactoryState factory;
      factory.unit_id = i;
      factory.loc = to_loc(my_factories[i].pos);
      factory.power = state.free_factory_power[i];
      nav.factories.emplace_back(factory);
    }

    for (size_t i = 0; i < MAX_UNIT_TYPE; i++) {
      auto& unit_cfg = nav.cost_table.unit_cfgs[i];
      std::vector<double> cycle;
      std::vector<double> day_power;
      double power = 0.0;
      for (int32_t step = nav.step; step < nav.step + nav.max_turns; step++) {
        double gain =
            unit_cfg.CHARGE * ((step % nav.cycle_length) < nav.day_length);
        cycle.emplace_back(power);
        day_power.emplace_back(gain);
        power += gain;
      }
      nav.power_cycles.emplace_back(std::move(cycle));
      nav.day_powers.emplace_back(std::move(day_power));
    }

    nav.actions.resize(nav.units.size());
    // TODO danger map
    return nav;
  }

  size_t get_factory_id(const Loc& loc) {
    for (size_t i = 0; i < factories.size(); i++) {
      auto& f_loc = factories[i].loc;
      if (abs(loc.first - f_loc.first) > 1) {
        continue;
      }
      if (abs(loc.second - f_loc.second) > 1) {
        continue;
      }
      return i;
    }
    return factories.size();
  }

  bool update(size_t unit_id, const lux::UnitAction& action) {
    // TODO this should do as much can be simulated (one-sided), e.g.:
    // - range/bounds checking (return false if bad)
    // - other unit should get bonus for transfer
    // - subtract from factory for pickup
    // - dig should increase resource, update rubble
    // - self destruct
    // - be conservative if there are unknowns (e.g. other units, etc.)
    auto& unit = units[unit_id];
    if (unit.step >= step + max_turns) {
      return false;
    }

    auto next_step = unit.step + 1;
    auto next_loc =
        action.isMoveAction() ? add(unit.loc, action.direction) : unit.loc;
    TimeLoc ts{next_step - step, next_loc};
    if (occupied.contains(ts)) {
      return false;
    }

    // TODO other factory pickups? transfers?
    auto factory_id = get_factory_id(unit.loc);
    double power_cost = cost_table.get_power_cost(unit, action);
    if (action.isPickupAction()) {
      if (-power_cost + factories[factory_id].power < 0) {
        return false;
      }

      factories[factory_id].power += power_cost;
    }

    unit.r_at(lux::Resource::POWER) -= power_cost;
    // would need to do power check here, before power gain
    unit.r_at(lux::Resource::POWER) +=
        day_powers[unit.unit_type][unit.step - step];
    unit.step = next_step;
    unit.loc = next_loc;
    unit.r_at(action.resource) -= cost_table.get_cargo_cost(action);
    occupied[ts] = unit;
    actions[unit_id].emplace_back(action);
    return true;
  }

  // tries to move along path, returns 0 if successful
  // otherwise returns the error location
  size_t apply_moves(size_t unit_id, const std::vector<Loc>& locs) {
    if (locs.empty()) {
      return -1;
    }

    for (size_t i = 1; i < locs.size(); i++) {
      auto dir = to_dir(sub(locs[i], locs[i - 1]));
      auto action = lux::UnitAction::Move(dir, 0, 1);
      if (!update(unit_id, action)) {
        return i;
      }
    }
    return 0;
  }
};

inline bool base_action_equals(
    const lux::UnitAction& a, const lux::UnitAction& b) {
  return a.type == b.type && a.direction == b.direction &&
      a.resource == b.resource && a.amount == b.amount;
}

inline std::vector<lux::UnitAction> compress_actions(
    const std::vector<lux::UnitAction>& actions) {
  std::vector<lux::UnitAction> next_actions;
  for (auto& action : actions) {
    if (next_actions.empty()) {
      next_actions.emplace_back(action);
      continue;
    }

    auto& next_action = next_actions.back();
    if (base_action_equals(next_action, action)) {
      next_action.repeat += action.repeat;
      next_action.n += action.n;
    } else {
      next_actions.emplace_back(action);
      continue;
    }
  }
  return next_actions;
}

inline void set_action_queue(AgentState& state, const NavState& nav) {
  auto& state_units = state.game.units[state.player];
  size_t queue_size = state.env_cfg.UNIT_ACTION_QUEUE_SIZE;
  for (size_t unit_id = 0; unit_id < state_units.size(); unit_id++) {
    auto& action_queue = state_units[unit_id].action_queue;
    auto nav_actions = compress_actions(nav.actions[unit_id]);
    nav_actions.resize(std::min(queue_size, nav_actions.size()));

    if (action_queue.empty()) {
      // don't need to add MOVE CENTER, since action_queue is already empty
      if (!nav_actions.empty()) {
        state.actions.units[unit_id] = nav_actions;
      }
      continue;
    }

    if (nav_actions.empty()) {
      nav_actions.emplace_back(
          lux::UnitAction::Move(lux::Direction::CENTER, 0, 1));
    }

    if (base_action_equals(action_queue[0], nav_actions[0])) {
      // defer changing action queue until we need to
      continue;
    }

    state.actions.units[unit_id] = nav_actions;
  }
}

inline std::vector<Loc> trace_path(
    const std::unordered_map<TimeLoc, TimeLoc, TimeLocHash>& came_from,
    const TimeLoc& end) {
  std::vector<Loc> total_path{end.second};
  TimeLoc current = end;
  while (true) {
    if (auto it = came_from.find(current); it != came_from.end()) {
      current = it->second;
      total_path.emplace_back(current.second);
    } else {
      break;
    }
  }
  std::reverse(std::begin(total_path), std::end(total_path));
  return total_path;
}

inline std::vector<std::pair<double, TimeLoc>> get_next_gs(
    const TimeLoc& tu,
    const TimeLocCosts& g,
    double min_power,
    double init_power,
    const std::vector<double>& power_cycle,
    const Eigen::ArrayXXd& cost,
    const OccupiedMap& occupied,
    const DangerMap& danger) {
  std::vector<std::pair<double, TimeLoc>> neighbors;
  auto& [t, u] = tu;
  for (auto& n : NEIGHBORS) {
    auto v = add(u, n);
    TimeLoc tv{t + 1, v};
    if (!in_bounds(v, shape(cost))) {
      continue;
    }
    if (occupied.contains(tv)) {
      continue;
    }
    double next_g = g.at(tu) + cost(v.first, v.second);
    // next_g can have a fractional part (e.g. # of steps / 2^N),
    // so check a >= b + 1 instead of a > b
    double max_min_power = std::max(get_default(danger, tv, 0), min_power);
    if (next_g >= init_power + power_cycle[t + 1] - max_min_power + 1) {
      continue;
    }
    neighbors.emplace_back(next_g, tv);
  }

  for (auto& n : {Loc{0, 0}}) {
    auto v = add(u, n);
    TimeLoc tv{t + 1, v};
    // skip in_bounds check, since u should be in bounds already.
    if (occupied.contains(tv)) {
      continue;
    }
    double next_g = g.at(tu);
    // moving robots always win
    if (danger.contains(tv)) {
      continue;
    }
    // no need for power check, since we are not moving
    neighbors.emplace_back(next_g, tv);
  }
  return neighbors;
}

template <class E, class G>
std::pair<double, std::vector<Loc>> a_star(
    const TimeLoc& start,
    E end_func,
    int32_t max_turns,
    int32_t search_limit,
    const Eigen::ArrayXXd& h,
    G next_g_func) {
  std::unordered_map<TimeLoc, TimeLoc, TimeLocHash> came_from(
      MAX_SIZE * MAX_SIZE * max_turns);
  std::unordered_set<TimeLoc, TimeLocHash> seen(
      MAX_SIZE * MAX_SIZE * max_turns);
  TimeLocCosts g(MAX_SIZE * MAX_SIZE * max_turns);
  TimeLocCosts f(MAX_SIZE * MAX_SIZE * max_turns);

  double start_h = h(start.second.first, start.second.second);
  g[start] = 0.0;
  f[start] = start_h;

  using Score = std::pair<double, TimeLoc>;
  std::priority_queue<Score, std::vector<Score>, std::greater<Score>> q;
  q.emplace(start_h, start);
  int32_t expanded = 0;
  while (!q.empty()) {
    auto [d, tu] = q.top();
    q.pop();
    auto& [t, u] = tu;
    if (end_func(tu)) {
      return {g[tu], trace_path(came_from, tu)};
    }
    if (seen.contains(tu)) {
      continue;
    }
    seen.emplace(tu);
    if (t + 1 >= max_turns) {
      continue;
    }
    if (expanded >= search_limit) {
      return {INF, {}};
    }
    expanded++;
    auto next_gs = next_g_func(tu, g);
    for (auto& [next_g, tv] : next_gs) {
      if (next_g >= get_default(g, tv, INF)) {
        continue;
      }

      came_from.emplace(tv, tu);
      g[tv] = next_g;

      auto& [t, v] = tv;
      double next_f = next_g + h(v.first, v.second);
      f[tv] = next_f;
      q.emplace(next_f, tv);
    }
  }
  return {INF, {}};
}

inline std::pair<double, std::vector<Loc>> avoid(
    const NavState& state,
    size_t unit_id,
    const int32_t turns,
    const Eigen::ArrayXXd& cost) {
  auto& unit = state.units[unit_id];
  TimeLoc start = {unit.step - state.step, unit.loc};
  double min_power = unit.unit_type * 1000;
  double power = min_power + unit.r_at(lux::Resource::POWER);
  auto& power_cycle = state.power_cycles[unit.unit_type];

  Eigen::ArrayXXd h =
      Eigen::ArrayXXd::Constant(cost.rows(), cost.cols(), turns * EPS);

  auto end_func = [&](const TimeLoc& tu) {
    return tu.first >= (turns + unit.step - state.step);
  };

  auto next_g_func = [&](const TimeLoc& tu, const TimeLocCosts& g) {
    return get_next_gs(
        tu,
        g,
        min_power,
        power,
        power_cycle,
        cost,
        state.occupied,
        state.danger);
  };
  return a_star(
      start, end_func, state.max_turns, state.search_limit, h, next_g_func);
}

inline std::pair<double, std::vector<Loc>> go_any(
    const NavState& state,
    size_t unit_id,
    const std::vector<Loc>& ends,
    const Eigen::ArrayXXd& cost,
    const Eigen::ArrayXXd& h) {
  auto& unit = state.units[unit_id];
  TimeLoc start = {unit.step - state.step, unit.loc};
  double min_power = unit.unit_type * 1000;
  double power = min_power + unit.r_at(lux::Resource::POWER);
  auto& power_cycle = state.power_cycles[unit.unit_type];

  std::unordered_set<Loc, LocHash> ends_set(MAX_SIZE * MAX_SIZE);
  for (auto& end : ends) {
    ends_set.emplace(end);
  }
  auto end_func = [&](const TimeLoc& tu) {
    return ends_set.contains(tu.second);
  };

  auto next_g_func = [&](const TimeLoc& tu, const TimeLocCosts& g) {
    return get_next_gs(
        tu,
        g,
        min_power,
        power,
        power_cycle,
        cost,
        state.occupied,
        state.danger);
  };
  return a_star(
      start, end_func, state.max_turns, state.search_limit, h, next_g_func);
}

inline std::pair<double, std::vector<Loc>> go_to(
    const NavState& state,
    size_t unit_id,
    const Loc& end,
    const Eigen::ArrayXXd& cost,
    const Eigen::ArrayXXd& h) {
  auto& unit = state.units[unit_id];
  TimeLoc start = {unit.step - state.step, unit.loc};
  double min_power = unit.unit_type * 1000;
  double power = min_power + unit.r_at(lux::Resource::POWER);
  auto& power_cycle = state.power_cycles[unit.unit_type];

  auto end_func = [&](const TimeLoc& tu) { return tu.second == end; };

  auto next_g_func = [&](const TimeLoc& tu, const TimeLocCosts& g) {
    return get_next_gs(
        tu,
        g,
        min_power,
        power,
        power_cycle,
        cost,
        state.occupied,
        state.danger);
  };
  return a_star(
      start, end_func, state.max_turns, state.search_limit, h, next_g_func);
}
} // namespace anim
