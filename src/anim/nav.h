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
      case lux::UnitAction::Type::MOVE:
        auto loc = add(unit.loc, action.direction);
        return move_power[unit.unit_type](loc.first, loc.second);
      case lux::UnitAction::Type::TRANSFER:
        return action.amount * (action.resource == lux::Resource::POWER);
        return move_power[unit.unit_type](loc.first, loc.second);
      case lux::UnitAction::Type::PICKUP:
        return -action.amount * (action.resource == lux::Resource::POWER);
      case lux::UnitAction::Type::DIG:
        return unit_cfgs[unit.unit_type].DIG_COST;
      case lux::UnitAction::Type::SELF_DESTRUCT:
        return unit_cfgs[unit.unit_type].SELF_DESTRUCT_COST;
      case lux::UnitAction::Type::RECHARGE:
        return 0.0;
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

std::vector<double> get_power_cycle(
    double power,
    double power_per_day,
    int32_t start_step,
    int32_t max_step,
    int32_t cycle_length,
    int32_t day_length) {
  std::vector<double> cycle;
  for (int32_t step = start_step; step < max_step; step++) {
    cycle.emplace_back(power);
    power += power_per_day * ((step % cycle_length) < day_length);
  }
  return cycle;
}

using OccupiedMap = std::unordered_map<TimeLoc, UnitState, TimeLocHash>;
using DangerMap = std::unordered_map<TimeLoc, double, TimeLocHash>;
using TimeLocCosts = std::unordered_map<TimeLoc, double, TimeLocHash>;

struct NavState {
  int32_t step;
  int32_t cycle_length;
  int32_t day_length;
  CostTable cost_table;
  std::vector<UnitState> units;
  int32_t max_turns = 100;
  int32_t search_limit = 10000;
  OccupiedMap occupied{MAX_SIZE * MAX_SIZE * max_turns};
  DangerMap danger{MAX_SIZE * MAX_SIZE * max_turns};
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
          static_cast<double>(my_unit.power)};
      nav.units.emplace_back(std::move(unit));
    }
    return nav;
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
    double power_cost = cost_table.get_power_cost(unit, action);
    unit.r_at(lux::Resource::POWER) -= power_cost;
    unit.step++;
    if (action.isMoveAction()) {
      unit.loc = add(unit.loc, action.direction);
    }
    unit.r_at(action.resource) -= cost_table.get_cargo_cost(action);
    TimeLoc ts = {unit.step - step, unit.loc};
    occupied[ts] = unit;
    actions[unit_id].emplace_back(action);
    return true;
  }
};

std::vector<Loc> trace_path(
    const std::unordered_map<TimeLoc, TimeLoc, TimeLocHash>& came_from,
    const TimeLoc& end) {
  std::vector<Loc> total_path{end.second};
  TimeLoc current = end;
  while (true) {
    if (auto it = came_from.find(current); it != came_from.end()) {
      total_path.emplace_back(std::move(current.second));
      current = it->second;
    } else {
      break;
    }
  }
  std::reverse(std::begin(total_path), std::end(total_path));
  return total_path;
}

std::vector<std::pair<double, TimeLoc>> get_next_gs(
    const TimeLoc& tu,
    const TimeLocCosts& g,
    double min_power,
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
    // next_g can have a fractional part (e.g. # of steps / 2^10),
    // so check a >= b + 1 instead of a > b
    double max_min_power = std::max(get_default(danger, tv, 0), min_power);
    if (next_g >= power_cycle[t + 1] - max_min_power + 1) {
      continue;
    }
    neighbors.emplace_back(next_g, tv);
  }

  for (auto& n : {Loc{0, 0}}) {
    auto& v = u;
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

std::vector<Loc> go_to(
    const NavState& state,
    size_t unit_id,
    const Loc& end,
    const Eigen::ArrayXXd& cost,
    const Eigen::ArrayXXd& h) {
  auto& unit = state.units[unit_id];
  auto& start = unit.loc;
  int32_t start_step = unit.step;
  int32_t max_step = start_step + state.max_turns;
  double min_power = unit.unit_type * 1000;
  double power = min_power + unit.r_at(lux::Resource::POWER);
  auto power_cycle = get_power_cycle(
      power,
      state.cost_table.unit_cfgs[unit.unit_type].CHARGE,
      start_step,
      max_step,
      state.cycle_length,
      state.day_length);

  auto next_g_func = [&](const TimeLoc& tu, const TimeLocCosts& g) {
    return get_next_gs(
        tu, g, min_power, power_cycle, cost, state.occupied, state.danger);
  };
  return a_star(
      start, end, state.max_turns, state.search_limit, h, next_g_func);
}

template <class G>
std::vector<Loc> a_star(
    const Loc& start,
    const Loc& end,
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

  g[{0, start}] = 0.0;
  f[{0, start}] = h(start.first, start.second);

  using Score = std::pair<double, TimeLoc>;
  std::priority_queue<Score, std::vector<Score>, std::greater<Score>> q;
  q.emplace(h(start.first, start.second), TimeLoc{0, start});
  size_t expanded = 0;
  while (!q.empty()) {
    auto [d, tu] = q.top();
    q.pop();
    auto& [t, u] = tu;
    if (u == end) {
      return trace_path(came_from, tu);
    }
    if (seen.contains(tu)) {
      continue;
    }
    seen.emplace(tu);
    if (t + 1 >= max_turns) {
      continue;
    }
    if (expanded >= search_limit) {
      return {};
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
  return {};
}

} // namespace anim
