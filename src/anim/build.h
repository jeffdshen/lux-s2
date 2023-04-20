#pragma once

#include <cstdint>
#include <string>

#include <Eigen/Dense>

#include "anim/lux.h"
#include "anim/nav.h"
#include "anim/state.h"
#include "anim/utils.h"

namespace anim {
bool can_build(
    const lux::Factory& factory,
    const std::string& unit_type,
    const lux::UnitConfigs& unit_cfgs) {
  if (factory.cargo.metal < unit_cfgs[unit_type].METAL_COST) {
    return false;
  }

  if (factory.power < unit_cfgs[unit_type].POWER_COST) {
    return false;
  }

  // if occupied, that robot can just move on the same turn
  return true;
}

void make_build(AgentState& state) {
  auto& factories = state.game.factories[state.player];
  auto& units = state.my_units();

  std::unordered_map<std::string, size_t> unit_counts;
  for (auto& unit : units) {
    unit_counts[unit.unit_type]++;
  }

  auto& unit_cfgs = state.env_cfg.ROBOTS;
  for (size_t i = 0; i < factories.size(); i++) {
    auto& factory = factories[i];
    if (can_build(factory, "HEAVY", unit_cfgs)) {
      state.actions.factories[i] = lux::FactoryAction::BuildHeavy();
      continue;
    }

    if (unit_counts["HEAVY"] > 0.2 * unit_counts["LIGHT"]) {
      if (can_build(factory, "LIGHT", unit_cfgs)) {
        state.actions.factories[i] = lux::FactoryAction::BuildLight();
        continue;
      }
    }

    int64_t turns_left = state.env_cfg.max_episode_length - state.step;
    int64_t min_water =
        std::min<int64_t>(std::max<int64_t>(turns_left, 50), 200);
    if (factory.cargo.water - min_water >= 2 * state.water_costs[i]) {
      state.actions.factories[i] = lux::FactoryAction::Water();
    }
  }

  for (size_t i = 0; i < factories.size(); i++) {
    auto& factory = factories[i];

    double metal_power = factory.cargo.metal * unit_cfgs.HEAVY.POWER_COST /
        unit_cfgs.HEAVY.METAL_COST;

    double free_power = std::max(factory.power - metal_power, 0.0);
    state.free_factory_power.emplace_back(free_power);
  }
}
} // namespace anim