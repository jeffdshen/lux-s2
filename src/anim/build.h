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
  auto& units = state.game.units[state.player];

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

    if (state.water_costs[i] <= factory.cargo.water / 2 - 200) {
      state.actions.factories[i] = lux::FactoryAction::Water();
    }
  }

  for (size_t i = 0; i < factories.size(); i++) {
    auto& factory = factories[i];

    double metal_power = factory.cargo.metal * unit_cfgs.HEAVY.POWER_COST /
        unit_cfgs.HEAVY.METAL_COST;
    double free_power = factory.power - metal_power;
    state.free_factory_power.emplace_back(free_power);
  }
}
} // namespace anim