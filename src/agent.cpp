#include "agent.hpp"

#include "lux/action.hpp"
#include "lux/common.hpp"
#include "lux/log.hpp"

#include "anim/actor.h"

json Agent::setup() {
  return agent_setup(state, player, step, obs, remainingOverageTime);
}

json Agent::act() {
  json actions = json::object();
  for (const auto& [unitId, factory] : obs.factories[player]) {
    if (step % 4 < 3 && factory.canBuildLight(obs)) {
      actions[unitId] = factory.buildLight(obs);
    } else if (factory.canBuildHeavy(obs)) {
      actions[unitId] = factory.buildHeavy(obs);
    } else if (factory.canWater(obs)) {
      actions[unitId] = factory.water(
          obs); // Alternatively set it to lux::FactoryAction::Water()
    }
  }
  for (const auto& [unitId, unit] : obs.units[player]) {
    for (int64_t i = 1; i < 5; ++i) {
      auto direction = lux::directionFromInt(i);
      auto moveCost = unit.moveCost(obs, direction);
      if (moveCost >= 0 && unit.power >= moveCost + unit.actionQueueCost(obs)) {
        LUX_LOG(
            "ordering unit " << unit.unit_id << " to move in direction " << i);
        // Alternatively, push lux::UnitAction::Move(direction, 0)
        actions[unitId].push_back(unit.move(direction, 2));
        break;
      }
    }
  }
  // dump your created actions in a file by uncommenting this line
  // lux::dumpJsonToFile("last_actions.json", actions);
  // or log them by uncommenting this line
  // LUX_LOG(actions);
  return actions;
}
