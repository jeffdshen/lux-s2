#include "agent.hpp"

#include "lux/action.hpp"
#include "lux/common.hpp"
#include "lux/log.hpp"

#include "anim/actor.h"

json Agent::setup() {
  return agent_setup(state, player, step, obs, remainingOverageTime);
}

json Agent::act() {
  json actions = agent_act(state, player, step, obs, remainingOverageTime);
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
  return actions;
}
