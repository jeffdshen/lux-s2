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
  return actions;
}
