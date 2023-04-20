#pragma once

#include <cstdint>
#include <string>

#include "anim/bid.h"
#include "anim/build.h"
#include "anim/lux.h"
#include "anim/mine.h"
#include "anim/state.h"

namespace anim {
inline json agent_setup(
    AgentState& state,
    const std::string& player,
    int64_t step,
    const lux::Observation& obs,
    int64_t remainingOverageTime) {
  state_reset(state, player, step, obs, remainingOverageTime);
  if (step == 0) {
    return make_bid(state);
  }

  bool place_turn = step % 2 == obs.teams.at(player).place_first;
  if (state.game.teams.at(state.player).factories_to_place && place_turn) {
    return make_spawn(state);
  }

  return json::object();
}

inline json agent_act(
    AgentState& state,
    const std::string& player,
    int64_t step,
    const lux::Observation& obs,
    int64_t remainingOverageTime) {
  // if (step > 2) {
  //   throw std::runtime_error("resign");
  // }
  state_reset(state, player, step, obs, remainingOverageTime);
  make_build(state);
  make_mine(state);
  return state.get_actions_json();
}
} // namespace anim
