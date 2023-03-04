#pragma once

#include "anim/lux.h"
#include "anim/utils.h"
#include "anim/state.h"

namespace anim {
lux::BidAction make_bid(AgentState& state);

lux::SpawnAction make_spawn(AgentState& state);
} // namespace anim
