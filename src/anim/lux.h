#pragma once

#include "lux/action.hpp"
#include "lux/board.hpp"
#include "lux/config.hpp"
#include "lux/exception.hpp"
#include "lux/json.hpp"
#include "lux/log.hpp"
#include "lux/observation.hpp"

#define LUX_INFO(...) std::cerr << __VA_ARGS__ << std::endl

namespace anim {
  constexpr size_t MAX_RESOURCE_TYPE = 5;
  constexpr size_t MAX_ACTION_TYPE = 6;
  constexpr size_t MAX_UNIT_TYPE = 2;
}