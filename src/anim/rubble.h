#pragma once

#include <cstdint>
#include <string>
#include <variant>

#include <Eigen/Dense>

#include "anim/lux.h"
#include "anim/nav.h"
#include "anim/state.h"
#include "anim/utils.h"

namespace anim {
inline Eigen::ArrayXXd get_lichen_scores(AgentState& state) {
  auto& rubble = state.game.board.rubble;
  auto& ice = state.game.board.ice;
  auto& ore = state.game.board.ore;
  Eigen::ArrayXXd scores = Eigen::ArrayXXd::Zero(rubble.rows(), rubble.cols());
  auto cost_name = "T0";

  for (auto& spots : state.factory_spots) {
    auto& dist = state.dcache.backward(spots, cost_name);
    // worth 400 at 1, 200 at 6, 400/3 at 11
    auto& factory_score = (2 * 200.0 * 5) / (dist.floor() + 4);
    scores = scores.max(factory_score);
  }
  for (Eigen::Index i = 0; i < rubble.rows(); i++) {
    for (Eigen::Index j = 0; j < rubble.cols(); j++) {
      if (rubble(i, j) <= 0.0) {
        scores(i, j) = 0.0;
      }

      // TODO use DEFAULT_RESOURCE_VALUE
      if (ice(i, j) > 0.0) {
        scores(i, j) = 5.0 * rubble(i, j);
      }
      if (ore(i, j) > 0.0) {
        scores(i, j) = 5.0 * rubble(i, j);
      }
    }
  }
  return scores;
}

inline void add_rubble_scores(AgentState& state) {
  auto& rubble = state.game.board.rubble;
  auto zeros = Eigen::ArrayXXd::Zero(rubble.rows(), rubble.cols());
  state.rubble_scores = RubbleScores{zeros, zeros, std::vector<Loc>{}};
  state.rubble_scores.reward += get_lichen_scores(state);
  state.rubble_scores.value = state.rubble_scores.reward;
  state.rubble_scores.value -= rubble / 2.0 * 5.0;
  state.rubble_scores.make_locs_sorted();
}

} // namespace anim