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
  Eigen::ArrayXXd anti_scores =
      Eigen::ArrayXXd::Zero(rubble.rows(), rubble.cols());
  auto cost_name = "T0";

  for (auto& spots : state.factory_spots) {
    auto& dist = state.dcache.backward(spots, cost_name);
    // worth 400 at 1, 200 at 6, 400/3 at 11
    auto& factory_score = (2 * 200.0 * 5) / (dist.floor() + 4);
    auto& rubble_dist = state.dcache.backward(spots, "R");
    scores = scores.max(factory_score - rubble_dist / 2.0 * 5.0);
  }

  for (auto& spots : state.enemy_adj) {
    auto& dist = state.dcache.backward(spots, cost_name);
    // worth 400 at 1, 200 at 6, 400/3 at 11
    auto& factory_score = (2 * 200.0 * 5) / (dist.floor() + 5);
    auto& rubble_dist = state.dcache.backward(spots, "R");
    anti_scores = anti_scores.max(factory_score - rubble_dist / 2.0 * 5.0);
  }
  for (Eigen::Index i = 0; i < rubble.rows(); i++) {
    for (Eigen::Index j = 0; j < rubble.cols(); j++) {
      if (rubble(i, j) <= 0.0) {
        scores(i, j) = 0.0;
      } else {
        scores(i, j) -= anti_scores(i, j);
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

inline Eigen::ArrayXXd get_enemy_lichen_scores(AgentState& state) {
  auto& lichen = state.team_lichen[state.opp_player].lichen;
  Eigen::ArrayXXd scores =
      Eigen::ArrayXXd::Constant(lichen.rows(), lichen.cols(), INF);
  auto cost_name = "P0";

  for (auto& spots : state.enemy_adj) {
    auto& dist = state.dcache.backward(spots, cost_name);
    scores = scores.min(dist.floor());
  }
  scores += lichen;
  for (Eigen::Index i = 0; i < lichen.rows(); i++) {
    for (Eigen::Index j = 0; j < lichen.cols(); j++) {
      if (lichen(i, j) <= 0.0) {
        scores(i, j) = 0.0;
      }
    }
  }
  return scores;
}

inline void add_lichen_scores(AgentState& state) {
  auto& rubble = state.game.board.rubble;
  auto zeros = Eigen::ArrayXXd::Zero(rubble.rows(), rubble.cols());
  state.lichen_scores = RubbleScores{zeros, zeros, std::vector<Loc>{}};
  state.lichen_scores.value = get_enemy_lichen_scores(state);
  state.lichen_scores.make_locs_sorted();
}
} // namespace anim