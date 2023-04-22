#include "anim/bid.h"

#include <utility>
#include <vector>

#include "anim/state.h"

namespace anim {
namespace {
Eigen::ArrayXXd get_factory_dist(const Eigen::ArrayXXd& dist) {
  Eigen::ArrayXXd factory_dist =
      Eigen::ArrayXXd::Constant(dist.rows(), dist.cols(), INF);
  auto m = dist.rows();
  auto n = dist.cols();
  for (Eigen::Index i = 0; i < 3; i++) {
    for (Eigen::Index j = 0; j < 3; j++) {
      int32_t k = abs(i - 1) + abs(j - 1);
      auto d = dist.block(i, j, m - 2, n - 2);
      auto f = factory_dist.block(1, 1, m - 2, n - 2);
      f = f.min(d + k);
    }
  }
  return factory_dist;
}

Eigen::ArrayXXd get_score(
    const Eigen::ArrayXXd& cost,
    const Eigen::ArrayXXd& resource,
    double loss = 1e-6) {
  auto locs = argwhere(resource, nonzero_f);
  Eigen::ArrayXXd sum_score = Eigen::ArrayXXd::Zero(cost.rows(), cost.cols());
  Eigen::ArrayXXd max_score = Eigen::ArrayXXd::Zero(cost.rows(), cost.cols());
  for (auto& loc : locs) {
    auto power = get_factory_dist(dijkstra({loc}, cost, backwards));
    Eigen::ArrayXXd score = 1.0 / (power + loss);
    sum_score += score;
    max_score = max_score.max(score);
  }
  return (sum_score / locs.size() + max_score) / 2.0;
}

Eigen::ArrayXXd get_cost(const AgentState& state) {
  auto& board = state.game.board;
  Eigen::ArrayXXd rubble = board.rubble * (1 - board.ice) * (1 - board.ore);
  auto& unit_cfg = state.env_cfg.ROBOTS.HEAVY;
  Eigen::ArrayXXd cost =
      (unit_cfg.MOVE_COST + rubble * unit_cfg.RUBBLE_MOVEMENT_COST).floor();
  auto& enemy_factories = state.game.factories.at(state.opp_player);
  for (auto& factory : enemy_factories) {
    auto& pos = factory.pos;
    cost.block(pos.x - 1, pos.y - 1, 3, 3) = INF;
  }
  return cost;
}

std::vector<std::pair<double, Loc>> make_sorted_scores(
    const AgentState& state) {
  auto& board = state.game.board;
  auto cost = get_cost(state);
  auto& ice = board.ice;
  auto& ore = board.ore;
  auto& spawns = board.valid_spawns_mask;

  auto ice_score = get_score(cost, ice);
  auto ore_score = get_score(cost, ore);

  Eigen::ArrayXXd overall = ice_score * (ore_score + 1 / 80.0);

  auto locs = argwhere(spawns, nonzero_f);
  std::vector<std::pair<double, Loc>> sorted_scores;
  for (auto& loc : locs) {
    double score = overall(loc.first, loc.second);
    sorted_scores.emplace_back(score, loc);
  }
  std::sort(sorted_scores.begin(), sorted_scores.end(), std::greater<>());
  return sorted_scores;
}
} // namespace

lux::BidAction make_bid(AgentState& state) {
  state.sorted_scores = make_sorted_scores(state);
  return {"AlphaStrike", 0};
}

lux::SpawnAction make_spawn(AgentState& state) {
  auto& valid_spawns_mask = state.game.board.valid_spawns_mask;
  for (auto& [score, loc] : state.sorted_scores) {
    if (valid_spawns_mask(loc.first, loc.second)) {
      return {{loc.first, loc.second}, 150, 150};
    }
  }
  return {};
}
} // namespace anim
