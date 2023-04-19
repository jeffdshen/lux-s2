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
  Eigen::ArrayXXd score = Eigen::ArrayXXd::Zero(cost.rows(), cost.cols());
  for (auto& loc : locs) {
    auto dist = get_factory_dist(dijkstra({loc}, cost, backwards));
    score += 1.0 / (dist + loss);
  }
  return score;
}

std::vector<std::pair<double, Loc>> make_sorted_scores(
    const AgentState& state) {
  auto& board = state.game.board;
  auto& cost = state.dcache.costs.at("P1");
  auto& ice = board.ice;
  auto& ore = board.ore;
  auto& spawns = board.valid_spawns_mask;

  auto ice_score = get_score(cost, ice, 60.0);
  auto ore_score = get_score(cost, ore, 60.0);

  Eigen::ArrayXXd overall = ice_score * (ore_score + 1 / 60.0);

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
