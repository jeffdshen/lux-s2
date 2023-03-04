#pragma once

#include <limits>
#include <queue>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>

namespace anim {
const int MAX_SIZE = 64;
const double INF = std::numeric_limits<double>::infinity();

using Loc = std::pair<int32_t, int32_t>;

const std::vector<Loc> NEIGHBORS{{-1, 0}, {0, -1}, {0, 1}, {1, 0}};

inline Loc add(const Loc& a, const Loc& b) {
  return {a.first + b.first, a.second + b.second};
}

inline Loc sub(const Loc& a, const Loc& b) {
  return {a.first - b.first, a.second - b.second};
}

struct LocHash {
  int64_t m;
  LocHash(int64_t m) : m(m) {}

  size_t operator()(const Loc& x) const { return x.first * m + x.second; }
};

inline Loc shape(const Eigen::ArrayXXd& m) {
  return {static_cast<int32_t>(m.rows()), static_cast<int32_t>(m.cols())};
}

inline bool in_bounds(const Loc& a, const Loc& b) {
  return a.first >= 0 && a.first < b.first && a.second >= 0 &&
      a.second < b.second;
}

template <class T>
Eigen::ArrayXXd to_eigen(const std::vector<std::vector<T>>& arr) {
  auto rows = arr.size();
  auto cols = arr[0].size();
  Eigen::ArrayXXd matrix(rows, cols);
  for (size_t i = 0; i < rows; i++) {
    for (size_t j = 0; j < cols; j++) {
      matrix(i, j) = arr[i][j];
    }
  }
  return matrix;
}

inline bool nonzero_f(double d) {
  return d > 0;
}

template <class F>
std::vector<Loc> argwhere(const Eigen::ArrayXXd& x, F f) {
  std::vector<Loc> res;
  for (int32_t i = 0; i < x.rows(); i++) {
    for (int32_t j = 0; j < x.cols(); j++) {
      if (f(x(i, j))) {
        res.emplace_back(i, j);
      }
    }
  }
  return res;
}

inline double forwards(const Loc&, const Loc& v, const Eigen::ArrayXXd& cost) {
  return cost(v.first, v.second);
}

inline double backwards(const Loc& u, const Loc&, const Eigen::ArrayXXd& cost) {
  return cost(u.first, u.second);
}

template <class F>
Eigen::ArrayXXd dijkstra(
    std::vector<Loc> starts, const Eigen::ArrayXXd& cost, F cost_f) {
  Eigen::ArrayXXd dist =
      Eigen::ArrayXXd::Constant(cost.rows(), cost.cols(), INF);
  std::unordered_set<Loc, LocHash> seen(MAX_SIZE * MAX_SIZE, LocHash(MAX_SIZE));
  for (auto& start : starts) {
    dist(start.first, start.second) = 0;
  }
  using Score = std::pair<double, Loc>;
  std::priority_queue<Score> q;
  for (auto& start : starts) {
    auto d = dist(start.first, start.second);
    q.emplace(d, start);
  }
  while (!q.empty()) {
    auto [d, u] = q.top();
    q.pop();
    if (seen.contains(u)) {
      continue;
    }

    seen.emplace(u);
    for (auto& n : NEIGHBORS) {
      Loc v = add(u, n);
      if (!in_bounds(v, shape(cost))) {
        continue;
      }
      double e = cost_f(u, v, cost);
      double next_d = dist(u.first, u.second) + e;
      if (next_d < dist(v.first, v.second)) {
        dist(v.first, v.second) = next_d;
        q.emplace(next_d, v);
      }
    }
  }
  return dist;
}

} // namespace anim
