#pragma once

#include <chrono>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <Eigen/Dense>

#include "anim/lux.h"

namespace anim {
const int MAX_SIZE = 63;
const double INF = std::numeric_limits<double>::infinity();
const double EPS = 0x1.0p-20;

using Loc = std::pair<int32_t, int32_t>;
using TimeLoc = std::pair<int32_t, Loc>;

const std::vector<Loc> NEIGHBORS{{-1, 0}, {0, -1}, {0, 1}, {1, 0}};

inline double zipd(double a, double b) {
  return a + (b * EPS);
}

inline std::pair<double, double> unzipd(double x) {
  return {std::round(x), (x - static_cast<int64_t>(std::round(x))) / EPS};
}

inline void unzipd(double x, double& a, double& b) {
  auto [aa, bb] = unzipd(x);
  a = aa;
  b = bb;
}

inline std::pair<Eigen::ArrayXXd, Eigen::ArrayXXd> unzipd(
    const Eigen::ArrayXXd& x) {
  // ieee754 std::round(INF) is well defined
  Eigen::ArrayXXd a = x.round();
  // ieee754 handle INF so we get INF - MAX = INF
  Eigen::ArrayXXd b = (x - a.min(std::numeric_limits<double>::max())) / EPS;
  return {std::move(a), std::move(b)};
}

inline Loc add(const Loc& a, const Loc& b) {
  return {a.first + b.first, a.second + b.second};
}

inline Loc sub(const Loc& a, const Loc& b) {
  return {a.first - b.first, a.second - b.second};
}

inline Loc to_loc(const lux::Position& pos) {
  return {pos.x, pos.y};
}

inline Loc add(const Loc& a, const lux::Direction& d) {
  return add(a, to_loc(lux::Position::Delta(d)));
}

const std::array<lux::Direction, 9> LOC_TO_DIR{
    lux::Direction::CENTER,
    lux::Direction::LEFT,
    lux::Direction::CENTER,
    lux::Direction::UP,
    lux::Direction::CENTER,
    lux::Direction::DOWN,
    lux::Direction::CENTER,
    lux::Direction::RIGHT,
    lux::Direction::CENTER};

inline lux::Direction to_dir(const Loc& a) {
  return LOC_TO_DIR[3 * (a.first + 1) + (a.second + 1)];
}

struct LocHash {
  int64_t m;
  LocHash() : LocHash(MAX_SIZE) {}
  LocHash(int64_t mm) : m(mm) {}

  size_t operator()(const Loc& x) const { return x.first * m + x.second; }
};

struct TimeLocHash {
  int64_t m;
  TimeLocHash() : TimeLocHash(MAX_SIZE) {}
  TimeLocHash(int64_t mm) : m(mm) {}

  size_t operator()(const TimeLoc& x) const {
    return m * m * x.first + x.second.first * m + x.second.second;
  }
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

template <
    class Map,
    typename Key = typename Map::key_type,
    typename Value = typename Map::mapped_type>
typename Map::mapped_type get_default(
    const Map& map, const Key& key, const typename Map::mapped_type& value) {
  if (auto pos = map.find(key); pos != map.end()) {
    return pos->second;
  }
  return value;
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
  std::unordered_set<Loc, LocHash> seen(MAX_SIZE * MAX_SIZE);
  using Score = std::pair<double, Loc>;
  std::priority_queue<Score, std::vector<Score>, std::greater<Score>> q;
  for (auto& start : starts) {
    dist(start.first, start.second) = 0.0;
    q.emplace(0.0, start);
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

struct DijkstraCache {
  std::unordered_map<
      std::string,
      std::unordered_map<Loc, Eigen::ArrayXXd, LocHash>>
      f_dists;
  std::unordered_map<
      std::string,
      std::unordered_map<Loc, Eigen::ArrayXXd, LocHash>>
      b_dists;
  std::unordered_map<std::string, std::map<std::vector<Loc>, Eigen::ArrayXXd>>
      b_multi_dists;
  std::unordered_map<std::string, Eigen::ArrayXXd> costs;

  // Move only
  DijkstraCache() = default;
  DijkstraCache(const DijkstraCache& other) = delete;
  DijkstraCache& operator=(const DijkstraCache& other) = delete;
  DijkstraCache(DijkstraCache&& other) noexcept = default;
  DijkstraCache& operator=(DijkstraCache&& other) noexcept = default;
  ~DijkstraCache() = default;

  void add_cost(std::string name, Eigen::ArrayXXd cost) {
    costs.emplace(std::move(name), std::move(cost));
  }

  const Eigen::ArrayXXd& forward(const Loc& start, const std::string& name) {
    auto& dist = f_dists[name];
    if (auto it = dist.find(start); it != dist.end()) {
      return it->second;
    }

    auto& cost = costs.at(name);
    auto [it, _] = dist.emplace(start, dijkstra({start}, cost, forwards));
    return it->second;
  }

  const Eigen::ArrayXXd& backward(const Loc& end, const std::string& name) {
    auto& dist = b_dists[name];
    if (auto it = dist.find(end); it != dist.end()) {
      return it->second;
    }

    auto& cost = costs.at(name);
    auto [it, _] = dist.emplace(end, dijkstra({end}, cost, backwards));
    return it->second;
  }

  const Eigen::ArrayXXd& backward(
      const std::vector<Loc>& ends, const std::string& name) {
    if (ends.size() == 1) {
      return backward(ends[0], name);
    }

    std::vector<Loc> e{ends};
    std::sort(std::begin(e), std::end(e));
    auto& dist = b_multi_dists[name];
    if (auto it = dist.find(e); it != dist.end()) {
      return it->second;
    }

    auto& cost = costs.at(name);
    auto [it, _] = dist.emplace(e, dijkstra(e, cost, backwards));
    return it->second;
  }
};

// A zone for groups of locs
struct Zones {
  Eigen::ArrayXXd dist;
  Eigen::ArrayXXd zone;
  std::vector<std::vector<Loc>> to_loc;
  std::unordered_map<Loc, size_t, LocHash> from_loc{MAX_SIZE * MAX_SIZE};
  std::vector<size_t> to_id;

  // Move only
  Zones() = default;
  Zones(const Zones& other) = delete;
  Zones& operator=(const Zones& other) = delete;
  Zones(Zones&& other) noexcept = default;
  Zones& operator=(Zones&& other) noexcept = default;
  ~Zones() = default;
};

struct ZonesCache {
  DijkstraCache* dcache = nullptr;
  std::unordered_map<std::string, std::unordered_map<std::string, Zones>> zones;

  Zones& get_zone(const std::string& zone_type, const std::string& cost_name) {
    return zones.at(zone_type).at(cost_name);
  }

  void make_zones(
      const std::string& zone_type,
      const std::string& cost_name,
      const std::vector<std::vector<Loc>>& loc_groups) {
    Zones zone;
    if (loc_groups.empty()) {
      zones[zone_type][cost_name] = std::move(zone);
      return;
    }
    std::vector<const Eigen::ArrayXXd*> dists;
    for (size_t i = 0; i < loc_groups.size(); i++) {
      auto& locs = loc_groups[i];
      dists.emplace_back(&dcache->backward(locs, cost_name));
      zone.to_id.emplace_back(i);
      zone.to_loc.emplace_back(locs);
      for (auto& loc : locs) {
        zone.from_loc[loc] = i;
      }
    }
    auto rows = dists[0]->rows();
    auto cols = dists[0]->cols();
    Eigen::ArrayXXd dist = *dists[0];
    Eigen::ArrayXXd idx = Eigen::ArrayXXd::Zero(rows, cols);
    for (Eigen::Index x = 0; x < rows; x++) {
      for (Eigen::Index y = 0; y < cols; y++) {
        for (size_t i = 1; i < dists.size(); i++) {
          if (dist(x, y) > (*dists[i])(x, y)) {
            dist(x, y) = (*dists[i])(x, y);
            idx(x, y) = i;
          }
        }
      }
    }
    zone.dist = std::move(dist);
    zone.zone = std::move(idx);
    zones[zone_type][cost_name] = std::move(zone);
  }
};

const std::vector<Loc> FACTORY_SPOTS = {
    {-1, -1}, {-1, 0}, {-1, 1}, {0, -1}, {0, 1}, {1, -1}, {1, 0}, {1, 1}};

inline std::vector<std::vector<Loc>> get_factory_spots(
    const std::vector<lux::Factory>& factories) {
  std::vector<std::vector<Loc>> spots;
  for (size_t i = 0; i < factories.size(); i++) {
    auto& factory = factories[i];
    spots.emplace_back();
    spots.back().reserve(FACTORY_SPOTS.size());
    for (auto& n : FACTORY_SPOTS) {
      spots.back().emplace_back(add(to_loc(factory.pos), n));
    }
  }
  return spots;
}

const std::vector<Loc> FACTORY_ADJACENT{
    {-2, -1},
    {-2, 0},
    {-2, 1},
    {-1, -2},
    {-1, 2},
    {0, -2},
    {0, 2},
    {1, -2},
    {1, 2},
    {2, -1},
    {2, 0},
    {2, 1}};
inline std::vector<std::vector<Loc>> get_factory_adjacent(
    const std::vector<lux::Factory>& factories, const Loc& bounds) {
  std::vector<std::vector<Loc>> spots;
  for (size_t i = 0; i < factories.size(); i++) {
    auto& factory = factories[i];
    spots.emplace_back();
    spots.back().reserve(FACTORY_ADJACENT.size());
    for (auto& n : FACTORY_ADJACENT) {
      auto spot = add(to_loc(factory.pos), n);
      if (!in_bounds(spot, bounds)) {
        continue;
      }
      spots.back().emplace_back(std::move(spot));
    }
  }
  return spots;
}

struct RubbleScores {
  Eigen::ArrayXXd reward;
  Eigen::ArrayXXd value;
  std::vector<Loc> locs;

  void make_locs_sorted() {
    locs = argwhere(value, nonzero_f);
    std::sort(locs.begin(), locs.end(), [&](auto& a, auto& b) {
      return value(a.first, a.second) > value(b.first, b.second);
    });
  }
};

inline lux::UnitAction pop_action(std::deque<lux::UnitAction>& actions) {
  if (actions.empty()) {
    return lux::UnitAction::Move(lux::Direction::CENTER, 0, 1);
  }

  auto& action = actions.front();
  auto result = action;
  if (action.isRechargeAction()) {
    return result;
  }

  action.n--;
  if (action.n > 0) {
    return result;
  }

  auto tmp = action;
  actions.pop_front();
  if (tmp.repeat > 0) {
    tmp.n = tmp.repeat;
    actions.push_back(tmp);
  }
  return result;
}

struct TeamLichenTable {
  Eigen::ArrayXXd lichen;
  Eigen::ArrayXXd factory_id;
};

template <
    class result_t = std::chrono::milliseconds,
    class clock_t = std::chrono::steady_clock,
    class duration_t = std::chrono::milliseconds>
auto since(std::chrono::time_point<clock_t, duration_t> const& start) {
  return std::chrono::duration_cast<result_t>(clock_t::now() - start);
}

class DurationTimer {
 public:
  DurationTimer(std::string name)
      : name_(std::move(name)), start_(std::chrono::steady_clock::now()) {}
  ~DurationTimer() {
    auto end = std::chrono::steady_clock::now();
    auto dur =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start_);
    LUX_INFO(name_ << ": " << dur.count() / 1000.0 << " seconds");
  }

 private:
  std::string name_;
  std::chrono::time_point<std::chrono::steady_clock> start_;
};
} // namespace anim