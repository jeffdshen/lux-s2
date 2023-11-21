# Lux AI Season 2 Bot

Code for the 40th place submission to the [Lux AI Season 2 Competition](https://www.kaggle.com/competitions/lux-ai-season-2/).

## Description

Main algorithms:

1. The [navigation](./src/anim/nav.h) uses a space-time A* algorithm. It simulates game-related constraints, e.g. day-night cycles, enemy movement, etc. and outputs the minimum power path.
2. The [strategy](./src/anim/mine.h) makes tasks (e.g. mining a location, clearing rubble) for every unit, estimates their discounted value, and then optimizes for the best task selection using a greedy matching algorithm. This is since a greedy matching algorithm returns a maximal matching which is at most 2x off from optimal.

Possible improvements:

1. More types of tasks (attacking, etc.)
2. Use the actual value for the task rather than an optimistic estimate.
3. Take into account the value of not having to switch action queues.
4. Handle some edge-cases in discounting, or simulate out multiple steps.

## Getting Started

The code requires [Eigen](https://eigen.tuxfamily.org/) to be present under `src/Eigen`.

See the original [Lux README](./LUX_README.md) for how to compile, run, and submit.

## License

All code except under `src/lux` or `main.cpp` is licensed under MIT license. See the [LICENSE](./LICENSE) file.

For `nlohmann_json.hpp`, see JSON for Modern C++'s [MIT License](https://github.com/nlohmann/json/blob/develop/LICENSE.MIT).

For everything else, see Lux S2's [Apache-2.0 License](https://github.com/Lux-AI-Challenge/Lux-Design-S2/blob/main/LICENSE).