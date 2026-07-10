# catan-rl

Training a Catan AI from scratch to deeply understand RL, self-play, and tree search.

Inspirations:
- [Eric Jang's AlphaGo-from-scratch rebuild](https://www.dwarkesh.com/p/eric-jang) — the playbook: hand-implement search + learning + self-play as the "primitives of intelligence," write it up in public.
- [Edward Zhou's Catan RL bot](https://www.edwardzhou.com/projects/catan-rl-bot) — gym-style Catan env, PPO baselines, Colonist.io replay crawler.

Why Catan and not Go: Catan breaks every assumption that makes AlphaZero clean — dice (stochastic transitions → chance nodes), hidden dev cards and hands (POMDP → belief tracking), 4 players (no minimax value), and trading (negotiation, exploding action space). Each stage of this project takes on exactly one of those breaks.

## Roadmap

Engine substrate: [Catanatron](https://github.com/bcollazo/catanatron) (open-source Catan engine, gym env, heuristic benchmark bots). The RL/search code is hand-written; the rules engine is not — the learning goal is the algorithms, not longest-road edge cases.

- **Stage 0 — Substrate.** ✅ (2026-07-10) Install Catanatron, read its state/action model, benchmark its heuristic bots against each other. Deliverable: a harness that runs N games between any two agents and reports win rates — `catan_rl/benchmark.py`. Ladder result: `value` ≫ `vp` ≈ `weighted` > `random`, and `value` beats `alphabeta` 58/42 at ~80× less compute — the bot ladder is a cliff, not a slope.
- **Stage 1 — Pure search.** ✅ (2026-07-10) 1v1, no player trading. Hand-written MCTS (UCT, then expectimax-style chance nodes for dice) with random rollouts — no neural net. Target: beat Catanatron's weak/medium heuristics. Result (`catan_rl/mcts.py`, 100 sims, 120-turn rollout horizon): 90% vs random, 82% vs weighted, 90% vs vp. Lesson learned: horizon-capped rollouts scored by VP leader make every move from a winning position look equal — the search dithers at the doorstep until a decisive-move check (game logic, not evaluation) takes the certain win.
- **Stage 2 — AlphaZero loop.** Add a policy+value network. Self-play → train policy head on MCTS visit distributions, value head on outcomes → stronger search. The credit-assignment sidestep, experienced firsthand.
- **Stage 3 — Hidden information.** Dev cards and unknown hands: determinization (sample consistent worlds, search each) vs. belief-state features. The POMDP stage.
- **Stage 4 — Multi-agent.** 4 players, opponent pool with Elo, population-based self-play. Where the minimax story ends and equilibrium weirdness begins.
- **Stage 5 (stretch) — Trading + human data.** Player-to-player trades; Colonist.io replays as behavior-cloning warm start and human-anchored eval.

Each stage ends with a written artifact (thread/post) before the next begins.

## Constraints

- Laptop-scale (M-series Mac, MPS/CPU). Small nets, fast env, sample efficiency over brute force.
- When goals conflict, understanding wins: hand-implement the core algorithm even when a library would be faster.
