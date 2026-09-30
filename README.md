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
- **Stage 2 — AlphaZero loop.** ✅ loop works, strength WIP (2026-07-10) Add a policy+value network. Self-play → train policy head on MCTS visit distributions, value head on outcomes → stronger search. The credit-assignment sidestep, experienced firsthand — literally: the winner-imitation control (`catan_rl/imitation.py`) drove its loss 4.1→0.8 over 15 iterations while win rate vs weighted stayed 0–3% and games stopped terminating; the AZ loop (`catan_rl/az.py`) climbed 0→20% on the same eval in 6 iterations. Two findings: (1) AlphaZero's bootstrap silently assumes games end — Catan under aimless play doesn't, so cold-start self-play stalls at the turn limit with signal-free z; fixed by a turn-cap curriculum scored by VP leader. (2) A per-sample LayerNorm "fix" on the heterogeneous feature vector made things worse, not better. Laptop-scale caveat: ~300 self-play games/run is 3+ orders of magnitude below AlphaZero regimes; beating `weighted` outright with the net is an overnight-compute problem (see `runs/az6.log`), not an algorithm problem.
- **Stage 3 — Hidden information.** ✅ (2026-07-10) Dev cards and unknown hands: determinization (sample consistent worlds, search each) vs. belief-state features. The POMDP stage. Result (`catan_rl/determinize.py`): perfect-info search cheats in the *tree* (opponent dev hand + deck order), not the features — Catanatron's feature vector was already information-set honest. Vs weighted: cheating 1×100-sim tree 82%, honest 4×50-sim worlds 77%, honest 8×13-sim worlds 57%, cheating 1×13-sim tree 50%. The honesty penalty is a few points; shallow trees cost 25+. In 1v1 Catan, hidden dev cards are cheap — search depth is expensive. (Belief-state features: not yet.)
- **Stage 4 — Multi-agent.** ✅ machinery (2026-07-10) 4 players, opponent pool with Elo, population-based self-play. Where the minimax story ends and equilibrium weirdness begins. The Stage 1 MCTS transferred to 4p unchanged (its per-mover backup was already max^n); the AZ net grew a per-seat value vector (`num_values`) since there's no single adversary to negate against. Pool tournament (`catan_rl/pool.py`, 60 games): value 1255 Elo (48/49 wins), mcts:60 990, weighted 943, random 921, vp 891 — vp, mid-pack in 1v1, is dead last at a 4p table. League training exists (`az.py --pool`); a serious 4p population run is future compute.
- **Stage 5 (stretch) — Trading + human data.** Player-to-player trades; Colonist.io replays as behavior-cloning warm start and human-anchored eval.
- **Stage 6 — An LLM policy trained with RL.** Scoped (2026-09-30). A small language model plays Catan decisions, and the search from Stages 1 to 3 supplies the reward. This is RL on an LLM (policy-gradient fine-tuning), where Stages 1 to 4 were RL on a small network. Laptop-scale: Qwen2.5-0.5B-Instruct with LoRA on MPS, overnight runs.
  - **6a. Text interface.** Serialize the game state and the legal actions into a prompt, give the model a short reasoning budget (about 64 tokens), and parse the chosen action. Baseline before any training: agreement with a 200-sim MCTS on held-out states, and win rate vs `weighted` with the model playing every decision.
  - **6b. Reward from search.** Collect decision states from existing self-play, and cache each legal action's MCTS value estimate. The reward for a completion is the normalized value of the action it picks, with a penalty for unparseable output. The reward is dense, computed rather than judged, and cheap to recompute.
  - **6c. GRPO, hand-written in PyTorch.** For each prompt, sample a group of G completions; each completion's advantage is its reward minus the group mean, divided by the group standard deviation. Update with the clipped policy ratio plus a KL penalty against the frozen reference model. No TRL: the point is to write the loop.
  - **6d. Evaluation and reward hacking.** Re-run the 6a baselines after training. Watch for the known failure modes: format exploits, collapse onto one safe action (END_TURN), and agreement rising while win rate does not (the Stage 2 imitation lesson again).
  - **Stop rule, fixed before the first run.** Success is +10 points of held-out MCTS agreement over the 6a baseline without a drop in win rate vs `weighted`. Whatever the outcome, it gets written up.

Each stage ends with a written artifact (thread/post) before the next begins.

## Constraints

- Laptop-scale (M-series Mac, MPS/CPU). Small nets, fast env, sample efficiency over brute force.
- When goals conflict, understanding wins: hand-implement the core algorithm even when a library would be faster.

## Running

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
uv run pytest
uv run python -m catan_rl.benchmark value weighted -n 100        # heuristic ladder (Stage 0)
uv run python -m catan_rl.benchmark mcts:100 weighted -n 50      # pure MCTS (Stage 1)
uv run python -m catan_rl.az --iterations 6 --out runs/az        # AlphaZero loop (Stage 2)
uv run python -m catan_rl.pool value mcts:60 weighted random vp -n 60   # 4-player Elo pool (Stage 4)
```

Run outputs and checkpoints go to `runs/`, which is not tracked.

## License

MIT. See [LICENSE](LICENSE).
