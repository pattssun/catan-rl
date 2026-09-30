# catan-rl — orientation for Codex sessions

Learning project: train a Catan AI to deeply understand RL / self-play / tree search. Modeled on [Eric Jang's AlphaGo-from-scratch rebuild](https://www.dwarkesh.com/p/eric-jang) (the conceptual backbone: MCTS/PUCT mechanics, credit-assignment sidestep, hidden-information framing). Second inspiration: [Edward Zhou's Catan RL bot](https://www.edwardzhou.com/projects/catan-rl-bot).

## Locked decisions (2026-07-10)

1. **Understanding wins every conflict.** Hand-implement MCTS, the self-play loop, and the policy/value training — even when Catanatron ships a bot or a library does it better. Don't "helpfully" swap in a faster off-the-shelf implementation.
2. **Catanatron is the rules engine, nothing more.** We consume its game state, legal-action masking, and heuristic bots (as opponents/benchmarks). All search and learning code is ours.
3. **Staged scope — one Catan complication per stage** (see README roadmap). Don't pull forward complexity from a later stage; 1v1-no-trading is not a placeholder, it's Stage 1's whole world.
4. **Laptop-scale.** M-series Mac, MPS/CPU. If a design needs a GPU cluster, it's the wrong design for this project.
5. **Build in public, Patrick's voice.** Each stage ends with a Patrick-written thread/post. Sessions should surface "this is postable" moments but never auto-post.

## Style

- Prefer editing existing files over creating new ones.
- Default to no comments; comment only non-obvious *why*.
- No speculative abstractions, no backwards-compat shims.
- No planning/analysis docs unless asked — conversation is the planning surface; README roadmap is the tracker.
- Propose changes, ask before `git commit`.
