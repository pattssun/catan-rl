# Replay and terminal-return experiment

Can we restore a decision state exactly? If so, does rewarding actual, earlier wins provide a more useful training signal than awarding a win to the VP leader at a rollout cutoff?

The previous audit found 77/192 flat training states and eight replay mismatches in a 24-state panel. Last-ID choice scored 71.1% agreement, above GRPO's 67.2%. These findings justify checking the environment and reward before another training run. The original experiments and their failed success rules remain unchanged.

## Limits

- One local CPU worker. No model training, paid compute, or new model downloads.
- At most two hours for the measured panel, including collection and verification. Each state comparison has a five-minute limit, capped by the remaining budget. Missing states stay missing; no replacements or extensions.
- Before measurement, freeze this protocol and the code in a fresh run directory. Record commands, source and dependency hashes, engine version, action ordering, seeds, elapsed time, and missing evidence.
- Keep historical datasets and interfaces unchanged. This experiment uses newly collected states.
- Stop at a failed prerequisite. Report failure or incomplete evidence without tuning the thresholds or trying another reward variant.

## 1. Exact state recovery

Diagnose action-order and random-number differences across fresh processes. Save versioned JSON with enough information to recover hidden hands, deck order, board state, pending decisions, and environment RNG state. An action journal is acceptable only if it records chance outcomes and verifies the complete recovered semantic state.

Canonicalize legal-action order at the new environment boundary without adding or removing actions. Keep this separate from the original Stage 6 interface. Reject corrupt records and incompatible engine versions. Restoring a state must not change unrelated random streams.

**Check:** cover setup, ordinary play, development cards, and robber decisions. Restore each fixture in fresh processes with `PYTHONHASHSEED=0,1,2`. Compare full semantic state and legal actions, then the next 100 actions or terminal outcome under the same deterministic policy and RNG. All checks must pass before the reward comparison. Matching only the public prompt is insufficient.

## 2. One fixed reward comparison

Collect 12 source games using seeds 230000 through 230011, `value` versus `weighted`, with alternating seats and canonical action order. Use reservoir sampling to select one setup and one play decision per game. Exclude forced decisions and menus over 100 actions. Use sampling seed `game_seed + 17` and decision seed `game_seed + 29`; cap each game at 400 turns or 8,000 actions. Save complete states when selected. Missing phases remain missing.

This gives up to 24 diagnostic states, not a held-out playing-strength evaluation. Compare two objectives on exactly these states:

| Setting | Control | Candidate |
| --- | --- | --- |
| Terminal outcome | Win +1, draw 0, loss -1 | Same |
| Nonterminal cutoff | VP leader wins; tied leaders draw | 0 |
| Time preference | None | Terminal return multiplied by `0.995 ** k` |
| Stored value | `(return + 1) / 2` | Same |

`k` is the number of actions after the candidate action until termination. An immediate winning action has `k=0` and value 1. Count tree edges as well as rollout actions when backing up a discounted return. Values are from the perspective of the player who chose the action.

Both arms use two determinized development-card worlds, 100 MCTS simulations per world, a 120-turn rollout horizon, and a 1,000-action rollout cap. Run three repeats per state with seeds `940000 + 10 * state_index + repeat_index`. Enumerate states by source seed, then setup/play. Resource hands remain privileged teacher information and must be labeled as such.

The control shares the new action cap and canonical ordering. It is a diagnostic control, not a reproduction of the original uncapped labels. The candidate changes both cutoff handling and time preference; this comparison cannot isolate their separate effects.

**Correctness check:** independently calculated trajectories and real-engine fixtures verify immediate wins, delayed wins, losses, draws, cutoff behavior, and player perspective. A cutoff lead must never count as a terminal win in the candidate.

**Acceptance checks, fixed before measurement:**

- All integrity, replay, and correctness checks pass; every action is visited in both worlds.
- All 24 states and all required repeats finish within budget.
- At least 18/24 candidate states have non-flat values in every repeat, using tolerance `1e-8`.
- At least 20/24 candidate states are non-flat in every repeat and share at least one maximizing semantic action across all three repeats.

The last threshold also implies the non-flat threshold. Report both as separate diagnostics, alongside the control's rates, value spreads, ties, visits, rollout cutoff frequency, and runtime. Passing these engineering checks establishes usable and repeatable signal on this small panel, not correct action rankings or stronger play.

## 3. Reproduction and review

Provide one command for replay checks and one for the comparison. Preserve raw states, repeated labels, manifests, hashes, and incomplete units. Recompute aggregates without rerunning search. Keep the previous audit available.

Extend the existing dashboard only where needed to compare the two objectives and inspect missing or failed evidence. Reuse the current state and board views.

**Check:** relevant Python tests pass. If the UI changes, the frontend builds and browser checks cover complete, failed, and missing evidence. Every reported number traces to a saved artifact. The original experiment results remain unchanged.

## 4. Learning checks

Extend [LEARNING.md](LEARNING.md) with three exercises grounded in this implementation:

1. Explain why a seed is not a saved state. Predict how action ordering and RNG position can change a replay, then verify a small example.
2. Calculate discounted returns and back up a tiny search tree from both players' perspectives. Check against independently calculated values.
3. Explain why more distinct rewards can still rank actions badly. Specify an experiment and a result that would reject the explanation.

Each exercise needs a prediction, a small calculation or code change, and an independent check. Completion means explaining the result without notes and correctly predicting a new case. Preparing exercises does not establish that understanding.

## Decision after the comparison

If a prerequisite or reward check fails, retain the result and identify the next specific cause to investigate. No additional reward variants or optimizer updates belong to this experiment.

If all checks pass, design a separate post-training experiment before running it: fresh game-separated train/dev/test data, untuned and SFT controls, hand-written GRPO, action-order controls, informative-state agreement, output validity, and paired actual-win evaluation. Freeze checkpoint selection, success thresholds, and a single overnight budget before the first update. Keep learning stronger play separate from matching a teacher.

## Progress

- [x] Exact state recovery and independent cross-process checks
- [x] Fixed reward comparison, or documented prerequisite failure
- [x] Reproducible artifacts and review checks
- [x] Learning exercises

Replay validation covers 12 setup/play/card/robber situations, including pending robber movement and free roads. Each matches the original state's next 100 transitions in three fresh Python processes. Independent checks compare state fields, hidden hands, deck order, legal successors, and RNG state. A separate trade-state probe reproduces differing raw action-menu order under hash seeds 0, 1, and 2; canonical menus agree. This demonstrates an ordering hazard without attributing every historical mismatch to it.

The registered attempt stopped at the replay gate after 78.7 seconds. All 24 original checks failed because the fingerprint encoded object attribute order; the engine's copy changes that order while retaining field values. No reward searches or optimizer updates ran. The original report remains incomplete.

A separate 36.5-second diagnosis used the frozen serializer and the same states and traces. After an engine copy normalized attribute order, all 72 state/process checks had identical fields and matched all 100 saved transitions. This diagnoses the checker, not the reward. The serializer now sorts attribute names, keeps nested dictionary order, and uses snapshot schema v2. New regression tests cover live collection, which the original copied-state fixtures missed.

The next reward attempt requires a separate registration. Do not extend this attempt or turn the diagnosis into a claim that the reward gate passed. Exercises 6 to 8 are prepared; explaining and modifying the implementation remains a separate learning check.

Verification: 85 Python tests pass, including current live-state recovery, per-world action coverage, failed and missing gates, and exact recomputation from the bundled original attempt. The frontend builds. Browser checks covered the measured incomplete report and diagnosis, board/state selection, and labeled synthetic complete, failed, and missing reports. Pytest discovery is restricted to `tests/` so frozen run snapshots are not collected as current tests.
