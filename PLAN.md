# Terminal-return retry and conditional post-training experiment

Can we restore a decision state exactly? If so, does rewarding actual, earlier wins provide a more useful training signal than awarding a win to the VP leader at a rollout cutoff?

The previous audit found 77/192 flat training states and eight replay mismatches in a 24-state panel. Last-ID choice scored 71.1% agreement, above GRPO's 67.2%. These findings justify checking the environment and reward before another training run. The original experiments and their failed success rules remain unchanged.

## Registered retry

Attempt `terminal-return-v1` stopped before reward search because the fingerprint included object attribute assignment order. Its report, frozen source, and separate diagnosis remain in `examples/terminal-return-inputs.zip`. All 72 follow-up checks matched field values and saved transitions after an engine copy normalized that order. No optimizer updates ran.

This is a separate attempt, `terminal-return-v2`. The only experimental change is the corrected snapshot checker, `catan_environment_v2`, with live-collection regression coverage. Keep the reward protocol, source-game seeds, sampling, search seeds, thresholds, and budgets below unchanged. Recollect the fixed panel; do not convert or overwrite the original snapshots. This is a known diagnostic panel, not fresh held-out evidence.

Freeze this document, code, tests, dependency lock, and hashes in the new run before measurement. Verification and packaging may change afterward; measured source and evidence may not. No third attempt, reward variant, extra training arm, or threshold change is authorized by this plan.

## Limits

- One local CPU worker for the reward comparison. No model training until the comparison passes. No paid compute or new model downloads in either phase.
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

A failed or incomplete comparison ends the measurement work. Preserve partial units, recompute the report, and state the specific observed failure. Do not train, replace missing states, or launch another reward variant. A negative result can complete this plan.

Only a complete pass permits the following separate post-training experiment. Implement and test its plumbing first, then freeze a second registration and source snapshot before collecting its data or updating weights. That registration must implement these fixed choices; it cannot relax them in response to results.

### Conditional data and controls

- Use fresh source games: train seeds 310000 through 310015, dev 320000 through 320003, test 330000 through 330011. Never use the diagnostic panel for training or evaluation.
- Collect two setup and four play states per game by phase-specific uniform reservoir sampling, with the same source opponents, alternating seats, canonical environment, RNG separation, menu limit, and game caps as above. Exclude prompts exceeding 2,048 tokens without truncation. Missing states are not replaced.
- Save full snapshots and live continuation references. Require exact replay and complete teacher action coverage for all 192 states. Label with the candidate objective above, one two-world search per state, seed `game_seed * 10000 + action_tick`. Retain raw values and visits.
- Train both arms on the same non-flat training states. Require at least 48 informative train states, 12 dev states, and 36 test states. Keep flat states in evaluation and report their count. Stop if these minimums or any integrity check fail.
- Use the existing pinned Qwen2.5-0.5B-Instruct weights, LoRA rank 8/alpha 16, and the hand-written PyTorch GRPO implementation. Compare untuned, SFT, and GRPO from the same initial weights. No TRL.
- Run 200 rounds, four sampled prompts per round, two optimizer updates per round, seed 611. GRPO group size is four. Keep the original learning rate `1e-5`, clipping `0.2`, KL coefficient `0.02`, gradient clip 1, temperature 1, and 16-token completion cap. SFT samples uniformly among tied best actions with a separate seeded RNG, rather than always selecting the smallest ID.
- Match the two arms' prompt schedule and update count. Record tokens and runtime; this is not a compute-matched comparison. Select the last completed round, with no selection on dev or test results. All 200 rounds must finish for a conclusive experiment.
- Hold training presentation in canonical order. Evaluate held-out prompts in both canonical order and a deterministic shuffled order, using a per-state permutation RNG seeded by `700000 + game_seed * 10000 + action_tick`. Map outputs back to semantic actions. Report disagreement across orders, first-ID, last-ID, and uniform-legal controls.

### Conditional evaluation and stop rules

- Evaluate every policy on the same 20 complete games: board seeds 340000 through 340009, both seats, against `weighted`. Use canonical action order for all arms, greedy decoding, explicit seeded uniform legal fallback for invalid responses, a 400-turn cap, and an 8,000-action cap.
- Count only actual terminal wins. Report wins, losses, timeouts, fallback rate, conditional END_TURN share, and action frequencies. Flag suspicious behavior for inspection; do not call a flag proof of reward hacking.
- Preserve the original success rule: at least 10 percentage points of full held-out teacher-agreement improvement over untuned and no drop in observed actual win rate. Also require at least 10 points of improvement on informative states, separately for canonical and shuffled presentations, and at least 99% valid held-out answers in each presentation. All data, training, and evaluations must finish.
- Report GRPO versus SFT as a separate comparison. Claim a GRPO advantage over SFT only if its informative agreement is higher in both presentations and its observed win count is at least as high. The main pass rule alone cannot establish an advantage over SFT.
- Report paired game-cluster 95% bootstrap intervals for agreement and wins, using 10,000 resamples and seed 611. A passing point-estimate rule does not establish a statistically reliable improvement or non-regression.
- Budget at most 12 measured hours after the reward gate: two hours for data collection, labeling, and replay; two hours per training arm; two hours per evaluation arm. Run sequentially. Stop at any exhausted phase budget, invalid evidence, or non-finite training values. Use owned subprocess deadlines, preserve the latest saved checkpoint and incomplete records, and do not extend budgets to finish a board or round. An interrupted board pair is incomplete, not replaced.
- Exclude implementation and smoke checks from measurement budgets, but restrict smoke checks to separate seeds and at most one training round per arm. They are execution checks, not evidence of learning. Do not inspect test results to revise the experiment.

## Autonomous completion checklist

- [x] Freeze this retry registration and source in `runs/terminal-return-v2` before measurement.
- [x] Run preflight and all 24 replay checks, each in three fresh processes.
- [x] Attempt the fixed reward comparison within two hours; retain every failed or missing unit.
- [x] Recompute the outcome from raw, hashed evidence. A failed or incomplete gate skips all training steps.
- [x] If the gate passes, implement and freeze the conditional experiment, then run it once within its 12-hour budget. Otherwise mark this step skipped with the measured reason.
- [x] Package reproducible inputs and reports, update the existing dashboard examples and concise README findings, and verify the displayed values against raw evidence.
- [x] Run relevant tests and the frontend build. Check the dashboard in a browser if its code or displayed examples change.
- [x] Link the final evidence and state what passed, failed, or remains unmeasured. Leave commits and pushes for explicit approval.

Autonomous completion means a verifiable result and a checked report, including an honest negative or incomplete result. It does not mean a stronger policy is guaranteed, or that the learning exercises have been completed.

## User learning checks

[LEARNING.md](LEARNING.md) contains the exercises. Start with exercises 6 to 8 while the comparison runs: predict a replay failure, calculate discounted returns from both players' perspectives, and explain why repeatable teacher agreement need not improve wins. Then work through the GRPO advantage, clipping, and reference-policy exercises using one real saved group.

For each, write a prediction before running code, calculate a small case by hand, then explain a changed case without notes. The autonomous work will provide checked examples and identify the relevant artifacts. These explanations remain a user-owned completion check.

## Prior verification

The previous milestone passed 85 Python tests and the frontend build. Browser checks covered the stopped report and separate diagnosis, board selection, and labeled synthetic complete, failed, and missing reports. The v1 result remains incomplete; the fixed checker is not evidence that the reward gate has passed.

## Retry outcome

The retry finished in 274.15 seconds, including 26 prerequisite tests and all 72 cross-process replay checks. All 24 comparison states completed. Independent recomputation from per-world values and visits matched the saved report:

| Objective | Non-flat states | Repeatable states |
| --- | ---: | ---: |
| Control | 14/24 | 3/24 |
| Candidate | 22/24 | 5/24 |

The candidate passed the 18-state non-flat criterion and failed the 20-state repeatability criterion. Conditional data collection, SFT, GRPO, and policy evaluation were therefore skipped. No new optimizer updates ran, and the original success rules and failed experiments remain unchanged.

The recollected environments and RNG positions match the original 24 states after sorting only the top-level state and board attribute names. The original v1 report hash is unchanged. The next question would be why candidate rankings change between search seeds at this budget; this plan does not authorize another experiment to answer it.

Final verification: 85 Python tests pass and the frontend builds. Extracting the [retry archive](examples/terminal-return-v2-inputs.zip) and running the documented summary command reproduces every computed report field. A frozen-source replay of state 00 matches 100 transitions in three processes. Browser checks of the bundled report verified the 14/24, 3/24, 22/24, and 5/24 counts, setup/play boards, six repeat columns, 19 failed-state filter entries, and an empty missing-state filter. The [README](README.md#replay-and-reward-comparison), [report](examples/runs/terminal-return-v2/comparison.json), [preview](examples/terminal-return-v2.png), and [worked learning example](LEARNING.md#8-does-a-more-detailed-reward-mean-better-decisions) preserve the result. No commits or pushes were made in this retry.
