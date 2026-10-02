# Next milestone: can we trust the reward?

The project already has hand-written search, self-play, GRPO, and a working dashboard. The next milestone is a reproducible explanation of why a better training score did not mean better play, plus checks that catch this before another expensive run.

The audit checks reward design, controls, and evaluation before further training.

## Scope and limits

Complete steps 1 to 4 below. Success means a working, tested audit and a defensible report, including a failed quality gate. It does not require a stronger policy. Keep the original experiment and its failed success rule unchanged.

- Local CPU/MPS only; no paid services, cloud machines, or new large models.
- At most two hours of new audit compute, one worker, with partial results marked incomplete. Each three-repeat state unit has a five-minute wall-clock limit, capped by the remaining total budget. No new optimizer updates in this goal.
- Routine code changes, tests, local measurements, and dashboard checks are authorized. Ask before each commit. Do not push, publish posts, change Git history, or touch another project's processes.
- Every measurement records its command, source and input hashes, seeds, sample count, elapsed time, and missing results. Never overwrite a run.
- If a check fails, retain the evidence. Do not lower the threshold, change the cohort, or keep trying variants until something passes.

## Ordered work and checks

### 1. Audit the existing evidence

Recompute label ties, action IDs, action types, and the SFT target distribution for every dataset split. Compare teacher agreement with uniform legal choice, first legal ID, last legal ID, and END_TURN when available (first ID otherwise). Separate setup/play and flat/informative states. Join the existing untuned/SFT/GRPO held-out predictions by state ID, with missing or duplicate IDs treated as errors.

For action-order controls, calculate the expected fixed-ID agreement after a uniform menu permutation. This is an analytic control, not a claim that the language model is invariant to menu order. Identify concrete examples where a high-scoring answer provides no strategic evidence.

**Check:** one command creates a hash-linked JSON report from the original files. Independent toy cases verify the arithmetic. The report distinguishes observed collapse, possible explanations, and causes not yet established. It must not call a suspicious episode a proven reward exploit.

### 2. Test teacher reliability and build a quality gate

Reconstruct 24 training states: for each of the first 12 training game seeds, take the earliest sampled setup state and earliest sampled play state. Match saved prompts and legal actions exactly before re-labeling. Missing states or replay mismatches are explicit failures, not replacements. Use three independent teacher searches per state, with seeds `900000 + 10 * panel_index + repeat_index`, preserving the original two-world, 100-simulations-per-world teacher. A prompt/menu match cannot verify hidden-state recovery because the original full states were not saved.

Also test environment correctness independently: executing a deterministic winning legal action must produce an engine-declared win; passing must not be labeled a terminal win solely for leading at a cutoff. Verify that prompt text does not change when inaccessible opponent card identities or deck order change while public counts stay fixed. Record the teacher's privileged resource access separately.

Quality thresholds below are engineering acceptance criteria, not statistical guarantees. The old experiment's aggregate results are already known; this is a retrospective audit, not a new preregistered success claim.

| Check | Requirement before recommending another training experiment |
| --- | --- |
| Integrity | All hashes, split separation, prediction joins, and replay matches pass. |
| Environment and information boundary | All deterministic correctness and prompt-privacy fixtures pass. |
| Usable reward | At least 75% of training states distinguish some legal actions. Report the full distribution, not just this threshold. |
| Teacher repeatability | At least 80% of the 24 panel states have a non-flat reward in all three repeats and at least one maximizing action shared across all three. Flat states count as failures. |
| Shortcut warning | Flag if first-ID or END_TURN control reaches within five agreement points of the trained model on informative held-out states. This is a review flag, not a claim of causality. |
| Completion | All required audit units finish within the budget; otherwise the recommendation is inconclusive. |

**Check:** a machine-readable gate reports pass/fail/incomplete for each requirement and refuses to issue a training recommendation on failure or missing evidence. Tests include deliberately corrupt inputs, all-tied rewards, unstable teacher rankings, and a valid small fixture. A recommendation is not proof of playing strength.

### 3. Make the diagnosis inspectable

Add one reward-quality view to the existing dashboard. Show the gate, denominators, trivial baselines, tie rates, and repeatability. Let a reviewer open flagged states, see their legal actions and values, and compare repeated labels. Reuse the existing board viewer where practical. Keep the current run/episode workflow.

**Check:** Python API tests, TypeScript build, and a manual browser check against both a complete report and missing/incomplete evidence. Bundle a small reproducible audit example. Avoid a redesign, authentication, cloud deployment, or a generic experiment platform.

### 4. Explain the result and make it learnable

Update the README with measured findings, limitations, the reproduction command, and the next decision. Add a short learning workbook anchored to actual functions and audit examples. Each lesson has a prediction, a small hand calculation or code task, and a way to check the answer:

1. State, observation, action, and actual terminal reward: trace one game decision.
2. MCTS: calculate visit counts and backed-up action values in a tiny tree.
3. GRPO: calculate advantages for `[0, 0, 1, 1]` and `[1, 1, 1, 1]`; explain what can still change under KL regularization.
4. Policy updates: explain old policy versus reference policy; predict clipping for positive and negative advantages.
5. Evaluation: explain why teacher agreement, output validity, and actual wins answer different questions; keep paired seats together when estimating uncertainty.

**Check:** another person can follow the commands and trace each public claim to an artifact. Tests pass, the frontend builds, and the original experiment's evidence is unchanged.

Exercise checks: explain a result without notes, modify a small function, and predict a new failure case.

## After this goal

If the reward gate fails, choose one specific repair from the evidence and register its hypothesis, thresholds, and budget before measuring it. Do not start another broad AlphaZero run or add another domain now.

If a corrected teacher passes, the next milestone is one controlled post-training experiment. Before its first update, freeze new game-separated train/dev/test cohorts, SFT and untuned controls, informative-state and uniform-legal baselines, action-order probes, paired actual-win evaluation, checkpoint selection, and compute limits. Preserve the old +10-point agreement/no-observed-win-drop rule as the historical result; never relabel a revised experiment as the original success. Plan for no more than one overnight training budget, with negative or inconclusive results accepted.

Decide whether to continue from the measured results and remaining uncertainties.

## Progress

- [x] 1. Existing-evidence audit
- [x] 2. Reliability panel and quality gate
- [x] 3. Dashboard and bundled example
- [x] 4. Report, learning workbook, and verification
- [ ] Learning exercises
- [x] Commit approval (separate from technical completion)

Completed 2026-10-01. [Measured report](examples/runs/reward-audit-v1/audit.json): 77/192 flat training states; last-ID agreement 71.1% versus GRPO 67.2%. All 24 panel units were attempted in 84 seconds. Sixteen matched the saved prompt/menu; seven of those passed repeatability. Eight mismatches leave the panel incomplete, so further training is not recommended. No new optimizer updates were run.

Verification: 58 Python tests pass, including recomputation from the bundled original inputs and rejection of incomplete, duplicate, or tampered evidence. The frontend builds. Browser checks covered the measured incomplete report, a labeled synthetic completed report, missing evidence, state selection, board rendering, and repeated labels. The first panel attempt exposed a reporter error on missing data; its artifacts are preserved in the input archive. Original experiment files and the core training code were not changed during this goal.

Next decision: first make state recovery testable across processes and preserve full states. Then test one reward repair against a fixed panel before proposing another training experiment. Patrick's first learning check is lesson 1 in [LEARNING.md](LEARNING.md).
