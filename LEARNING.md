# Learn the project by checking its failures

Do one lesson at a time, about 20 minutes each. Predict the answer before running code. Then explain it aloud without notes. Reading the solution is not the completion check.

Start the [dashboard](README.md#dashboard) and open **Stage 6 reward quality**. The [audit report](examples/runs/reward-audit-v1/audit.json) contains the examples below. No training is needed for these exercises.

## 1. What is the agent actually learning?

Read `prompt_for` in [llm.py](catan_rl/llm.py) and `score_output` in [stage6.py](catan_rl/stage6.py). Open state `110001-31` in the dashboard.

**Predict:** All seven legal actions have value zero. What reward does A0 get? What about A1, or an invalid answer? What does supervised training target? Can agreement distinguish passing from building a road here?

**Do:** Trace one action from the text menu through the parser and reward function. Identify what the model can observe and what the teacher knows but the model cannot see in this prompt.

<details><summary>Check your answer</summary>

A0 and A1 both get -1; invalid output gets -2. Both legal answers count as agreement. SFT targets A0 because it chooses the smallest tied ID, which is END_TURN in this state. This supplies a formatting signal and a supervised preference for passing, without evidence that passing is better. It is a possible contributor to the observed collapse, not a causal explanation by itself.

The prompt includes the player's own hand and public information. Teacher search still uses exact opponent resource hands; development-card identities are resampled. A high score on this reward does not mean an actual game win.

</details>

**Teach-back:** Explain the difference between game state, observation, proxy reward, and terminal outcome using this decision.

## 2. What does search estimate?

Read `_search_statistics`, `_select_action`, and `_rollout` in [mcts.py](catan_rl/mcts.py).

**Calculate:** Action A has two chance outcomes with `(visits, accumulated value)` equal to `(3, 2)` and `(1, 0)`. Action B has `(2, 1.5)`. Compute each action's mean value. Which action has more visits? Which has the larger value?

**Do:** Write a ten-line Python calculation that pools these outcomes by visits. Explain why averaging the two outcome means equally gives the wrong result for A. Locate where a cutoff VP lead becomes a rollout win in our code.

<details><summary>Check your answer</summary>

A has value 2/4 = 0.5; B has 1.5/2 = 0.75. A has more visits. The search agent normally selects by visits, whereas the Stage 6 labels use mean values. Chance outcomes need their actual sampling weights. These are noisy search estimates, and our cutoff rule differs from the actual-win evaluator.

</details>

**Teach-back:** Explain selection, expansion, rollout, and backup. Show whose perspective a node's accumulated value uses.

## 3. Why does GRPO need different rewards?

Read `group_advantages` in [grpo.py](catan_rl/grpo.py).

**Calculate:** Find the mean, population standard deviation, and normalized advantages for `[0, 0, 1, 1]`, then `[1, 1, 1, 1]`. Include the code's epsilon.

**Do:** Reimplement this calculation with plain Python lists, then compare it with the PyTorch result. Predict the effect of adding ten to every reward before checking.

<details><summary>Check your answer</summary>

The first group has mean 0.5 and standard deviation 0.5; advantages are approximately `[-0.9998, -0.9998, 0.9998, 0.9998]`. The second gives four zeros. Adding a constant changes neither result.

Equal rewards provide no relative policy-gradient signal. Parameters can still change through KL regularization or optimizer momentum. Equal rewards among legal choices can still teach output validity when invalid answers receive a lower reward.

</details>

**Teach-back:** Explain why higher sampled reward can coexist with no useful strategic learning.

## 4. What keeps a policy update under control?

Read `grpo_loss` and its [independent gradient tests](tests/test_grpo.py).

**Predict:** With probability ratio 1.5 and clip 0.2, calculate the clipped surrogate for advantage +1 and for advantage -1, ignoring KL. Does clipping block both gradients?

**Do:** Run `uv run pytest tests/test_grpo.py -q`. In an interactive Python session, change the test's advantages and predict which action probability will increase before calling backward.

<details><summary>Check your answer</summary>

The maximized surrogate is `min(ratio * advantage, clipped_ratio * advantage)`. It is 1.2 for +1 and -1.5 for -1. The positive branch is clipped flat; the negative branch still penalizes increasing the bad action's probability. The loss negates this surrogate.

The old policy generated this batch and supplies the importance ratio. The reference is the frozen initial model, used to limit drift. They serve different purposes. Padding must contribute no gradient; EOS is scored once.

</details>

**Teach-back:** Explain old policy, reference policy, clipping, and KL without reading the function.

## 5. What would convince you it learned?

Inspect the audit's all-state and informative-state controls, then the actual game results in [README.md](README.md#results-so-far).

**Predict:** Why can last-ID choice beat GRPO on teacher agreement without establishing a better Catan player? Why are two seats on the same board not two independent board samples?

**Do:** Reproduce the static audit using the README commands. Propose one experiment that could separate a formatting improvement from a strategic improvement. State what result would make you reject your explanation before running anything.

<details><summary>Check your answer</summary>

Agreement depends on noisy labels and counts ties as success. Actual wins test a different outcome. The last-ID control has not been evaluated in full games here. Report output validity, informative-state agreement, fallback use, and actual outcomes separately.

Resample paired boards together for uncertainty estimates. Eight panel states failed the saved prompt/menu check, so their teacher reliability is unknown. Even matching prompts cannot verify hidden-state recovery. Missing evidence cannot be treated as success. A new experiment needs saved replayable state, fixed cohorts, and a meaningful reward before more optimizer updates.

</details>

**Teach-back:** Give a two-minute account of the negative result, one plausible cause, one alternative explanation, and the experiment that would distinguish them.

Completion check: explain the five lessons without notes, make one small code change with a prediction, and verify the result independently.
