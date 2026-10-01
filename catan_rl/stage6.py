"""Frozen MCTS labels, LoRA training, and paired evaluation for Stage 6."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import shutil
import signal
import subprocess
import sys
import time

import numpy as np
import torch
from catanatron import Color, Game
from catanatron.models.player import Player
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

from catan_rl.determinize import determinize
from catan_rl.evaluation import play_episode
from catan_rl.grpo import group_advantages, grpo_loss
from catan_rl.llm import (MODEL_ID, completion_mask, encode_prompt, load_policy,
                          parse_action, prompt_for, token_log_probs)
from catan_rl.mcts import MCTS
from catan_rl.randomness import isolated_random
from catan_rl.telemetry import RunWriter, action_json, public_board, public_state, write_json

REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
RULES = {
    "version": "stage6_v1", "model": MODEL_ID, "revision": REVISION,
    "prompt_version": "catan_action_v2_ports", "seed": 611,
    "source": "value versus weighted, alternating colors; 400 turns / 8000 actions",
    "splits": {"train": [110000, 24], "dev": [120000, 8], "test": [130000, 16]},
    "states_per_game": {"setup": 2, "play": 6},
    "sampling": "uniform reservoir per phase, non-forced decisions, no prompt truncation",
    "teacher": "2 determinized dev-card worlds x 100 MCTS simulations; random rollouts; horizon 120",
    "teacher_limit": "Resource hands remain privileged; only hidden dev cards and deck are resampled.",
    "coverage": "Every legal action visited in both worlds; menus above 100 actions excluded from dataset",
    "reward": "2 * pooled root mean value - 1; invalid output -2",
    "agreement": "any maximum-value action within 1e-8; invalid disagrees; forced excluded",
    "sft_target": "smallest ID among tied maximum-value actions",
    "max_prompt_tokens": 2048, "max_new_tokens": 16,
    "lora": {"rank": 8, "alpha": 16, "dropout": 0, "targets": ["q_proj", "k_proj", "v_proj", "o_proj"]},
    "rounds": 200, "prompts_per_round": 4, "group": 4, "updates_per_round": 2,
    "hours_per_training_arm": 8, "learning_rate": 1e-5, "clip": .2, "beta": .02,
    "gradient_clip": 1., "optimizer": "AdamW, weight_decay=0, eps=1e-8",
    "sampling_temperature": 1., "top_p": 1., "top_k": 0,
    "checkpoint": "last completed round; no selection on dev/test",
    "control": "SFT from same initial weights, same sampled prompts and update count; lower token compute",
    "dev_every": 25, "checkpoint_every": 25, "game_seed": 170000, "evaluation_games": 20,
    "game_turn_cap": 400, "game_action_cap": 8000,
    "evaluation_hours_per_arm": 4,
    "game_decode": "greedy; invalid/overlength prompt uses seeded uniform legal fallback, logged",
    "success": "test agreement gain >= .10 and observed actual win rate >= untuned; all evaluations complete",
    "interval": "paired cluster bootstrap of differences, 10000 resamples seed 611; game/board is cluster",
    "budget_stop": "finish current training round or evaluation board pair; incomplete evaluation is inconclusive",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(directory):
    root = Path(__file__).resolve().parents[1]
    target = Path(directory) / "source"
    target.mkdir()
    shutil.copytree(root / "catan_rl", target / "catan_rl", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("README.md", "pyproject.toml", "uv.lock"):
        shutil.copy2(root / name, target / name)


def best_actions(values):
    high = max(values)
    return [i for i, value in enumerate(values) if high - value <= 1e-8]


def score_output(text, row):
    index = parse_action(text, len(row["actions"]))
    return index, -2. if index is None else 2 * row["values"][index] - 1


def teacher_values(game, seed):
    actions = list(game.playable_actions)
    if len(actions) > 100:
        raise ValueError("Two 100-simulation trees cannot guarantee coverage of this menu")
    totals, visits = np.zeros(len(actions)), np.zeros(len(actions), dtype=int)
    rng = random.Random(seed)
    for world_index in range(2):
        world = determinize(game, game.state.current_color(), rng)
        stats = MCTS(horizon=120, seed=seed + world_index).search_statistics(world, 100)
        if any(a not in stats or stats[a]["visits"] < 1 for a in actions):
            raise RuntimeError("Incomplete teacher action coverage")
        for i, action in enumerate(actions):
            visits[i] += stats[action]["visits"]
            totals[i] += stats[action]["value"] * stats[action]["visits"]
    return (totals / visits).tolist(), visits.tolist()


def collect_states(seed, tokenizer):
    players = [ValueFunctionPlayer(Color.RED), WeightedRandomPlayer(Color.BLUE)]
    if seed % 2:
        players = [WeightedRandomPlayer(Color.RED), ValueFunctionPlayer(Color.BLUE)]
    game = Game(players, seed=seed)
    reservoir = {"setup": [], "play": []}
    seen, rejected = Counter(), Counter()
    choose_rng, decision_rng = random.Random(seed + 17), random.Random(seed + 29)
    for tick in range(8000):
        if game.winning_color() is not None or game.state.num_turns >= 400:
            break
        if len(game.playable_actions) > 1:
            phase = "setup" if game.state.is_initial_build_phase else "play"
            try:
                if len(game.playable_actions) > 100:
                    raise ValueError("menu")
                encode_prompt(tokenizer, prompt_for(game), "cpu", RULES["max_prompt_tokens"])
            except ValueError:
                rejected[phase] += 1
            else:
                seen[phase] += 1
                limit = RULES["states_per_game"][phase]
                slot = choose_rng.randrange(seen[phase])
                if len(reservoir[phase]) < limit:
                    reservoir[phase].append((tick, game.copy()))
                elif slot < limit:
                    reservoir[phase][slot] = (tick, game.copy())
        player = game.state.players[game.state.current_player_index]
        with isolated_random(decision_rng):
            action = player.decide(game, game.playable_actions)
        game.execute(action)
    return sorted(reservoir["setup"] + reservoir["play"], key=lambda x: x[0]), dict(rejected)


def build_dataset(directory, smoke=False):
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION)
    writer = RunWriter(directory, "teacher_dataset", {**RULES, "smoke": smoke})
    snapshot(directory)
    hashes, counts = {}, {}
    try:
        for split, (first_seed, games) in RULES["splits"].items():
            if smoke:
                first_seed += 30000
                games = 1
            path = writer.directory / f"{split}.jsonl"
            count = 0
            with path.open("x") as output:
                for number in range(games):
                    seed = first_seed + number
                    states, rejected = collect_states(seed, tokenizer)
                    if smoke:
                        states = states[:2]
                    for tick, game in states:
                        started = time.monotonic()
                        values, visits = teacher_values(game, seed * 10000 + tick)
                        row = {"id": f"{seed}-{tick}", "game_seed": seed, "split": split,
                               "prompt": prompt_for(game), "actions": [action_json(a) for a in game.playable_actions],
                               "values": values, "visits": visits, "best": best_actions(values),
                               "state": public_state(game), "board": public_board(game)}
                        output.write(json.dumps(row, allow_nan=False) + "\n")
                        output.flush()
                        count += 1
                        writer.event("label", split=split, state_id=row["id"], actions=len(values),
                                     minimum_visits=min(visits), tied_best=len(row["best"]),
                                     seconds=time.monotonic() - started)
                    writer.event("collection", split=split, game_seed=seed, states=len(states), rejected=rejected)
                    print(f"{split} game {number + 1}/{games}: {count} labeled states", flush=True)
            hashes[split], counts[split] = digest(path), count
        writer.finish(dataset_sha256=hashes, counts=counts)
    except BaseException as exc:
        writer.finish("failed", error=f"{type(exc).__name__}: {exc}")
        raise


def read_dataset(directory, smoke):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["status"] != "completed" or manifest["config"] != {**RULES, "smoke": smoke}:
        raise ValueError("Dataset is incomplete or its frozen rules differ")
    rows, seeds = {}, {}
    for split in RULES["splits"]:
        path = directory / f"{split}.jsonl"
        if digest(path) != manifest["dataset_sha256"][split]:
            raise ValueError(f"Dataset hash mismatch: {split}")
        rows[split] = [json.loads(line) for line in path.read_text().splitlines()]
        identifiers = [r["id"] for r in rows[split]]
        if len(set(identifiers)) != len(identifiers) or len(rows[split]) != manifest["counts"][split]:
            raise ValueError("Duplicate states or incorrect dataset count")
        for row in rows[split]:
            if (row["split"] != split or len(row["actions"]) < 2 or
                len(row["values"]) != len(row["actions"]) or
                len(row["visits"]) != len(row["actions"]) or min(row["visits"]) < 2 or
                not all(np.isfinite(v) and 0 <= v <= 1 for v in row["values"]) or
                row["best"] != best_actions(row["values"])):
                raise ValueError("Invalid teacher row")
        seeds[split] = {r["game_seed"] for r in rows[split]}
        if not rows[split]:
            raise ValueError(f"Empty {split} split")
    if any(seeds[a] & seeds[b] for a, b in (("train", "dev"), ("train", "test"), ("dev", "test"))):
        raise ValueError("Source game leaked between dataset splits")
    return rows, manifest


def generate(model, tokenizer, prompt, device, group=1, sample=False):
    tokens = encode_prompt(tokenizer, prompt, device, RULES["max_prompt_tokens"])
    options = dict(do_sample=sample, num_return_sequences=group, max_new_tokens=16,
                   eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id,
                   repetition_penalty=1., no_repeat_ngram_size=0)
    if sample:
        options.update(temperature=1., top_p=1., top_k=0)
    else:
        options.update(temperature=None, top_p=None, top_k=None)
    with torch.no_grad():
        sequences = model.generate(**tokens, **options)
    length = tokens["input_ids"].shape[1]
    completion = sequences[:, length:]
    mask = completion_mask(completion, tokenizer.eos_token_id)
    return sequences, length, mask, tokenizer.batch_decode(completion, skip_special_tokens=True)


def offline_evaluation(model, tokenizer, rows, device):
    predictions = []
    for row in rows:
        _, _, _, texts = generate(model, tokenizer, row["prompt"], device)
        index, reward = score_output(texts[0], row)
        predictions.append({"id": row["id"], "game_seed": row["game_seed"], "output": texts[0],
                            "action": index, "valid": index is not None, "reward": reward,
                            "agreement": index in row["best"], "tied_best": len(row["best"]),
                            "action_count": len(row["actions"])})
    count = len(predictions)
    informative = [p for p in predictions if p["tied_best"] < p["action_count"]]
    return {"agreement": sum(p["agreement"] for p in predictions) / count,
            "uniform_legal_agreement": sum(p["tied_best"] / p["action_count"] for p in predictions) / count,
            "valid_rate": sum(p["valid"] for p in predictions) / count,
            "mean_reward": sum(p["reward"] for p in predictions) / count,
            "informative_agreement": sum(p["agreement"] for p in informative) / len(informative) if informative else None,
            "informative_states": len(informative), "states": count, "predictions": predictions}


def checkpoint(model, tokenizer, optimizer, rng, round_index, elapsed, writer):
    target = writer.directory / f"round{round_index:04d}"
    model.save_pretrained(target)
    tokenizer.save_pretrained(target)
    torch.save({"optimizer": optimizer.state_dict(), "round": round_index,
                "elapsed_seconds": elapsed, "python_rng": rng.getstate(),
                "torch_rng": torch.get_rng_state(),
                "mps_rng": torch.mps.get_rng_state() if next(model.parameters()).device.type == "mps" else None},
               target / "training.pt")
    write_json(target / "run.json", {"config": writer.manifest["config"],
                                    "adapter_sha256": digest(target / "adapter_model.safetensors")})
    return target


def load_adapter(model, directory, expected_config):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    directory = Path(directory)
    metadata = json.loads((directory / "run.json").read_text())
    if metadata["config"] != expected_config:
        raise ValueError("Resume must preserve the full experiment configuration")
    path = directory / "adapter_model.safetensors"
    if digest(path) != metadata["adapter_sha256"]:
        raise ValueError("Adapter hash mismatch")
    set_peft_model_state_dict(model, load_file(path))


def train(directory, dataset, arm, device="mps", smoke=False, resume=None):
    torch.set_num_threads(1)
    torch.manual_seed(RULES["seed"])
    rows, data_manifest = read_dataset(dataset, smoke)
    model, tokenizer = load_policy(device, REVISION)
    model.eval()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=RULES["learning_rate"], weight_decay=0, eps=1e-8)
    rng = random.Random(RULES["seed"])
    config = {**RULES, "smoke": smoke, "device": device,
              "dataset_sha256": data_manifest["dataset_sha256"], "arm": arm}
    completed, elapsed_before = 0, 0.
    if resume:
        load_adapter(model, resume, config)
        saved = torch.load(Path(resume) / "training.pt", weights_only=False, map_location="cpu")
        optimizer.load_state_dict(saved["optimizer"])
        rng.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"])
        if device == "mps":
            torch.mps.set_rng_state(saved["mps_rng"])
        completed, elapsed_before = saved["round"], saved["elapsed_seconds"]
    writer = RunWriter(directory, arm, config)
    snapshot(directory)
    started = time.monotonic()
    rounds = 1 if smoke else RULES["rounds"]
    target = None
    stopping = []
    handlers = {sig: signal.getsignal(sig) for sig in (signal.SIGINT, signal.SIGTERM)}
    for sig in handlers:
        signal.signal(sig, lambda signum, frame: stopping.append(signum))
    try:
        for step in range(completed + 1, rounds + 1):
            if stopping or elapsed_before + time.monotonic() - started >= RULES["hours_per_training_arm"] * 3600:
                break
            batch = [rng.choice(rows["train"]) for _ in range(RULES["prompts_per_round"])]
            samples, rewards, selected, flat_groups = [], [], Counter(), 0
            for row in batch:
                if arm == "grpo":
                    seq, length, mask, texts = generate(model, tokenizer, row["prompt"], device, RULES["group"], True)
                    scored = [score_output(text, row) for text in texts]
                    group_rewards = torch.tensor([[s[1] for s in scored]], device=device)
                    advantage = group_advantages(group_rewards)[0]
                    flat_groups += len(set(s[1] for s in scored)) == 1
                    writer.event("rollout", step=step, state_id=row["id"], outputs=texts,
                                 actions=[s[0] for s in scored], rewards=[s[1] for s in scored],
                                 advantages=advantage.tolist())
                    rewards.extend(s[1] for s in scored)
                    for index, _ in scored:
                        selected["INVALID" if index is None else row["actions"][index]["type"]] += 1
                    for j in range(len(texts)):
                        one, one_mask = seq[j:j+1], mask[j:j+1]
                        with torch.no_grad():
                            old = token_log_probs(model, one, length, one_mask).detach()
                            with model.disable_adapter():
                                reference = token_log_probs(model, one, length, one_mask).detach()
                        samples.append((one, length, one_mask, old, reference, advantage[j:j+1]))
                else:
                    tokens = encode_prompt(tokenizer, row["prompt"], device)
                    answer = tokenizer.encode(f"A{min(row['best'])}", add_special_tokens=False) + [tokenizer.eos_token_id]
                    completion = torch.tensor([answer], device=device)
                    seq = torch.cat([tokens["input_ids"], completion], dim=1)
                    samples.append((seq, tokens["input_ids"].shape[1], torch.ones_like(completion).float(), None, None, None))
            losses, kls, clipped = [], [], []
            for _ in range(RULES["updates_per_round"]):
                optimizer.zero_grad(set_to_none=True)
                for seq, length, mask, old, reference, advantage in samples:
                    logp = token_log_probs(model, seq, length, mask)
                    if arm == "grpo":
                        loss, stats = grpo_loss(logp, old, reference, advantage, mask,
                                               clip=RULES["clip"], beta=RULES["beta"])
                        kls.append(stats["kl"])
                        clipped.append(stats["clip_fraction"])
                    else:
                        loss = -(logp * mask).sum() / mask.sum()
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite training loss")
                    (loss / len(samples)).backward()
                    losses.append(loss.item())
                norm = torch.nn.utils.clip_grad_norm_(parameters, RULES["gradient_clip"], error_if_nonfinite=True)
                optimizer.step()
            metrics = {"step": step, "loss": float(np.mean(losses)), "gradient_norm": float(norm),
                       "elapsed_seconds": elapsed_before + time.monotonic() - started}
            if rewards:
                metrics.update(mean_reward=float(np.mean(rewards)), rewards=rewards,
                               valid_rate=1 - selected["INVALID"] / len(rewards), action_counts=dict(selected),
                               flat_group_rate=flat_groups / len(batch),
                               kl=float(np.mean(kls)), clip_fraction=float(np.mean(clipped)))
            if step % RULES["dev_every"] == 0 or step == rounds:
                dev = offline_evaluation(model, tokenizer, rows["dev"], device)
                write_json(writer.directory / f"dev-{step:04d}.json", dev)
                metrics.update(dev_agreement=dev["agreement"], dev_valid_rate=dev["valid_rate"])
            writer.event("metrics", **metrics)
            completed = step
            if step % RULES["checkpoint_every"] == 0 or step == rounds or stopping:
                target = checkpoint(model, tokenizer, optimizer, rng, step, metrics["elapsed_seconds"], writer)
            print(f"{arm} round {step}/{rounds}: loss {metrics['loss']:.4f}, reward {metrics.get('mean_reward')}", flush=True)
        if completed and (target is None or target.name != f"round{completed:04d}"):
            target = checkpoint(model, tokenizer, optimizer, rng, completed,
                                elapsed_before + time.monotonic() - started, writer)
        writer.finish("completed" if completed == rounds else "paused" if stopping else "budget_exhausted",
                      final_checkpoint=str(target) if target else None)
    except BaseException as exc:
        writer.finish("failed", error=f"{type(exc).__name__}: {exc}", last_checkpoint=str(target) if target else None)
        raise
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
    return target


class LanguageAgent(Player):
    def __init__(self, color, model, tokenizer, device, seed):
        super().__init__(color)
        self.model, self.tokenizer, self.device = model, tokenizer, device
        self.rng = random.Random(seed)
        self.last_decision = {}

    def decide(self, game, playable_actions):
        reason = None
        try:
            _, _, _, outputs = generate(self.model, self.tokenizer, prompt_for(game), self.device)
            text = outputs[0]
            index = parse_action(text, len(playable_actions))
            if index is None:
                reason = "invalid_output"
        except ValueError as exc:
            if "No truncation allowed" not in str(exc):
                raise
            text, index, reason = "", None, "prompt_over_limit"
        fallback = index is None
        if fallback:
            index = self.rng.randrange(len(playable_actions))
        self.last_decision = {"text": text, "fallback": fallback, "reason": reason, "action_id": index}
        return playable_actions[index]


def evaluate(directory, dataset, adapter=None, device="mps", smoke=False):
    from peft import set_peft_model_state_dict
    from safetensors.torch import load_file
    torch.set_num_threads(1)
    torch.manual_seed(RULES["seed"])
    rows, data_manifest = read_dataset(dataset, smoke)
    model, tokenizer = load_policy(device, REVISION)
    adapter_hash = None
    if adapter:
        adapter = Path(adapter)
        meta = json.loads((adapter / "run.json").read_text())
        if meta["config"]["dataset_sha256"] != data_manifest["dataset_sha256"] or meta["config"]["smoke"] != smoke:
            raise ValueError("Adapter trained on different data or run type")
        adapter_hash = digest(adapter / "adapter_model.safetensors")
        if adapter_hash != meta["adapter_sha256"]:
            raise ValueError("Adapter hash mismatch")
        set_peft_model_state_dict(model, load_file(adapter / "adapter_model.safetensors"))
    model.eval()
    games = 2 if smoke else RULES["evaluation_games"]
    turn_cap = 4 if smoke else RULES["game_turn_cap"]
    action_cap = 80 if smoke else RULES["game_action_cap"]
    first_seed = RULES["game_seed"] + (10000 if smoke else 0)
    writer = RunWriter(directory, "llm_evaluation", {**RULES, "smoke": smoke,
                         "dataset_sha256": data_manifest["dataset_sha256"],
                         "adapter_sha256": adapter_hash, "evaluation": "actual_outcome_v1",
                         "agent": str(adapter) if adapter else "untuned", "opponent": "weighted",
                         "games": games, "seed": first_seed, "turn_cap": turn_cap,
                         "action_cap": action_cap})
    snapshot(directory)
    started = time.monotonic()
    try:
        offline = offline_evaluation(model, tokenizer, rows["test"], device)
        write_json(writer.directory / "heldout.json", offline)
        writer.event("metrics", step=0, agreement=offline["agreement"], valid_rate=offline["valid_rate"],
                     mean_reward=offline["mean_reward"], informative_agreement=offline["informative_agreement"])
        outcomes = []
        for i in range(games):
            if i % 2 == 0 and time.monotonic() - started >= RULES["evaluation_hours_per_arm"] * 3600:
                break
            seed = first_seed + i // 2
            episode = play_episode("llm", "weighted", seed, i % 2,
                turn_cap, action_cap, f"{i:04d}",
                agent_factory=lambda color: LanguageAgent(color, model, tokenizer, device, seed + 123))
            outputs = [s["policy_output"] for s in episode["steps"] if "policy_output" in s]
            episode["invalid_outputs"] = sum(o["fallback"] for o in outputs)
            episode["policy_decisions"] = len(outputs)
            if episode["invalid_outputs"]:
                episode["flags"].append({"code": "llm_fallback", "reason": f"{episode['invalid_outputs']} invalid or overlength decisions used a uniform legal fallback."})
            writer.episode(episode)
            outcomes.append(episode["outcome"])
            writer.event("metrics", step=i + 1, games=i + 1, wins=outcomes.count("win"),
                         losses=outcomes.count("loss"), timeouts=outcomes.count("timeout"),
                         win_rate=outcomes.count("win") / len(outcomes),
                         timeout_rate=outcomes.count("timeout") / len(outcomes), agreement=offline["agreement"],
                         agreement_win_gap=offline["agreement"] - outcomes.count("win") / len(outcomes))
            print(f"evaluation {i+1}/{games}: {episode['outcome']}, invalid {episode['invalid_outputs']}/{len(outputs)}", flush=True)
        write_json(writer.directory / "result.json", {"agreement": offline["agreement"], "outcomes": outcomes,
                   "win_rate": outcomes.count("win") / len(outcomes) if outcomes else None,
                   "complete": len(outcomes) == games, "smoke": smoke})
        writer.finish("completed" if len(outcomes) == games else "budget_exhausted")
    except BaseException as exc:
        writer.finish("failed", error=f"{type(exc).__name__}: {exc}")
        raise


def paired_delta(differences, clusters):
    keys = sorted(set(clusters))
    totals = np.array([sum(d for d, c in zip(differences, clusters) if c == key) for key in keys])
    counts = np.array([clusters.count(key) for key in keys])
    rng = np.random.default_rng(611)
    indices = rng.integers(0, len(keys), (10000, len(keys)))
    samples = totals[indices].sum(axis=1) / counts[indices].sum(axis=1)
    return {"difference": sum(differences) / len(differences),
            "interval": np.quantile(samples, [.025, .975]).tolist(), "clusters": len(keys),
            "degenerate_bootstrap": bool(np.ptp(samples) == 0)}


def compare_evaluations(baseline, trained):
    base, new = Path(baseline), Path(trained)
    manifests = [json.loads((p / "manifest.json").read_text()) for p in (base, new)]
    for key in ("dataset_sha256", "smoke", "revision", "game_seed", "evaluation_games", "game_turn_cap", "game_action_cap"):
        if manifests[0]["config"][key] != manifests[1]["config"][key]:
            raise ValueError(f"Incomparable evaluations: {key}")
    results = [json.loads((p / "result.json").read_text()) for p in (base, new)]
    predictions = [json.loads((p / "heldout.json").read_text())["predictions"] for p in (base, new)]
    if [p["id"] for p in predictions[0]] != [p["id"] for p in predictions[1]]:
        raise ValueError("Held-out states differ")
    agreement = paired_delta([int(n["agreement"]) - int(b["agreement"])
                              for b, n in zip(*predictions)], [p["game_seed"] for p in predictions[0]])
    complete = all(r["complete"] for r in results)
    wins = None
    if complete:
        if len(results[0]["outcomes"]) != len(results[1]["outcomes"]):
            raise ValueError("Game cohorts differ")
        wins = paired_delta([int(n == "win") - int(b == "win")
                             for b, n in zip(results[0]["outcomes"], results[1]["outcomes"])],
                            [i // 2 for i in range(len(results[0]["outcomes"]))])
    return {"agreement": agreement, "wins": wins, "complete": complete,
            "outcome": "inconclusive" if not complete else
                       "passed" if agreement["difference"] >= .10 and wins["difference"] >= 0 else "failed",
            "smoke": results[0]["smoke"],
            "caveat": "Point-estimate stop rule only; equal wins do not establish non-regression. Degenerate bootstrap intervals do not establish certainty."}


def experiment(directory, device="mps", smoke=False):
    directory = Path(directory)
    writer = RunWriter(directory, "stage6_experiment", {**RULES, "device": device, "smoke": smoke})
    snapshot(directory)
    def run(command, suffix, *options):
        target = directory.with_name(f"{directory.name}-{suffix}")
        args = [sys.executable, "-u", "-m", "catan_rl.stage6", command,
                "--out", str(target), "--device", device, *map(str, options)]
        if smoke:
            args.append("--smoke")
        writer.event("phase", command=command, run=target.name)
        subprocess.run(args, check=True)
        return target
    try:
        dataset = run("dataset", "data")
        baseline = run("evaluate", "untuned", "--dataset", dataset)
        comparisons = {}
        for arm in ("sft", "grpo"):
            trained = run(arm, arm, "--dataset", dataset)
            manifest = json.loads((trained / "manifest.json").read_text())
            if manifest["status"] not in ("completed", "budget_exhausted") or not manifest.get("final_checkpoint"):
                raise RuntimeError(f"{arm} did not finish a training checkpoint")
            evaluation = run("evaluate", f"{arm}-eval", "--dataset", dataset,
                             "--adapter", manifest["final_checkpoint"])
            comparisons[arm] = compare_evaluations(baseline, evaluation)
            write_json(directory / "comparison.json", comparisons)
            writer.event("comparison", arm=arm, **comparisons[arm])
        writer.finish(outcome=comparisons["grpo"]["outcome"])
    except BaseException as exc:
        writer.finish("failed", error=f"{type(exc).__name__}: {exc}")
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["experiment", "dataset", "grpo", "sft", "evaluate"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--dataset")
    parser.add_argument("--adapter")
    parser.add_argument("--resume", help="Full-state training checkpoint; requires a fresh --out")
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--smoke", action="store_true", help="Disjoint tiny engineering run, excluded from performance claims")
    args = parser.parse_args()
    if args.command == "experiment":
        experiment(args.out, args.device, args.smoke)
    elif args.command == "dataset":
        build_dataset(args.out, args.smoke)
    elif not args.dataset:
        parser.error("--dataset is required")
    elif args.command == "evaluate":
        evaluate(args.out, args.dataset, args.adapter, args.device, args.smoke)
    else:
        train(args.out, args.dataset, args.command, args.device, args.smoke, args.resume)


if __name__ == "__main__":
    main()
