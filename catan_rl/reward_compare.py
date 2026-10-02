"""Bounded comparison of cutoff-leader and discounted terminal rewards."""

import argparse
from collections import Counter
import json
import math
import os
from pathlib import Path
import random
import signal
import shutil
import subprocess
import sys
import time

from catanatron.models.player import Color
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer
from catan_rl.determinize import determinize
from catan_rl.mcts import MCTS
from catan_rl.randomness import isolated_random
from catan_rl.replay import (CanonicalGame, action_key, capture, checksum, compatibility,
                             continuation, decode, dumps, restore, semantic_state, verify_replay)
from catan_rl.telemetry import RunWriter, public_board, public_state, write_json

RULES = {
    "schema": "terminal_return_comparison_v1", "game_seeds": list(range(230000, 230012)),
    "phases": ["setup", "play"], "sample_seed_offset": 17, "decision_seed_offset": 29,
    "max_turns": 400, "max_actions": 8000, "max_menu": 100,
    "worlds": 2, "simulations": 100, "horizon": 120, "rollout_action_cap": 1000,
    "repeats": 3, "search_seed": 940000, "discount": .995, "tolerance": 1e-8,
    "minimum_nonflat": 18, "minimum_repeatable": 20, "panel_states": 24,
    "max_seconds": 7200, "max_unit_seconds": 300, "replay_steps": 100,
    "hash_seeds": ["0", "1", "2"], "resource_hands": "privileged",
    "arms": ["control", "candidate"],
}


def file_hash(path):
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def collect(seed):
    players = [ValueFunctionPlayer(Color.RED), WeightedRandomPlayer(Color.BLUE)]
    if seed % 2:
        players = [WeightedRandomPlayer(Color.RED), ValueFunctionPlayer(Color.BLUE)]
    engine_rng = random.Random(seed)
    with isolated_random(engine_rng):
        game = CanonicalGame(players, seed=seed)
    sample_rng, decision_rng = random.Random(seed + 17), random.Random(seed + 29)
    seen, rejected, selected = Counter(), Counter(), {}
    for tick in range(RULES["max_actions"]):
        if game.winning_color() is not None or game.state.num_turns >= RULES["max_turns"]:
            break
        phase = "setup" if game.state.is_initial_build_phase else "play"
        if len(game.playable_actions) > RULES["max_menu"]:
            rejected[phase] += 1
        elif len(game.playable_actions) > 1:
            seen[phase] += 1
            if sample_rng.randrange(seen[phase]) == 0:
                selected[phase] = {"tick": tick, "snapshot": capture(game, engine_rng),
                                   "state": public_state(game), "board": public_board(game),
                                   "live_game": game.copy(), "live_rng": engine_rng.getstate()}
        with isolated_random(decision_rng):
            action = game.state.current_player().decide(game, game.playable_actions)
        with isolated_random(engine_rng):
            game.execute(action)
    for row in selected.values():
        live_game, live_rng = row.pop("live_game"), random.Random()
        live_rng.setstate(row.pop("live_rng"))
        row["expected"] = {"initial_sha256": checksum(semantic_state(live_game)),
                           "trace": continuation(live_game, live_rng, RULES["replay_steps"])}
    return {"seed": seed, "states": selected, "eligible": dict(seen), "rejected": dict(rejected),
            "actions": len(game.state.action_records), "turns": game.state.num_turns,
            "winner": game.winning_color().value if game.winning_color() else None}


def label(game, seed, arm):
    actions = game.playable_actions
    totals, visits, counts, worlds = [0.] * len(actions), [0] * len(actions), Counter(), []
    world_rng = random.Random(seed)
    for world_index in range(RULES["worlds"]):
        world = determinize(game, game.state.current_color(), world_rng)
        search = MCTS(seed=seed + world_index, horizon=RULES["horizon"],
                      rollout_action_cap=RULES["rollout_action_cap"],
                      terminal_returns=arm == "candidate",
                      discount=RULES["discount"] if arm == "candidate" else 1.)
        stats = search.search_statistics(world, RULES["simulations"])
        if any(a not in stats or stats[a]["visits"] < 1 for a in actions):
            raise ValueError("Incomplete action coverage")
        for i, action in enumerate(actions):
            visits[i] += stats[action]["visits"]
            totals[i] += stats[action]["visits"] * stats[action]["value"]
        counts.update(search.rollout_counts)
        worlds.append({"visits": [stats[a]["visits"] for a in actions],
                       "values": [stats[a]["value"] for a in actions]})
    return {"seed": seed, "values": [v / n for v, n in zip(totals, visits)],
            "visits": visits, "rollouts": dict(counts), "worlds": worlds}


def compare_state(record, index, partial=None):
    game, _ = restore(record)
    arms = {}
    result = {"index": index, "snapshot_sha256": record["sha256"],
              "actions": [action_key(a) for a in game.playable_actions], "arms": arms}
    for arm in RULES["arms"]:
        arms[arm] = []
        for repeat in range(RULES["repeats"]):
            arms[arm].append(label(game, RULES["search_seed"] + 10 * index + repeat, arm))
            if partial:
                write_json(partial, result)
    return result


def summarize_repeats(repeats, actions, index):
    if len(repeats) != RULES["repeats"] or len(set(actions)) != len(actions) or len(actions) < 2:
        raise ValueError("Incomplete repeats or action menu")
    maxima, spreads, cutoff, total = [], [], 0, 0
    for j, repeat in enumerate(repeats):
        values, visits = repeat["values"], repeat["visits"]
        if repeat["seed"] != RULES["search_seed"] + 10 * index + j:
            raise ValueError("Repeat seed differs from protocol")
        if len(values) != len(actions) or len(visits) != len(actions):
            raise ValueError("Action/value alignment mismatch")
        if any(not math.isfinite(v) or not 0 <= v <= 1 for v in values):
            raise ValueError("Invalid values")
        if any(type(n) is not int or n < RULES["worlds"] for n in visits) or sum(visits) != 200:
            raise ValueError("Incomplete visits")
        worlds = repeat["worlds"]
        if len(worlds) != RULES["worlds"]:
            raise ValueError("Incomplete worlds")
        for world in worlds:
            if (len(world["visits"]) != len(actions) or len(world["values"]) != len(actions)
                    or any(type(n) is not int or n < 1 for n in world["visits"])
                    or sum(world["visits"]) != RULES["simulations"]
                    or any(not math.isfinite(v) or not 0 <= v <= 1 for v in world["values"])):
                raise ValueError("Incomplete world coverage")
        for i in range(len(actions)):
            pooled_visits = sum(w["visits"][i] for w in worlds)
            pooled_value = sum(w["visits"][i] * w["values"][i] for w in worlds) / pooled_visits
            if visits[i] != pooled_visits or not math.isclose(values[i], pooled_value, abs_tol=1e-12):
                raise ValueError("Pooled values differ from world evidence")
        counts = repeat["rollouts"]
        if counts["total"] != 200 or not 0 <= counts["cutoff"] <= 200:
            raise ValueError("Invalid rollout counters")
        cutoff += counts["cutoff"]
        total += counts["total"]
        spreads.append(max(values) - min(values))
        maxima.append({actions[i] for i, value in enumerate(values) if max(values) - value <= RULES["tolerance"]})
    nonflat = all(spread > RULES["tolerance"] for spread in spreads)
    return {"nonflat": nonflat, "repeatable": nonflat and bool(set.intersection(*maxima)),
            "value_spreads": spreads, "tied_maxima": [len(m) for m in maxima],
            "shared_maxima": sorted(set.intersection(*maxima)), "cutoff_fraction": cutoff / total}


def aggregate(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if manifest["config"] != RULES:
        raise ValueError("Comparison protocol differs")
    preflight = json.loads((directory / "preflight.json").read_text())
    prerequisites = preflight["returncode"] == 0
    required_source = {"catan_rl/replay.py", "catan_rl/mcts.py", "catan_rl/reward_compare.py", "PLAN.md", "uv.lock", "pyproject.toml"}
    if not required_source <= manifest.get("frozen_sha256", {}).keys():
        raise ValueError("Missing frozen source hashes")
    evidence_paths = {str(p.relative_to(directory)) for name in ("states", "units", "collection", "replay")
                      for p in (directory / name).glob("*.json")} | {"preflight.json"}
    if evidence_paths != manifest.get("evidence_sha256", {}).keys():
        raise ValueError("Missing or unexpected evidence hashes")
    for relative, expected_hash in manifest["frozen_sha256"].items():
        if file_hash(directory / "source" / relative) != expected_hash:
            raise ValueError("Frozen source hash mismatch")
    for relative, expected_hash in manifest.get("evidence_sha256", {}).items():
        if file_hash(directory / relative) != expected_hash:
            raise ValueError("Evidence hash mismatch")
    states = []
    for index in range(RULES["panel_states"]):
        seed, phase = RULES["game_seeds"][index // 2], RULES["phases"][index % 2]
        collection_path = directory / "collection" / f"{seed}.json"
        selected = (json.loads(collection_path.read_text())["states"].get(phase)
                    if collection_path.exists() else None)
        path = directory / "units" / f"{index:02d}.json"
        if not path.exists():
            row = {"index": index, "complete": False, "error": "missing unit"}
        else:
            row = json.loads(path.read_text())
        if row["index"] != index:
            raise ValueError("Wrong state index")
        row.update(seed=seed, phase=phase)
        if selected:
            row.update(state=selected["state"], board=selected["board"], tick=selected["tick"])
        if row.get("complete"):
            snapshot_path = directory / "states" / f"{index:02d}.json"
            record = json.loads(snapshot_path.read_text())
            payload = {k: v for k, v in record.items() if k != "sha256"}
            if checksum(payload) != record["sha256"] or row["snapshot_sha256"] != record["sha256"]:
                raise ValueError("Snapshot evidence mismatch")
            expected = json.loads((directory / "states" / f"{index:02d}.expected.json").read_text())
            if (selected is None or selected["snapshot"] != record or selected["expected"] != expected
                    or expected["initial_sha256"] != checksum(record["environment"])
                    or decode(record["environment"]["state"])["is_initial_build_phase"] != (phase == "setup")):
                raise ValueError("Collection or phase evidence mismatch")
            replay = row["replay"]
            if (replay["snapshot_sha256"] != record["sha256"] or replay["reference"] != "live_environment"
                    or replay["expected_sha256"] != checksum(expected) or not replay["passed"]
                    or [p["hash_seed"] for p in replay["processes"]] != RULES["hash_seeds"]
                    or any(not p["matches"] or p["trace_sha256"] != checksum(expected) for p in replay["processes"])):
                raise ValueError("Replay prerequisite failed")
            if (record["environment"]["seed"] != RULES["game_seeds"][index // 2]
                    or row["actions"] != [action_key(a) for a in decode(record["environment"]["legal_actions"])]):
                raise ValueError("Panel identity or action alignment mismatch")
            row["summaries"] = {arm: summarize_repeats(row["arms"][arm], row["actions"], index)
                                for arm in RULES["arms"]}
        states.append(row)
    complete = prerequisites and all(row.get("complete") for row in states)
    arms = {}
    for arm in RULES["arms"]:
        available = [row["summaries"][arm] for row in states if row.get("complete")]
        arms[arm] = {"measured": len(available), "denominator": RULES["panel_states"],
                     "nonflat": sum(r["nonflat"] for r in available),
                     "repeatable": sum(r["repeatable"] for r in available)}
    candidate = arms["candidate"]
    passed = complete and candidate["nonflat"] >= RULES["minimum_nonflat"] and candidate["repeatable"] >= RULES["minimum_repeatable"]
    return {"rules": RULES, "complete": complete, "prerequisites": prerequisites,
            "status": "incomplete" if not complete else "pass" if passed else "fail",
            "recommend_training_design": passed, "arms": arms, "states": states,
            "note": "Diagnostic reward signal and repeatability only; not evidence of correct rankings or stronger play."}


def bounded_process(command, cwd, timeout, hash_seed="0"):
    process = subprocess.Popen(command, cwd=cwd, start_new_session=True,
                               env={**os.environ, "PYTHONHASHSEED": hash_seed, "PYTHONPATH": ""},
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        # A replay worker has children; terminate its entire owned process group.
        os.killpg(process.pid, signal.SIGKILL)
        process.communicate()
        raise
    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def worker(command, output, cwd, timeout):
    started = time.monotonic()
    full_command = [sys.executable, "-m", "catan_rl.reward_compare", *command]
    try:
        result = bounded_process(full_command, cwd, timeout)
    except subprocess.TimeoutExpired:
        write_json(output.with_suffix(".process.json"), {"command": full_command,
                   "seconds": time.monotonic() - started, "returncode": None, "error": "timeout"})
        raise
    write_json(output.with_suffix(".process.json"), {"command": command, "seconds": time.monotonic() - started,
               "returncode": result.returncode, "stderr": result.stderr[-8000:]})
    if result.returncode:
        raise RuntimeError(f"Worker failed: {result.stderr[-2000:]}")
    write_json(output, json.loads(result.stdout))


def run(directory):
    directory = Path(directory).resolve()
    writer = RunWriter(directory, "terminal_return_comparison", RULES)
    root = Path(__file__).resolve().parents[1]
    source = directory / "source"
    source.mkdir()
    for name in ("catan_rl", "tests"):
        shutil.copytree(root / name, source / name, ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("PLAN.md", "README.md", "pyproject.toml", "uv.lock"):
        shutil.copy2(root / name, source / name)
    (source / "examples").mkdir()
    shutil.copy2(root / "examples/terminal-return-inputs.zip", source / "examples/terminal-return-inputs.zip")
    for name in ("states", "units", "collection", "replay"):
        (directory / name).mkdir()
    writer.manifest.update(compatibility=compatibility(), command=[sys.executable, "-m", "catan_rl.reward_compare", "run", "--out", str(directory)],
                           frozen_sha256={str(p.relative_to(source)): file_hash(p) for p in sorted(source.rglob("*")) if p.is_file()})
    write_json(directory / "manifest.json", writer.manifest)
    started = time.monotonic()
    deadline = started + RULES["max_seconds"]
    replay_failed = False
    try:
        preflight = bounded_process([sys.executable, "-m", "pytest", "tests/test_replay.py", "tests/test_terminal_returns.py",
                                     "tests/test_reward_compare.py", "-q"], source, min(300, deadline - time.monotonic()))
        write_json(directory / "preflight.json", {"returncode": preflight.returncode,
                   "stdout": preflight.stdout, "stderr": preflight.stderr})
        if preflight.returncode:
            raise ValueError("Replay or reward correctness prerequisite failed")
        for number, seed in enumerate(RULES["game_seeds"]):
            path = directory / "collection" / f"{seed}.json"
            worker(["collect", "--seed", str(seed)], path, source, min(300, max(.01, deadline - time.monotonic())))
            collection = json.loads(path.read_text())
            for offset, phase in enumerate(RULES["phases"]):
                index = 2 * number + offset
                selected = collection["states"].get(phase)
                if selected is None:
                    replay_failed = True
                    write_json(directory / "units" / f"{index:02d}.json", {"index": index, "complete": False, "error": "missing phase"})
                    continue
                record = selected["snapshot"]
                state_path = directory / "states" / f"{index:02d}.json"
                write_json(state_path, record)
                expected_path = directory / "states" / f"{index:02d}.expected.json"
                write_json(expected_path, selected["expected"])
                replay_path = directory / "replay" / f"{index:02d}.json"
                worker(["replay", "--state", str(state_path), "--expected", str(expected_path)], replay_path, source,
                       min(300, max(.01, deadline - time.monotonic())))
                replay = json.loads(replay_path.read_text())
                if not replay["passed"]:
                    replay_failed = True
                writer.event("replay", index=index, passed=replay["passed"])
        if replay_failed:
            raise ValueError("Panel state recovery prerequisite failed")
        for index in range(RULES["panel_states"]):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            path = directory / "units" / f"{index:02d}.json"
            try:
                worker(["unit", "--state", str(directory / "states" / f"{index:02d}.json"), "--index", str(index),
                        "--out", str(path.with_suffix(".partial.json"))],
                       path, source, min(RULES["max_unit_seconds"], remaining))
                row = json.loads(path.read_text())
                row.update(complete=True, replay=json.loads((directory / "replay" / f"{index:02d}.json").read_text()))
                write_json(path, row)
                writer.event("comparison", index=index, complete=True)
            except (RuntimeError, subprocess.TimeoutExpired) as error:
                write_json(path, {"index": index, "complete": False, "error": str(error)})
                writer.event("comparison", index=index, complete=False)
    except (ValueError, RuntimeError, subprocess.TimeoutExpired) as error:
        writer.event("prerequisite_failure", error=str(error))
        if not (directory / "preflight.json").exists():
            write_json(directory / "preflight.json", {"returncode": None, "error": str(error)})
    writer.manifest["evidence_sha256"] = {str(p.relative_to(directory)): file_hash(p)
        for name in ("states", "units", "collection", "replay") for p in sorted((directory / name).glob("*.json"))}
    writer.manifest["evidence_sha256"]["preflight.json"] = file_hash(directory / "preflight.json")
    write_json(directory / "manifest.json", writer.manifest)
    report = aggregate(directory)
    report["seconds"] = time.monotonic() - started
    write_json(directory / "comparison.json", report)
    writer.finish(report["status"], comparison_sha256=file_hash(directory / "comparison.json"))
    print(dumps({"status": report["status"], "arms": report["arms"], "seconds": report["seconds"]}))


def diagnose(directory):
    directory = Path(directory).resolve()
    aggregate(directory)
    probe = '''import json, sys
from pathlib import Path
from catan_rl.replay import restore, semantic_state, checksum, continuation
record = json.loads(Path(sys.argv[1]).read_text())
expected = json.loads(Path(sys.argv[2]).read_text())
game, rng = restore(record)
before, after = semantic_state(game), semantic_state(game.copy())
fields_equal = dict(before['state']['mapping']) == dict(after['state']['mapping'])
other_equal = {k:v for k,v in before.items() if k != 'state'} == {k:v for k,v in after.items() if k != 'state'}
actual = {'initial_sha256': checksum(after), 'trace': continuation(game.copy(), rng)}
print(json.dumps({'same_fields': fields_equal and other_equal,
                  'before_matches': checksum(before) == expected['initial_sha256'],
                  'copy_matches': actual == expected, 'trace_sha256': checksum(actual)}))'''
    cases = []
    started = time.monotonic()
    for index in range(RULES["panel_states"]):
        results = []
        for hash_seed in RULES["hash_seeds"]:
            process = bounded_process([sys.executable, "-c", probe,
                str(directory / "states" / f"{index:02d}.json"),
                str(directory / "states" / f"{index:02d}.expected.json")], directory / "source", 45, hash_seed)
            if process.returncode:
                raise RuntimeError(process.stderr)
            results.append({"hash_seed": hash_seed, **json.loads(process.stdout)})
        cases.append({"index": index, "processes": results})
    return {"source_run": directory.name, "report_sha256": file_hash(directory / "comparison.json"),
            "probe": probe, "cases": cases, "seconds": time.monotonic() - started,
            "note": "Fingerprint diagnosis with the frozen serializer, not a new reward comparison. The original run remains incomplete."}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["run", "collect", "replay", "unit", "summarize", "diagnose"])
    parser.add_argument("--out", type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--index", type=int)
    parser.add_argument("--expected", type=Path)
    args = parser.parse_args()
    if args.command == "run":
        run(args.out)
    elif args.command == "collect":
        print(dumps(collect(args.seed)))
    elif args.command == "summarize":
        print(dumps(aggregate(args.out)))
    elif args.command == "diagnose":
        print(dumps(diagnose(args.out)))
    else:
        record = json.loads(args.state.read_text())
        result = (verify_replay(record, expected=json.loads(args.expected.read_text()))
                  if args.command == "replay" else compare_state(record, args.index, partial=args.out))
        print(dumps(result))


if __name__ == "__main__":
    main()
