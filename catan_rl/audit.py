"""Retrospective reward checks. A completed audit can reject further training."""

import argparse
from collections import Counter
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time

from catan_rl.stage6 import RULES, digest, read_dataset, score_output, snapshot
from catan_rl.telemetry import RunWriter, write_json


AUDIT_RULES = {
    "version": "reward_audit_v1",
    "purpose": "retrospective quality audit, not a new training success claim",
    "minimum_informative_fraction": .75,
    "minimum_repeatable_fraction": .8,
    "shortcut_margin": .05,
    "panel_games": 12,
    "panel_states": 24,
    "panel_selection": "earliest saved setup and play state from each of the first 12 training seeds",
    "repeats": 3,
    "teacher_seed": "900000 + 10 * panel_index + repeat_index",
    "teacher": RULES["teacher"],
    "max_seconds": 7200,
    "max_unit_seconds": 300,
    "optimizer_updates": 0,
}
FIXTURE_NAMES = {"terminal_win", "pass_is_not_terminal_win", "hidden_cards", "deck_order"}


def fraction(numerator, denominator):
    return numerator / denominator if denominator else None


def controls(row):
    count = len(row["actions"])
    passing = next((i for i, a in enumerate(row["actions"]) if a["type"] == "END_TURN"), 0)
    return {"uniform_legal": len(row["best"]) / count,
            "first_id": int(0 in row["best"]),
            "last_id": int(count - 1 in row["best"]),
            "end_turn_else_first": int(passing in row["best"])}


def summarize(rows):
    count = len(rows)
    informative = sum(len(r["best"]) < len(r["actions"]) for r in rows)
    target_ids = Counter(str(min(r["best"])) for r in rows)
    target_types = Counter(r["actions"][min(r["best"])]["type"] for r in rows)
    eligible = [r for r in rows if any(a["type"] == "END_TURN" for a in r["actions"])]
    chosen_passes = sum(r["actions"][min(r["best"])]["type"] == "END_TURN" for r in eligible)
    baselines = {key: fraction(sum(controls(r)[key] for r in rows), count)
                 for key in ("uniform_legal", "first_id", "last_id", "end_turn_else_first")}
    return {"states": count, "source_games": len({r["game_seed"] for r in rows}),
            "informative_states": informative, "flat_states": count - informative,
            "informative_fraction": fraction(informative, count),
            "sft_target_ids": dict(target_ids), "sft_target_types": dict(target_types),
            "end_turn_eligible": len(eligible), "sft_end_turn_targets": chosen_passes,
            "sft_end_turn_fraction": fraction(chosen_passes, len(eligible)),
            "agreement_baselines": baselines,
            "permuted_fixed_id_expected_agreement": baselines["uniform_legal"]}


def cohorts(rows):
    informative = lambda r: len(r["best"]) < len(r["actions"])
    groups = {"all": rows, "informative": [r for r in rows if informative(r)],
              "flat": [r for r in rows if not informative(r)]}
    for phase in ("setup", "play"):
        groups[phase] = [r for r in rows if r["state"]["phase"] == phase]
        groups[f"{phase}_informative"] = [r for r in groups[phase] if informative(r)]
    return groups


def join_predictions(rows, predictions):
    indexed = {p["id"]: p for p in predictions}
    if len(indexed) != len(predictions) or set(indexed) != {r["id"] for r in rows}:
        raise ValueError("Missing, extra, or duplicate prediction IDs")
    for row in rows:
        p = indexed[row["id"]]
        index, reward = score_output(p["output"], row)
        if (p["game_seed"] != row["game_seed"] or p["action"] != index or
                p["valid"] != (index is not None) or p["agreement"] != (index in row["best"]) or
                p["action_count"] != len(row["actions"]) or p["tied_best"] != len(row["best"]) or
                not math.isclose(p["reward"], reward, abs_tol=1e-8)):
            raise ValueError(f"Prediction does not match source label: {row['id']}")
    return indexed


def existing_evidence(dataset, evaluations):
    rows, manifest = read_dataset(dataset, False)
    hashes = {f"dataset/{split}.jsonl": manifest["dataset_sha256"][split] for split in rows}
    hashes["dataset/manifest.json"] = digest(Path(dataset) / "manifest.json")
    summaries = {split: {name: summarize(group) for name, group in cohorts(items).items()}
                 for split, items in rows.items()}
    models, indexed = {}, {}
    for arm, directory in evaluations.items():
        directory = Path(directory)
        meta = json.loads((directory / "manifest.json").read_text())
        config = meta["config"]
        # Evaluation records its board seed in the generic seed field.
        if (meta["status"] != "completed" or config["smoke"] or
                config["dataset_sha256"] != manifest["dataset_sha256"] or
                any(config.get(k) != v for k, v in RULES.items() if k != "seed")):
            raise ValueError(f"Evaluation does not match the frozen dataset: {arm}")
        for filename in ("manifest.json", "heldout.json"):
            hashes[f"{arm}/{filename}"] = digest(directory / filename)
        indexed[arm] = join_predictions(rows["test"], json.loads((directory / "heldout.json").read_text())["predictions"])
        models[arm] = {}
        for name, group in cohorts(rows["test"]).items():
            selected = [indexed[arm][r["id"]] for r in group]
            models[arm][name] = {"states": len(selected),
                "agreement": fraction(sum(p["agreement"] for p in selected), len(selected)),
                "valid_rate": fraction(sum(p["valid"] for p in selected), len(selected))}
    flags = []
    baselines = summaries["test"]["informative"]["agreement_baselines"]
    for arm in ("sft", "grpo"):
        agreement = models[arm]["informative"]["agreement"]
        for baseline in ("first_id", "end_turn_else_first"):
            value = baselines[baseline]
            if agreement is not None and value is not None and value >= agreement - AUDIT_RULES["shortcut_margin"]:
                flags.append({"arm": arm, "baseline": baseline, "baseline_agreement": value,
                              "model_agreement": agreement,
                              "reason": "Trivial control is within five points or better on informative states; causal review needed."})
    examples = []
    for split, items in rows.items():
        for row in items:
            target = min(row["best"])
            tags = []
            if len(row["best"]) == len(row["actions"]):
                tags.append("flat_reward")
            if row["actions"][target]["type"] == "END_TURN":
                tags.append("sft_pass_target")
            examples.append({k: row[k] for k in ("id", "split", "game_seed", "actions", "values", "visits", "best", "state", "board")}
                            | {"flags": tags, "sft_target": target, "controls": controls(row),
                               "predictions": {arm: p[row["id"]] for arm, p in indexed.items()} if split == "test" else {}})
    return {"splits": summaries, "models": models, "shortcut_flags": flags,
            "input_sha256": hashes, "states": examples,
            "permutation_note": "Expected agreement for a fixed position under a uniform menu permutation equals uniform legal choice. No model permutation inference was run."}


def quality_gate(report):
    integrity = report.get("integrity")
    fixtures = report.get("fixtures")
    panel = report.get("panel", [])
    usable = report["splits"]["train"]["all"]["informative_fraction"]
    computed = [repeatability(p.get("repeats", [])) for p in panel]
    complete_panel = (len(panel) == AUDIT_RULES["panel_states"] and
                      {p.get("index") for p in panel} == set(range(AUDIT_RULES["panel_states"])) and
                      all(p.get("complete") is True and p.get("replay_match") is True and
                          result is not None and p.get("repeatable") is result and
                          [r.get("seed") for r in p.get("repeats", [])] ==
                          [900000 + 10 * p["index"] + i for i in range(AUDIT_RULES["repeats"])]
                          for p, result in zip(panel, computed)))
    repeatable = sum(result is True for result in computed)
    fixture_result = (None if fixtures is None or set(fixtures) != FIXTURE_NAMES
                      else all(v is True for v in fixtures.values()))
    def check(name, value, passed, detail):
        return {"name": name, "status": "incomplete" if passed is None else "pass" if passed else "fail",
                "value": value, "requirement": detail}
    checks = [
        check("integrity", integrity, integrity, "Hashes, source-game splits, prediction joins, and replay matches all pass."),
        check("environment_and_privacy", fixtures, fixture_result,
              "All deterministic outcome and prompt-privacy fixtures pass."),
        check("usable_reward", usable, None if usable is None else usable >= AUDIT_RULES["minimum_informative_fraction"],
              "At least 75% of training states have a non-flat reward."),
        check("repeatability", fraction(repeatable, AUDIT_RULES["panel_states"]) if complete_panel else None,
              None if not complete_panel else repeatable / AUDIT_RULES["panel_states"] >= AUDIT_RULES["minimum_repeatable_fraction"],
              "At least 80% of 24 states are non-flat in all three repeats and share a maximizing action."),
        check("completion", complete_panel and report.get("complete", False),
              True if complete_panel and report.get("complete") else None, "All required units finish within the audit budget."),
    ]
    status = "incomplete" if any(c["status"] == "incomplete" for c in checks) else "fail" if any(c["status"] == "fail" for c in checks) else "pass"
    return {"status": status, "recommend_training": status == "pass", "checks": checks,
            "note": "Operational quality criteria, not proof of playing strength. Shortcut warnings require review."}


def repeatability(repeats):
    if len(repeats) != AUDIT_RULES["repeats"]:
        return None
    values = [r.get("values", []) for r in repeats]
    if (len({len(v) for v in values}) != 1 or len(values[0]) < 2 or
            any(not isinstance(v, (int, float)) or not math.isfinite(v) or not 0 <= v <= 1
                for row in values for v in row)):
        return None
    from catan_rl.stage6 import best_actions
    best = [set(best_actions(r["values"])) for r in repeats]
    return (all(len(b) < len(r["values"]) for b, r in zip(best, repeats))
            and bool(set.intersection(*best)))


def environment_fixtures():
    from catanatron import Color, Game
    from catanatron.models.enums import ActionType
    from catanatron.players.value import ValueFunctionPlayer
    from catanatron.state_functions import player_key
    from catan_rl.llm import prompt_for
    from catan_rl.mcts import is_chance
    game = Game([ValueFunctionPlayer(Color.RED), ValueFunctionPlayer(Color.BLUE)], seed=3)
    for _ in range(8000):
        before = game.copy()
        record = game.play_tick()
        if game.winning_color() is not None:
            break
    else:
        raise RuntimeError("Terminal fixture did not finish")
    winner = before.state.current_color()
    winning = before.copy()
    winning.execute(record.action)
    passing = before.copy()
    action = next(a for a in passing.playable_actions if a.action_type == ActionType.END_TURN)
    passing.execute(action)
    original = prompt_for(before)
    hidden = before.copy()
    other = next(c for c in hidden.state.colors if c != winner)
    key = player_key(hidden.state, other)
    state = hidden.state.player_state
    state[f"{key}_WOOD_IN_HAND"], state[f"{key}_BRICK_IN_HAND"] = (
        state[f"{key}_BRICK_IN_HAND"], state[f"{key}_WOOD_IN_HAND"])
    # Exchange hidden VP cards with knights in the deck, conserving the card pool.
    count = state[f"{key}_VICTORY_POINT_IN_HAND"]
    if not count or hidden.state.development_listdeck.count("KNIGHT") < count:
        raise RuntimeError("Hidden-card fixture no longer exercises card identities")
    for _ in range(count):
        hidden.state.development_listdeck.remove("KNIGHT")
        hidden.state.development_listdeck.append("VICTORY_POINT")
    state[f"{key}_VICTORY_POINT_IN_HAND"] = 0
    state[f"{key}_KNIGHT_IN_HAND"] += count
    state[f"{key}_ACTUAL_VICTORY_POINTS"] -= count
    deck = before.copy()
    deck.state.development_listdeck.reverse()
    return {"terminal_win": not is_chance(record.action) and winning.winning_color() == winner,
            "pass_is_not_terminal_win": before.winning_color() is None and passing.winning_color() is None,
            "hidden_cards": prompt_for(hidden) == original,
            "deck_order": prompt_for(deck) == original}


def panel_row(rows, index):
    seed = RULES["splits"]["train"][0] + index // 2
    phase = ("setup", "play")[index % 2]
    candidates = [r for r in rows if r["game_seed"] == seed and r["state"]["phase"] == phase]
    if not candidates:
        raise ValueError(f"Missing panel state: seed {seed}, phase {phase}")
    return min(candidates, key=lambda r: int(r["id"].rsplit("-", 1)[1]))


def panel_unit(dataset, index, output):
    from transformers import AutoTokenizer
    from catan_rl.llm import MODEL_ID, prompt_for
    from catan_rl.stage6 import REVISION, collect_states, teacher_values
    from catan_rl.telemetry import action_json
    rows, _ = read_dataset(dataset, False)
    row = panel_row(rows["train"], index)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=REVISION, local_files_only=True)
    states, _ = collect_states(row["game_seed"], tokenizer)
    game = next((g for tick, g in states if f"{row['game_seed']}-{tick}" == row["id"]), None)
    matched = (game is not None and prompt_for(game) == row["prompt"] and
               [action_json(a) for a in game.playable_actions] == row["actions"])
    unit = {"index": index, "state_id": row["id"], "replay_match": matched,
            "repeats": [], "complete": False, "repeatable": None}
    write_json(output, unit)
    if not matched:
        raise ValueError(f"Saved state failed deterministic replay: {row['id']}")
    from catan_rl.stage6 import best_actions
    start = time.monotonic()
    for repeat in range(AUDIT_RULES["repeats"]):
        seed = 900000 + 10 * index + repeat
        values, visits = teacher_values(game, seed)
        unit["repeats"].append({"seed": seed, "values": values, "visits": visits, "best": best_actions(values)})
        unit["seconds"] = time.monotonic() - start
        write_json(output, unit)
    unit.update(complete=True, repeatable=repeatability(unit["repeats"]))
    write_json(output, unit)


def run_panel(writer, report, dataset, started):
    units = writer.directory / "units"
    units.mkdir()
    for index in range(AUDIT_RULES["panel_states"]):
        remaining = AUDIT_RULES["max_seconds"] - (time.monotonic() - started)
        if remaining <= 0:
            break
        target = (units / f"{index:02d}.json").resolve()
        command = [sys.executable, "-m", "catan_rl.audit", "--dataset", str(Path(dataset).resolve()),
                   "--worker-index", str(index), "--worker-output", str(target)]
        error = None
        try:
            result = subprocess.run(command, cwd=writer.directory / "source", capture_output=True, text=True,
                                    timeout=min(remaining, AUDIT_RULES["max_unit_seconds"]))
            if result.returncode:
                error = result.stderr[-2000:]
        except subprocess.TimeoutExpired:
            error = "Audit unit exceeded its wall-clock budget"
        unit = json.loads(target.read_text()) if target.exists() else {"index": index, "complete": False, "replay_match": None}
        if error:
            unit.update(complete=False, error=error)
        report["panel"].append(unit)
        report["elapsed_seconds"] = time.monotonic() - started
        report["gate"] = quality_gate(report)
        write_json(writer.directory / "audit.json", report)
        writer.event("audit_unit", **unit)
        print(f"teacher panel {index+1}/24: complete={unit['complete']}, repeatable={unit.get('repeatable')}", flush=True)


def verify_gate(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    if digest(directory / "audit.json") != manifest.get("audit_sha256"):
        raise ValueError("Audit hash mismatch or unfinished report")
    if any(manifest["config"].get(k) != v for k, v in AUDIT_RULES.items()):
        raise ValueError("Audit rules differ")
    report = json.loads((directory / "audit.json").read_text())
    gate = quality_gate(report)
    if report["gate"] != gate:
        raise ValueError("Saved gate differs from evidence")
    return gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset")
    parser.add_argument("--untuned")
    parser.add_argument("--sft")
    parser.add_argument("--grpo")
    parser.add_argument("--out")
    parser.add_argument("--static-only", action="store_true", help="Skip repeat searches; gate remains incomplete")
    parser.add_argument("--check-run", help="Verify a saved gate; exit 0 only for a pass, 2 otherwise")
    parser.add_argument("--worker-index", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.check_run:
        gate = verify_gate(args.check_run)
        print(json.dumps(gate, indent=2))
        raise SystemExit(0 if gate["recommend_training"] else 2)
    import torch
    torch.set_num_threads(1)
    if args.worker_index is not None:
        if not 0 <= args.worker_index < AUDIT_RULES["panel_states"] or not args.worker_output:
            parser.error("Invalid audit worker arguments")
        panel_unit(args.dataset, args.worker_index, args.worker_output)
        return
    if not all((args.dataset, args.out, args.untuned, args.sft, args.grpo)):
        parser.error("--dataset, --out, --untuned, --sft, and --grpo are required")
    writer = RunWriter(args.out, "reward_audit", {**AUDIT_RULES, "command": [sys.executable, *sys.argv]},
                       title="Stage 6 reward quality")
    snapshot(args.out)
    shutil.copy2(Path(__file__).resolve().parents[1] / "PLAN.md", writer.directory / "source" / "PLAN.md")
    start = time.monotonic()
    try:
        report = existing_evidence(args.dataset, {arm: getattr(args, arm) for arm in ("untuned", "sft", "grpo")})
        report.update(complete=False, integrity=None, fixtures=None, panel=[], elapsed_seconds=time.monotonic() - start)
        report["gate"] = quality_gate(report)
        write_json(writer.directory / "audit.json", report)
        if not args.static_only:
            report["fixtures"] = environment_fixtures()
            run_panel(writer, report, args.dataset, start)
            matches = [p.get("replay_match") for p in report["panel"]]
            report["integrity"] = (False if False in matches else True if len(matches) == AUDIT_RULES["panel_states"] and all(matches) else None)
            report["complete"] = (len(report["panel"]) == AUDIT_RULES["panel_states"] and
                                  all(p["complete"] for p in report["panel"]))
        report["elapsed_seconds"] = time.monotonic() - start
        report["gate"] = quality_gate(report)
        write_json(writer.directory / "audit.json", report)
        writer.finish("completed" if report["complete"] else "incomplete", audit_complete=report["complete"], gate=report["gate"]["status"],
                      audit_sha256=digest(writer.directory / "audit.json"))
        print(json.dumps({"gate": report["gate"], "shortcut_flags": report["shortcut_flags"]}, indent=2))
    except BaseException as exc:
        writer.finish("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    main()
