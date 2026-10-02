import copy

import pytest
from catan_rl.reward_compare import summarize_repeats


def repeats(values):
    return [{"seed": 940000 + j, "values": list(v), "visits": [100, 100],
             "rollouts": {"total": 200, "cutoff": 150, "actions": 900},
             "worlds": [{"visits": [50, 50], "values": list(v)} for _ in range(2)]}
            for j, v in enumerate(values)]


def test_repeatability_uses_semantic_actions_and_all_repeats():
    report = summarize_repeats(repeats([[.1, .9], [.2, .7], [.4, .6]]), ["pass", "build"], 0)
    assert report["nonflat"] and report["repeatable"]
    assert report["shared_maxima"] == ["build"]
    assert report["value_spreads"] == pytest.approx([.8, .5, .2])
    assert report["cutoff_fraction"] == .75
    for values in ([[.1, .9], [.2, .7], [.5, .5]], [[.1, .9], [.8, .2], [.4, .6]]):
        assert not summarize_repeats(repeats(values), ["pass", "build"], 0)["repeatable"]


def test_rejects_missing_or_invalid_search_evidence():
    good = repeats([[.1, .9]] * 3)
    with pytest.raises(ValueError):
        summarize_repeats(good[:2], ["pass", "build"], 0)
    for field, value in (("seed", 0), ("values", [float("nan"), .9]),
                         ("visits", [1, 199]), ("visits", [2, 197]),
                         ("rollouts", {"total": 199, "cutoff": 100})):
        bad = copy.deepcopy(good)
        bad[0][field] = value
        with pytest.raises(ValueError):
            summarize_repeats(bad, ["pass", "build"], 0)


def synthetic_run(directory):
    import json
    from catanatron.models.enums import Action, ActionType
    from catanatron.models.player import Color
    from catan_rl.replay import checksum, encode, action_key
    from catan_rl.reward_compare import RULES
    from catan_rl.telemetry import write_json
    for name in ("units", "states", "collection", "replay", "source"):
        (directory / name).mkdir()
    from catan_rl.reward_compare import file_hash
    sources = {}
    for name in ("catan_rl/replay.py", "catan_rl/mcts.py", "catan_rl/reward_compare.py", "PLAN.md", "uv.lock", "pyproject.toml"):
        source_path = directory / "source" / name
        source_path.parent.mkdir(parents=True, exist_ok=True)
        source_path.write_text("synthetic test fixture")
        sources[name] = file_hash(source_path)
    write_json(directory / "manifest.json", {"config": RULES, "frozen_sha256": sources})
    write_json(directory / "preflight.json", {"returncode": 0})
    actions = [Action(Color.RED, ActionType.BUILD_SETTLEMENT, n) for n in (0, 1)]
    for index in range(24):
        seed, phase = 230000 + index // 2, ["setup", "play"][index % 2]
        record = {"environment": {"seed": seed, "legal_actions": encode(actions),
                  "state": encode({"is_initial_build_phase": phase == "setup"})}}
        record["sha256"] = checksum(record)
        expected = {"initial_sha256": checksum(record["environment"]), "trace": []}
        replay = {"passed": True, "reference": "live_environment", "snapshot_sha256": record["sha256"],
                  "expected_sha256": checksum(expected), "processes": [
                      {"hash_seed": str(i), "matches": True, "trace_sha256": checksum(expected)} for i in range(3)]}
        labels = repeats([[.1, .9]] * 3)
        for j, label in enumerate(labels):
            label["seed"] = 940000 + 10 * index + j
        write_json(directory / "units" / f"{index:02d}.json", {"index": index, "complete": True,
                   "snapshot_sha256": record["sha256"], "replay": replay, "actions": [action_key(a) for a in actions],
                   "arms": {"control": labels, "candidate": labels}})
        write_json(directory / "states" / f"{index:02d}.json", record)
        write_json(directory / "states" / f"{index:02d}.expected.json", expected)
        path = directory / "collection" / f"{seed}.json"
        collection = json.loads(path.read_text()) if path.exists() else {"states": {}}
        collection["states"][phase] = {"state": {}, "board": {}, "tick": 0, "snapshot": record, "expected": expected}
        write_json(path, collection)
    seal(directory)


def test_fixed_thresholds_and_missing_evidence(tmp_path):
    import json
    from catan_rl.reward_compare import aggregate
    from catan_rl.telemetry import write_json
    synthetic_run(tmp_path)
    assert aggregate(tmp_path)["status"] == "pass"
    for index in range(5):
        path = tmp_path / "units" / f"{index:02d}.json"
        row = json.loads(path.read_text())
        for repeat in row["arms"]["candidate"]:
            repeat["values"] = [.5, .5]
            for world in repeat["worlds"]:
                world["values"] = [.5, .5]
        write_json(path, row)
    seal(tmp_path)
    report = aggregate(tmp_path)
    assert report["status"] == "fail"
    assert report["arms"]["candidate"]["repeatable"] == 19
    (tmp_path / "units" / "00.json").unlink()
    seal(tmp_path)
    report = aggregate(tmp_path)
    assert report["status"] == "incomplete"
    assert not report["recommend_training_design"]
    assert report["arms"]["candidate"]["denominator"] == 24


def test_replay_claim_and_world_coverage_must_match_evidence(tmp_path):
    import json
    from catan_rl.reward_compare import aggregate
    from catan_rl.telemetry import write_json
    synthetic_run(tmp_path)
    path = tmp_path / "units" / "00.json"
    original = json.loads(path.read_text())
    bad = copy.deepcopy(original)
    bad["replay"]["processes"][0]["trace_sha256"] = "wrong"
    write_json(path, bad)
    seal(tmp_path)
    with pytest.raises(ValueError, match="Replay prerequisite"):
        aggregate(tmp_path)
    bad = copy.deepcopy(original)
    bad["arms"]["candidate"][0]["worlds"][0]["visits"] = [0, 100]
    write_json(path, bad)
    seal(tmp_path)
    with pytest.raises(ValueError, match="world coverage"):
        aggregate(tmp_path)
    write_json(path, original)
    write_json(tmp_path / "preflight.json", {"returncode": 1})
    seal(tmp_path)
    assert not aggregate(tmp_path)["recommend_training_design"]


def test_timeout_is_recorded_and_process_is_reaped(tmp_path):
    import subprocess
    import sys
    from catan_rl.reward_compare import bounded_process
    with pytest.raises(subprocess.TimeoutExpired):
        bounded_process([sys.executable, "-c", "import time; time.sleep(10)"], tmp_path, .1)


def seal(directory):
    import json
    from catan_rl.reward_compare import file_hash
    from catan_rl.telemetry import write_json
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["evidence_sha256"] = {str(p.relative_to(directory)): file_hash(p)
        for name in ("states", "units", "collection", "replay") for p in (directory / name).glob("*.json")}
    manifest["evidence_sha256"]["preflight.json"] = file_hash(directory / "preflight.json")
    write_json(path, manifest)


def test_missing_hashes_and_changed_files_cannot_pass(tmp_path):
    import json
    from catan_rl.reward_compare import aggregate
    from catan_rl.telemetry import write_json
    synthetic_run(tmp_path)
    path = tmp_path / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["evidence_sha256"].pop("preflight.json")
    write_json(path, manifest)
    with pytest.raises(ValueError, match="evidence hashes"):
        aggregate(tmp_path)
    seal(tmp_path)
    write_json(tmp_path / "preflight.json", {"returncode": 1})
    with pytest.raises(ValueError, match="Evidence hash mismatch"):
        aggregate(tmp_path)


def test_collection_preserves_live_reference_before_restoration():
    from catan_rl.reward_compare import collect
    from catan_rl.replay import checksum, continuation, restore, semantic_state
    result = collect(931)
    assert set(result["states"]) == {"setup", "play"}
    for row in result["states"].values():
        game, rng = restore(row["snapshot"])
        assert checksum(semantic_state(game)) == row["expected"]["initial_sha256"]
        assert continuation(game, rng) == row["expected"]["trace"]


def test_bundled_original_attempt_recomputes_without_search(tmp_path):
    import json
    from pathlib import Path
    import zipfile
    from catan_rl.reward_compare import aggregate, file_hash
    with zipfile.ZipFile(Path(__file__).resolve().parents[1] / "examples/terminal-return-inputs.zip") as archive:
        archive.extractall(tmp_path)
    run = tmp_path / "terminal-return-v1"
    report = json.loads((run / "comparison.json").read_text())
    report.pop("seconds")
    assert aggregate(run) == report
    assert report["status"] == "incomplete"
    assert report["arms"]["candidate"]["measured"] == 0
    diagnosis = json.loads((tmp_path / "diagnosis.json").read_text())
    assert diagnosis["report_sha256"] == file_hash(run / "comparison.json")
    cases = [p for c in diagnosis["cases"] for p in c["processes"]]
    assert len(cases) == 72
    assert all(p["same_fields"] and p["copy_matches"] and not p["before_matches"] for p in cases)
