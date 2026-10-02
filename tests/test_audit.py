import copy
import json

import pytest

from catan_rl.audit import (FIXTURE_NAMES, controls, environment_fixtures, join_predictions,
                            panel_row, quality_gate, repeatability, summarize, verify_gate)


def row(identifier="one", values=None, best=None):
    return {"id": identifier, "game_seed": 1, "values": values or [.2, .8, .8],
            "actions": [{"type": "END_TURN"}, {"type": "BUILD_ROAD"}, {"type": "BUILD_CITY"}],
            "best": [1, 2] if best is None else best}


def test_control_arithmetic_and_smallest_tied_target():
    result = summarize([row(), row("two", [1, 1, 1], [0, 1, 2])])
    assert result["informative_fraction"] == .5
    assert result["agreement_baselines"] == pytest.approx({
        "uniform_legal": 5 / 6, "first_id": .5, "last_id": 1, "end_turn_else_first": .5})
    assert result["sft_target_ids"] == {"1": 1, "0": 1}
    assert result["sft_end_turn_fraction"] == .5
    assert result["permuted_fixed_id_expected_agreement"] == pytest.approx(5 / 6)
    assert summarize([])["informative_fraction"] is None


def test_end_turn_baseline_uses_action_meaning_not_position():
    item = row()
    item["actions"] = [{"type": "BUILD_ROAD"}, {"type": "BUILD_CITY"}, {"type": "END_TURN"}]
    assert controls(item)["end_turn_else_first"] == 1
    assert controls(item)["first_id"] == 0


def test_prediction_join_checks_semantics_not_saved_summary():
    prediction = {"id": "one", "game_seed": 1, "output": "A1", "action": 1,
                  "valid": True, "reward": .6, "agreement": True, "action_count": 3, "tied_best": 2}
    assert join_predictions([row()], [prediction])["one"] == prediction
    for bad in ([], [prediction, prediction], [{**prediction, "id": "missing"}],
                [{**prediction, "agreement": False}], [{**prediction, "output": "A0"}]):
        with pytest.raises(ValueError):
            join_predictions([row()], bad)


def test_gate_cannot_recommend_training_with_missing_or_failed_evidence():
    report = {"splits": {"train": {"all": {"informative_fraction": .9}}},
              "integrity": True, "fixtures": dict.fromkeys(FIXTURE_NAMES, True),
              "panel": [{"index": i, "complete": True, "repeatable": True, "replay_match": True,
                         "repeats": [{"seed": 900000 + 10 * i + j, "values": [0., 1.]} for j in range(3)]}
                        for i in range(24)], "complete": True}
    assert quality_gate(report)["recommend_training"]
    for field in ("integrity", "fixtures", "panel", "complete"):
        bad = copy.deepcopy(report)
        bad[field] = [] if field == "panel" else None
        assert not quality_gate(bad)["recommend_training"]
    bad = copy.deepcopy(report)
    bad["fixtures"] = {}
    assert not quality_gate(bad)["recommend_training"]
    bad = copy.deepcopy(report)
    bad["panel"][0] = {"complete": False, "repeatable": None}
    assert quality_gate(bad)["status"] == "incomplete"
    bad = copy.deepcopy(report)
    bad["splits"]["train"]["all"]["informative_fraction"] = 0
    assert quality_gate(bad)["status"] == "fail"
    bad = copy.deepcopy(report)
    for item in bad["panel"][:5]:
        item["repeatable"] = False
        item["repeats"][0]["values"] = [1., 0.]
    assert quality_gate(bad)["status"] == "fail"
    bad = copy.deepcopy(report)
    bad["panel"][1] = copy.deepcopy(bad["panel"][0])
    assert not quality_gate(bad)["recommend_training"]


def test_repeatability_requires_signal_in_every_repeat_and_common_action():
    assert repeatability([{"values": [.2, .8, .8]}, {"values": [.1, .9, .7]}, {"values": [.3, .9, .9]}])
    assert not repeatability([{"values": [1, 1]}, {"values": [.2, .8]}, {"values": [.2, .8]}])
    assert not repeatability([{"values": [1, 0, 0]}, {"values": [0, 1, 0]}, {"values": [0, 0, 1]}])
    assert repeatability([]) is None
    assert repeatability([{"values": [float("nan"), 1]}] * 3) is None


def test_panel_selection_does_not_replace_missing_states():
    rows = [{"id": f"110000-{i}", "game_seed": 110000, "state": {"phase": phase}}
            for i, phase in [(12, "play"), (0, "setup"), (7, "play")]]
    assert panel_row(rows, 0)["id"] == "110000-0"
    assert panel_row(rows, 1)["id"] == "110000-7"
    with pytest.raises(ValueError, match="Missing panel state"):
        panel_row(rows, 2)


def test_environment_fixtures_use_real_terminal_outcome_and_hidden_identities():
    result = environment_fixtures()
    assert set(result) == FIXTURE_NAMES
    assert all(result.values())


def test_saved_gate_verification_refuses_tampering(tmp_path):
    from catan_rl.audit import AUDIT_RULES
    from catan_rl.stage6 import digest
    from catan_rl.telemetry import RunWriter, write_json
    writer = RunWriter(tmp_path / "audit", "reward_audit", AUDIT_RULES)
    report = {"splits": {"train": {"all": {"informative_fraction": 0}}}}
    report["gate"] = quality_gate(report)
    path = writer.directory / "audit.json"
    write_json(path, report)
    writer.finish(audit_sha256=digest(path))
    assert not verify_gate(writer.directory)["recommend_training"]
    report["gate"]["recommend_training"] = True
    write_json(path, report)
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_gate(writer.directory)
    meta = json.loads((writer.directory / "manifest.json").read_text())
    meta["audit_sha256"] = digest(path)
    write_json(writer.directory / "manifest.json", meta)
    with pytest.raises(ValueError, match="differs from evidence"):
        verify_gate(writer.directory)


def test_bundled_audit_recomputes_from_original_inputs(tmp_path):
    import zipfile
    from pathlib import Path
    from catan_rl.audit import existing_evidence
    root = Path(__file__).resolve().parents[1]
    with zipfile.ZipFile(root / "examples/reward-audit-inputs.zip") as archive:
        archive.extractall(tmp_path)
    actual = existing_evidence(tmp_path / "dataset", {arm: tmp_path / arm for arm in ("untuned", "sft", "grpo")})
    expected = json.loads((root / "examples/runs/reward-audit-v1/audit.json").read_text())
    assert actual == {key: expected[key] for key in actual}
    assert not verify_gate(root / "examples/runs/reward-audit-v1")["recommend_training"]
