import json
import random

import pytest
from catanatron import Color, Game, RandomPlayer

from catan_rl import stage6
from catan_rl.telemetry import RunWriter, write_json


def test_teacher_covers_every_action_without_mutating_game_or_engine_rng():
    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=42)
    rng = random.getstate()
    actions = list(game.playable_actions)
    values, visits = stage6.teacher_values(game, 71)
    assert len(values) == len(actions) == 54
    assert min(visits) >= 2
    assert sum(visits) == 200
    assert all(0 <= v <= 1 for v in values)
    assert random.getstate() == rng
    assert game.playable_actions == actions
    assert not game.state.board.buildings


def test_reward_and_ties_are_fixed_without_sample_normalization():
    row = {"actions": [{}, {}, {}], "values": [.25, .75, .75]}
    assert stage6.best_actions(row["values"]) == [1, 2]
    assert stage6.score_output("A0", row) == (0, -.5)
    assert stage6.score_output("A2", row) == (2, .5)
    assert stage6.score_output("A3", row) == (None, -2.)
    assert stage6.score_output("A1 or A2", row) == (None, -2.)


def make_dataset(path, duplicate_seed=False):
    writer = RunWriter(path, "teacher_dataset", {**stage6.RULES, "smoke": True})
    hashes = {}
    for i, split in enumerate(("train", "dev", "test")):
        row = {"id": str(i), "game_seed": 42 if duplicate_seed else i, "split": split,
               "actions": [{}, {}], "values": [.25, .75], "visits": [100, 100], "best": [1]}
        target = path / f"{split}.jsonl"
        target.write_text(json.dumps(row) + "\n")
        hashes[split] = stage6.digest(target)
    writer.finish(dataset_sha256=hashes, counts=dict.fromkeys(hashes, 1))


def test_dataset_rejects_game_leakage_even_with_distinct_state_ids(tmp_path):
    make_dataset(tmp_path, duplicate_seed=True)
    with pytest.raises(ValueError, match="Source game leaked"):
        stage6.read_dataset(tmp_path, True)


def test_dataset_detects_edits_and_smoke_data_cannot_train_real_run(tmp_path):
    make_dataset(tmp_path)
    assert len(stage6.read_dataset(tmp_path, True)[0]["train"]) == 1
    with pytest.raises(ValueError, match="frozen rules"):
        stage6.read_dataset(tmp_path, False)
    with (tmp_path / "test.jsonl").open("a") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        stage6.read_dataset(tmp_path, True)


def test_bootstrap_keeps_correlated_seats_together():
    result = stage6.paired_delta([1, -1, 1, -1], [42, 42, 43, 43])
    assert result["difference"] == 0
    assert result["interval"] == [0, 0]
    assert result["clusters"] == 2
    assert result["degenerate_bootstrap"]


def test_comparison_cannot_pass_with_incomplete_games(tmp_path):
    for name, complete, agreements in [("base", True, [False, False]), ("new", False, [True, True])]:
        path = tmp_path / name
        writer = RunWriter(path, "llm_evaluation", {**stage6.RULES, "dataset_sha256": {}, "smoke": False})
        write_json(path / "heldout.json", {"predictions": [
            {"id": str(i), "game_seed": i, "agreement": a} for i, a in enumerate(agreements)]})
        write_json(path / "result.json", {"complete": complete, "outcomes": ["loss", "loss"], "smoke": False})
        writer.finish()
    result = stage6.compare_evaluations(tmp_path / "base", tmp_path / "new")
    assert result["agreement"]["difference"] == 1
    assert result["wins"] is None
    assert result["outcome"] == "inconclusive"


def test_language_fallback_is_explicit_and_reproducible(monkeypatch):
    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=12)
    monkeypatch.setattr(stage6, "generate", lambda *args: (None, None, None, ["A99999"]))
    first = stage6.LanguageAgent(Color.RED, None, None, "cpu", 71)
    second = stage6.LanguageAgent(Color.RED, None, None, "cpu", 71)
    assert first.decide(game, game.playable_actions) == second.decide(game, game.playable_actions)
    assert first.last_decision["fallback"]
    assert first.last_decision["reason"] == "invalid_output"
