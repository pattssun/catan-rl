import json
import random

import pytest
from fastapi.testclient import TestClient
from catanatron import Color, Game, RandomPlayer

from catan_rl.evaluation import isolated_random, play_episode
from catan_rl.mcts import MCTS
from catan_rl.server import create_app
from catan_rl.telemetry import RunWriter, import_legacy, public_state, wilson


def test_actual_timeout_is_not_a_win():
    episode = play_episode("weighted", "random", 42, 0, turn_cap=0)
    assert episode["outcome"] == "timeout"
    assert episode["reward"] == 0
    assert episode["winner"] is None


def test_seating_is_real_and_board_is_paired():
    first = play_episode("random", "weighted", 13, 0, action_cap=1)
    second = play_episode("random", "weighted", 13, 1, action_cap=1)
    assert first["agent_color"] != second["agent_color"]
    assert first["board"] == second["board"]
    assert first["steps"][0]["state"] == second["steps"][0]["state"]
    assert first["steps"][0]["actor"] == first["agent_color"]


def test_rollout_does_not_change_environment_randomness():
    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=6)
    before = random.getstate()
    MCTS(seed=11, horizon=1).search_visits(game, 2)
    assert random.getstate() == before
    assert not game.state.board.buildings


def test_isolation_restores_on_error_and_advances_own_stream():
    random.seed(23)
    outer = random.getstate()
    local = random.Random(7)
    with pytest.raises(RuntimeError):
        with isolated_random(local):
            random.random()
            raise RuntimeError()
    assert random.getstate() == outer
    expected = random.Random(7)
    expected.random()
    assert local.random() == expected.random()


def test_public_state_does_not_expose_hidden_hands():
    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=2)
    encoded = json.dumps(public_state(game))
    assert "IN_HAND" not in encoded
    assert "development_listdeck" not in encoded
    assert all(set(p) == {"color", "vp", "resources", "development_cards"}
               for p in public_state(game)["players"])


def test_seeded_episode_repeats():
    a = play_episode("random", "weighted", 21, 0, action_cap=20)
    b = play_episode("random", "weighted", 21, 0, action_cap=20)
    assert [(s["state"], s["action"], s["result"]) for s in a["steps"]] == [
        (s["state"], s["action"], s["result"]) for s in b["steps"]]


def test_wilson_matches_independent_known_interval():
    lo, hi = wilson(50, 100)
    assert lo == pytest.approx(.40383153)
    assert hi == pytest.approx(.59616847)
    assert wilson(0, 0) is None


def test_legacy_import_preserves_metric_meaning(tmp_path):
    log = tmp_path / "old.log"
    log.write_text("iter  1 | selfplay 541s avg_turns 249 {'RED': 18} | buffer 10 | ploss 1.246 vloss 0.042 | vs weighted 3% vs value 0% | total 667s\n")
    import_legacy(log, tmp_path / "runs" / "legacy")
    events = [json.loads(s) for s in (tmp_path / "runs/legacy/events.jsonl").read_text().splitlines()]
    assert events[0]["legacy_weighted_score"] == .03
    assert "win_rate" not in events[0]
    assert not list((tmp_path / "runs/legacy/episodes").iterdir())


def test_run_api_review_audit_and_partial_event(tmp_path):
    root = tmp_path / "runs"
    writer = RunWriter(root / "test", "evaluation", {"evaluation": "actual_outcome_v1"})
    episode = play_episode("random", "weighted", 5, 0, action_cap=2)
    writer.episode(episode)
    writer.event("metrics", step=1, games=1, win_rate=0, timeout_rate=1)
    writer.finish()
    with (root / "test/events.jsonl").open("a") as f:
        f.write('{"kind":')
    client = TestClient(create_app([root], tmp_path / "review.sqlite", tmp_path / "no-dist"))
    runs = client.get("/api/runs").json()
    assert runs[0]["episode_count"] == 1
    assert runs[0]["warnings"]
    assert client.get("/api/runs/0-test/episodes/0000").json()["steps"]
    url = "/api/runs/0-test/episodes/0000/reviews"
    assert client.post(url, json={"status": "confirmed", "note": ""}).status_code == 422
    for status in ["reviewed", "dismissed"]:
        assert client.post(url, json={"status": status, "note": "Test evidence"}).status_code == 200
    assert client.get("/api/runs/0-test").json()["episodes"][0]["review"]["status"] == "dismissed"
    assert len(client.get("/api/runs/0-test/episodes/0000").json()["reviews"]) == 2
    assert json.loads((root / "test/episodes/0000.json").read_text()) == episode


def test_api_rejects_symlink_escape(tmp_path):
    root = tmp_path / "runs"
    root.mkdir()
    outside = tmp_path / "outside"
    RunWriter(outside, "test", {})
    (root / "escaped").symlink_to(outside, target_is_directory=True)
    client = TestClient(create_app([root], tmp_path / "review.sqlite", tmp_path / "no-dist"))
    assert client.get("/api/runs").json() == []
    assert client.get("/api/runs/0-escaped").status_code == 404


def test_audit_api_distinguishes_missing_partial_and_tampered_evidence(tmp_path):
    import hashlib
    root = tmp_path / "runs"
    writer = RunWriter(root / "audit", "reward_audit", {})
    client = TestClient(create_app([root], tmp_path / "review.sqlite", tmp_path / "no-dist"))
    url = "/api/runs/0-audit/audit"
    assert client.get(url).status_code == 404
    path = writer.directory / "audit.json"
    path.write_text(json.dumps({"gate": {"status": "incomplete", "recommend_training": False}}))
    assert client.get(url).json()["gate"]["recommend_training"] is False
    writer.finish(audit_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    assert client.get(url).status_code == 200
    path.write_text(json.dumps({"gate": {"status": "pass", "recommend_training": True}}))
    assert client.get(url).status_code == 409
    path.unlink()
    outside = tmp_path / "outside.json"
    outside.write_text('{}')
    path.symlink_to(outside)
    assert client.get(url).status_code == 404


def test_existing_run_cannot_be_overwritten(tmp_path):
    RunWriter(tmp_path / "a", "test", {})
    with pytest.raises(FileExistsError):
        RunWriter(tmp_path / "a", "test", {})


def test_interval_resamples_boards_not_individual_seats():
    from catan_rl.evaluation import paired_interval
    # Every board has exactly one win. Resampling whole boards preserves 50%.
    assert paired_interval(["win", "loss"] * 10) == [.5, .5]
    assert paired_interval(["loss", "timeout"] * 10) == [0., 0.]
    assert paired_interval(["win", "loss"]) is None


def test_checkpoint_restores_optimizer_for_identical_next_update(tmp_path):
    import torch
    from catan_rl.az import PolicyValueNet, save_checkpoint, load_checkpoint
    from catan_rl.net import Encoder
    from catanatron import Color
    encoder = Encoder([Color.RED, Color.BLUE])
    net = PolicyValueNet(encoder.num_features, encoder.num_actions)
    optimizer = torch.optim.Adam(net.parameters())
    inputs = torch.ones(2, encoder.num_features)
    def update(model, opt):
        opt.zero_grad()
        policy, value = model(inputs)
        loss = policy.square().mean() + value.square().mean()
        loss.backward()
        opt.step()
    update(net, optimizer)
    path = tmp_path / "resume.pt"
    save_checkpoint(net, encoder, path, training={"optimizer": optimizer.state_dict()})
    restored, _ = load_checkpoint(path)
    restored_optimizer = torch.optim.Adam(restored.parameters())
    restored_optimizer.load_state_dict(torch.load(path, weights_only=False)["training"]["optimizer"])
    update(net, optimizer)
    update(restored, restored_optimizer)
    for expected, actual in zip(net.parameters(), restored.parameters()):
        assert torch.equal(expected, actual)


def test_full_resume_matches_uninterrupted_training(tmp_path, monkeypatch):
    import shutil
    import sys
    import torch
    from catan_rl import az, evaluation
    first = tmp_path / "first.pt"
    original_save = az.save_checkpoint
    def capture_first(net, encoder, path, training=None):
        original_save(net, encoder, path, training)
        if training and training["iteration"] == 1:
            shutil.copyfile(path, first)
    monkeypatch.setattr(az, "save_checkpoint", capture_first)
    monkeypatch.setattr(evaluation, "play_episode", lambda *args: {"id": args[-1], "outcome": "loss"})
    settings = ["--iterations", "2", "--games", "1", "--sims", "2", "--train-steps", "1",
                "--turn-cap", "2", "--eval-games", "2", "--eval-sims", "2", "--seed", "37"]
    for name, extra in [("continuous", []), ("resumed", ["--resume", str(first)])]:
        monkeypatch.setattr(sys, "argv", ["az", *settings, "--out", str(tmp_path / name), *extra])
        az.main()
    continuous = torch.load(tmp_path / "continuous/iter002.pt", weights_only=False)["state_dict"]
    resumed = torch.load(tmp_path / "resumed/iter002.pt", weights_only=False)["state_dict"]
    for key in continuous:
        assert torch.equal(continuous[key], resumed[key]), key
