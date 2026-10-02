import copy
import json
import os
import random
import subprocess
import sys

import pytest
from catanatron.models.actions import generate_playable_actions
from catanatron.models.player import Color
from catanatron.players.weighted_random import WeightedRandomPlayer
from catan_rl.randomness import isolated_random
from catan_rl.replay import (CanonicalGame, action_key, capture, checksum, continuation,
                             restore, semantic_state)


@pytest.fixture(scope="module")
def positions():
    engine_rng = random.Random(913)
    with isolated_random(engine_rng):
        game = CanonicalGame([WeightedRandomPlayer(Color.RED), WeightedRandomPlayer(Color.BLUE)], seed=913)
    decision_rng = random.Random(942)
    wanted = {"BUILD_SETTLEMENT", "END_TURN", "MOVE_ROBBER", "BUY_DEVELOPMENT_CARD",
              "PLAY_KNIGHT_CARD", "PLAY_MONOPOLY", "PLAY_ROAD_BUILDING", "PLAY_YEAR_OF_PLENTY",
              "DISCARD_RESOURCE", "MARITIME_TRADE"}
    found = {}
    for _ in range(2000):
        for action in game.playable_actions:
            name = action.action_type.value
            if name in wanted and name not in found:
                found[name] = (game.copy(), copy.deepcopy(engine_rng))
        if found.keys() == wanted or game.winning_color() is not None:
            break
        with isolated_random(decision_rng):
            action = game.state.current_player().decide(game, game.playable_actions)
        with isolated_random(engine_rng):
            game.execute(action)
    assert found.keys() == wanted
    for name in ("PLAY_KNIGHT_CARD", "PLAY_ROAD_BUILDING"):
        position, rng = found[name]
        pending, pending_rng = position.copy(), copy.deepcopy(rng)
        action = next(a for a in pending.playable_actions if a.action_type.value == name)
        with isolated_random(pending_rng):
            pending.execute(action)
        found["pending_" + name] = (pending, pending_rng)
    return found


def test_complete_fields_and_rng_are_recovered_independently(positions):
    for original, rng in positions.values():
        before = random.getstate()
        recovered, recovered_rng = restore(json.loads(json.dumps(capture(original, rng))))
        assert random.getstate() == before
        assert recovered_rng.getstate() == rng.getstate()
        for key, value in vars(original.state).items():
            if key not in ("players", "board"):
                assert getattr(recovered.state, key) == value, key
        for key, value in vars(original.state.board).items():
            if key not in ("map", "buildable_subgraph", "buildable_edges_cache", "player_port_resources_cache"):
                assert getattr(recovered.state.board, key) == value, key
        for coordinate, tile in original.state.board.map.tiles.items():
            assert vars(recovered.state.board.map.tiles[coordinate]) == vars(tile)
        assert list(recovered.state.board.buildable_subgraph.edges) == list(original.state.board.buildable_subgraph.edges)
        assert recovered.playable_actions == original.playable_actions
        assert set(recovered.playable_actions) == set(generate_playable_actions(original.state))
        assert isinstance(recovered.copy(), CanonicalGame)
        for action in original.playable_actions:
            live, saved = original.copy(), recovered.copy()
            live_rng, saved_rng = copy.deepcopy(rng), copy.deepcopy(recovered_rng)
            with isolated_random(live_rng):
                live_result = live.execute(action)
            with isolated_random(saved_rng):
                saved_result = saved.execute(action)
            assert live_result == saved_result
            assert live.state.player_state == saved.state.player_state
            assert live.state.development_listdeck == saved.state.development_listdeck
            assert live.state.resource_freqdeck == saved.state.resource_freqdeck
            assert live.playable_actions == saved.playable_actions
            assert live_rng.getstate() == saved_rng.getstate()


def test_cross_process_full_state_and_100_future_transitions(positions, tmp_path):
    for name, (game, rng) in positions.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(capture(game, rng)))
        expected = {"initial_sha256": checksum(semantic_state(game)),
                    "trace": continuation(game.copy(), copy.deepcopy(rng))}
        for hash_seed in ("0", "1", "2"):
            result = subprocess.run([sys.executable, "-m", "catan_rl.replay", str(path)],
                                    env={**os.environ, "PYTHONHASHSEED": hash_seed},
                                    capture_output=True, text=True, timeout=30, check=True)
            assert json.loads(result.stdout) == expected, (name, hash_seed)


def test_canonical_order_keeps_every_action(positions):
    for game, _ in positions.values():
        raw = generate_playable_actions(game.state)
        assert len(raw) == len(game.playable_actions)
        assert set(raw) == set(game.playable_actions)
        assert game.playable_actions == sorted(raw, key=action_key)


def test_rejects_corruption_version_mismatch_and_wrong_menu(positions):
    game, rng = next(iter(positions.values()))
    record = capture(game, rng)
    corrupted = copy.deepcopy(record)
    corrupted["environment"]["vps_to_win"] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        restore(corrupted)
    for field, value in (("schema", "unknown"), ("compatibility", {})):
        bad = copy.deepcopy(record)
        bad[field] = value
        bad["sha256"] = checksum({k: v for k, v in bad.items() if k != "sha256"})
        with pytest.raises(ValueError):
            restore(bad)
    bad = copy.deepcopy(record)
    bad["environment"]["legal_actions"]["list"].pop()
    bad["sha256"] = checksum({k: v for k, v in bad.items() if k != "sha256"})
    with pytest.raises(ValueError, match="legal actions differ"):
        restore(bad)


def test_hidden_deck_order_and_rng_position_are_part_of_snapshot(positions):
    game, rng = positions["PLAY_KNIGHT_CARD"]
    baseline = capture(game, rng)
    changed = game.copy()
    changed.state.development_listdeck.reverse()
    assert capture(changed, rng)["sha256"] != baseline["sha256"]
    advanced = copy.deepcopy(rng)
    advanced.random()
    assert capture(game, advanced)["sha256"] != baseline["sha256"]


def test_raw_set_menu_order_varies_but_canonical_menu_does_not(positions, tmp_path):
    game, rng = positions["MARITIME_TRADE"]
    path = tmp_path / "trading.json"
    path.write_text(json.dumps(capture(game, rng)))
    code = '''import json, sys
from catan_rl.replay import restore, action_key
from catanatron.models.actions import generate_playable_actions
game, _ = restore(json.load(open(sys.argv[1])))
print(json.dumps({"raw": [action_key(a) for a in generate_playable_actions(game.state)],
                  "canonical": [action_key(a) for a in game.playable_actions]}))'''
    outputs = [json.loads(subprocess.check_output([sys.executable, "-c", code, str(path)],
                env={**os.environ, "PYTHONHASHSEED": seed}, text=True)) for seed in ("0", "1", "2")]
    assert len({tuple(o["raw"]) for o in outputs}) > 1
    assert len({tuple(o["canonical"]) for o in outputs}) == 1
    assert all(sorted(o["raw"]) == o["canonical"] for o in outputs)


def test_live_game_and_engine_copy_have_the_same_semantic_fingerprint(tmp_path):
    from catan_rl.replay import verify_replay
    rng = random.Random(931)
    with isolated_random(rng):
        live = CanonicalGame([WeightedRandomPlayer(Color.RED), WeightedRandomPlayer(Color.BLUE)], seed=931)
    assert list(vars(live.state)) != list(vars(live.copy().state))
    assert semantic_state(live) == semantic_state(live.copy())
    for _ in range(12):
        with isolated_random(rng):
            live.execute(live.playable_actions[0])
    assert semantic_state(live) == semantic_state(live.copy())
    record = capture(live, rng)
    result = verify_replay(record, live, rng)
    assert result["passed"]
    assert result["reference"] == "live_environment"
    assert result["steps"] == 100
