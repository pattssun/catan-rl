"""Versioned, complete environment snapshots for new diagnostic experiments."""

import argparse
from collections import defaultdict
from enum import Enum
from functools import lru_cache
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import tempfile

import catanatron
from catanatron.game import Game
from catanatron.models.board import Board
from catanatron.models.enums import Action, ActionPrompt, ActionRecord, ActionType
from catanatron.models.map import CatanMap, LandTile, Port, Water
from catanatron.models.player import Color, Player
from catanatron.state import State
from catan_rl.randomness import isolated_random

SCHEMA = "catan_environment_v2"
ORDERING = "typed_json_lexicographic_v1"
CACHE_FIELDS = {"buildable_subgraph", "buildable_edges_cache", "player_port_resources_cache"}


def dumps(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def checksum(value):
    return hashlib.sha256(dumps(value).encode()).hexdigest()


def encode(value):
    if isinstance(value, Enum):
        return {"enum": type(value).__name__, "value": value.value}
    if value is None or type(value) in (str, int, float, bool):
        return value
    if isinstance(value, (Action, ActionRecord)):
        return {"record": type(value).__name__, "items": [encode(v) for v in value]}
    if isinstance(value, dict):
        items = [[encode(k), encode(v)] for k, v in value.items()]
        # Dictionary insertion order affects some engine tie-breaking paths.
        return {"mapping": items, "default": (value.default_factory.__name__
                if isinstance(value, defaultdict) else None)}
    if isinstance(value, (list, tuple, set, frozenset)):
        items = [encode(v) for v in value]
        if isinstance(value, (set, frozenset)):
            items.sort(key=dumps)
        return {type(value).__name__: items}
    if type(value) in (CatanMap, LandTile, Port, Water):
        return {"object": type(value).__name__, "fields": encode(vars(value))}
    raise TypeError(f"Unsupported environment field: {type(value).__name__}")


def decode(value):
    if not isinstance(value, dict):
        if value is None or type(value) in (str, int, float, bool):
            return value
        raise ValueError("Invalid encoded field")
    if "enum" in value:
        from catanatron.models.coordinate_system import Direction
        from catanatron.models.map import EdgeRef, NodeRef
        enums = {cls.__name__: cls for cls in (Color, ActionType, ActionPrompt, Direction, EdgeRef, NodeRef)}
        return enums[value["enum"]](value["value"])
    if "record" in value:
        return {"Action": Action, "ActionRecord": ActionRecord}[value["record"]](
            *(decode(v) for v in value["items"]))
    if "mapping" in value:
        pairs = [(decode(k), decode(v)) for k, v in value["mapping"]]
        if len({k for k, _ in pairs}) != len(pairs):
            raise ValueError("Duplicate mapping key")
        if value["default"] is None:
            return dict(pairs)
        return defaultdict({"list": list, "int": int, "set": set}[value["default"]], pairs)
    for name, constructor in (("list", list), ("tuple", tuple), ("set", set), ("frozenset", frozenset)):
        if name in value:
            return constructor(decode(v) for v in value[name])
    if "object" in value:
        cls = {c.__name__: c for c in (CatanMap, LandTile, Port, Water)}[value["object"]]
        obj = cls.__new__(cls)
        obj.__dict__.update(decode(value["fields"]))
        return obj
    raise ValueError("Unknown encoded field")


def action_key(action):
    return dumps(encode(action))


class CanonicalGame(Game):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if hasattr(self, "playable_actions"):
            self.playable_actions = sorted(self.playable_actions, key=action_key)

    def execute(self, *args, **kwargs):
        result = super().execute(*args, **kwargs)
        self.playable_actions = sorted(self.playable_actions, key=action_key)
        return result

    def copy(self):
        game = super().copy()
        game.__class__ = CanonicalGame
        return game


@lru_cache(maxsize=1)
def compatibility():
    root = Path(catanatron.__file__).parent
    source = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        source.update(str(path.relative_to(root)).encode())
        source.update(path.read_bytes())
    return {"python": platform.python_version(), "catanatron": version("catanatron"),
            "engine_sha256": source.hexdigest(), "networkx": version("networkx")}


def semantic_state(game):
    state = game.state
    return {"seed": game.seed, "vps_to_win": game.vps_to_win,
            "friendly_robber": game.friendly_robber,
            # Attribute assignment order is not game state; nested mapping order can be.
            "state": encode({k: vars(state)[k] for k in sorted(vars(state)) if k not in ("players", "board")}),
            "board": encode({k: vars(state.board)[k] for k in sorted(vars(state.board)) if k not in CACHE_FIELDS}),
            "legal_actions": encode(game.playable_actions)}


def capture(game, environment_rng):
    if not isinstance(game, CanonicalGame):
        raise ValueError("Snapshots require the canonical environment boundary")
    payload = {"schema": SCHEMA, "ordering": ORDERING, "compatibility": compatibility(),
               "environment": semantic_state(game), "rng": encode(environment_rng.getstate())}
    return {**payload, "sha256": checksum(payload)}


def restore(record):
    payload = {k: v for k, v in record.items() if k != "sha256"}
    if record.get("sha256") != checksum(payload):
        raise ValueError("Snapshot hash mismatch")
    if record.get("schema") != SCHEMA or record.get("ordering") != ORDERING:
        raise ValueError("Unsupported snapshot format")
    if record.get("compatibility") != compatibility():
        raise ValueError("Incompatible engine or Python version")
    saved = record["environment"]
    rng = random.Random()
    rng.setstate(decode(record["rng"]))
    game = CanonicalGame(players=[], initialize=False)
    game.seed, game.id = saved["seed"], "restored"
    game.vps_to_win, game.friendly_robber = saved["vps_to_win"], saved["friendly_robber"]
    game.state = State([], initialize=False)
    game.state.__dict__.update(decode(saved["state"]))
    game.state.players = [Player(c) for c in game.state.colors]
    board_fields = decode(saved["board"])
    game.state.board = Board(board_fields["map"])
    game.state.board.__dict__.update(board_fields)
    from catanatron.models.actions import generate_playable_actions
    game.playable_actions = sorted(generate_playable_actions(game.state), key=action_key)
    if semantic_state(game) != saved:
        raise ValueError("Recovered semantic state or legal actions differ")
    return game, rng


def continuation(game, environment_rng, steps=100):
    trace = []
    for _ in range(steps):
        if game.winning_color() is not None:
            break
        # Fixed random policy with its own stream, independent of engine draws.
        # The counter seed makes the probe independent of prior agent state.
        action = random.Random(710000 + len(trace)).choice(game.playable_actions)
        with isolated_random(environment_rng):
            result = game.execute(action)
        trace.append({"action": encode(action), "result": encode(result),
                      "state_sha256": checksum(semantic_state(game)),
                      "rng_sha256": checksum(encode(environment_rng.getstate()))})
    return trace


def verify_replay(record, game=None, environment_rng=None, steps=100, expected=None):
    reference = "live_environment" if expected is not None or game is not None else "restored_environment"
    if expected is None:
        if game is None:
            game, environment_rng = restore(record)
        rng = random.Random()
        rng.setstate(environment_rng.getstate())
        expected = {"initial_sha256": checksum(semantic_state(game)),
                    "trace": continuation(game.copy(), rng, steps)}
    processes = []
    with tempfile.TemporaryDirectory(prefix="catan-replay-") as directory:
        path = Path(directory) / "state.json"
        path.write_text(dumps(record))
        for hash_seed in ("0", "1", "2"):
            result = subprocess.run([sys.executable, "-m", "catan_rl.replay", str(path), "--steps", str(steps)],
                                    env={**os.environ, "PYTHONHASHSEED": hash_seed},
                                    capture_output=True, text=True, timeout=45, check=True)
            actual = json.loads(result.stdout)
            processes.append({"hash_seed": hash_seed, "matches": actual == expected,
                              "trace_sha256": checksum(actual)})
    return {"passed": all(p["matches"] for p in processes), "steps": len(expected["trace"]),
            "snapshot_sha256": record["sha256"], "expected_sha256": checksum(expected),
            "reference": reference,
            "processes": processes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--expected", type=Path)
    args = parser.parse_args()
    record = json.loads(args.snapshot.read_text())
    if args.check:
        expected = json.loads(args.expected.read_text()) if args.expected else None
        result = verify_replay(record, steps=args.steps, expected=expected)
        print(dumps(result))
        raise SystemExit(0 if result["passed"] else 2)
    game, rng = restore(record)
    print(dumps({"initial_sha256": checksum(semantic_state(game)),
                 "trace": continuation(game, rng, args.steps)}))


if __name__ == "__main__":
    main()
