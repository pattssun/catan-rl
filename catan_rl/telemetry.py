"""Append-only metrics and inspectable evaluation episodes."""

import hashlib
import importlib.metadata
import json
import platform
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path



def now():
    return datetime.now(timezone.utc).isoformat()


def write_json(path, value):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=None if "steps" in value else 2, allow_nan=False) + "\n")
    tmp.replace(path)


def provenance():
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted((root / "catan_rl").glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True))
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = None, None
    return {"git_revision": revision, "dirty": dirty,
            "source_sha256": digest.hexdigest(), "python": platform.python_version(),
            "catanatron": importlib.metadata.version("catanatron"),
            "torch": importlib.metadata.version("torch")}


class RunWriter:
    def __init__(self, directory, algorithm, config, title=None, source="measured"):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        if (self.directory / "manifest.json").exists():
            raise FileExistsError(f"Run already exists: {self.directory}")
        (self.directory / "episodes").mkdir(exist_ok=True)
        self.manifest = {"schema_version": 1, "id": self.directory.name,
                         "title": title or self.directory.name, "algorithm": algorithm,
                         "source": source, "created_at": now(), "updated_at": now(),
                         "status": "running", "config": config, "provenance": provenance(),
                         "config_sha256": hashlib.sha256(json.dumps(
                             config, sort_keys=True).encode()).hexdigest()}
        write_json(self.directory / "manifest.json", self.manifest)

    def event(self, kind, **values):
        record = {"kind": kind, "timestamp": now(), **values}
        with (self.directory / "events.jsonl").open("a") as stream:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
        self.manifest["updated_at"] = record["timestamp"]
        write_json(self.directory / "manifest.json", self.manifest)

    def episode(self, episode):
        write_json(self.directory / "episodes" / f"{episode['id']}.json", episode)
        self.event("episode", **{k: v for k, v in episode.items()
                                  if k not in ("steps", "board")})

    def finish(self, status="completed", **values):
        self.event("finished", status=status, **values)
        self.manifest.update(status=status, **values)
        write_json(self.directory / "manifest.json", self.manifest)


def public_state(game):
    from catanatron.models.enums import DEVELOPMENT_CARDS, RESOURCES
    from catanatron.state_functions import player_key
    state = game.state
    players = []
    for color in state.colors:
        key = player_key(state, color)
        ps = state.player_state
        players.append({"color": color.value,
                        "vp": ps[f"{key}_VICTORY_POINTS"],
                        "resources": sum(ps[f"{key}_{r}_IN_HAND"] for r in RESOURCES),
                        "development_cards": sum(ps[f"{key}_{r}_IN_HAND"]
                                                 for r in DEVELOPMENT_CARDS)})
    return {"turn": state.num_turns, "current_color": state.current_color().value,
            "phase": "setup" if state.is_initial_build_phase else "play",
            "players": players,
            "buildings": [[n, c.value, b] for n, (c, b) in state.board.buildings.items()],
            "roads": [[list(edge), color.value] for edge, color in state.board.roads.items()],
            "robber": list(state.board.robber_coordinate)}


def public_board(game):
    from catanatron.json import GameEncoder
    encoded = json.loads(json.dumps(game, cls=GameEncoder))
    return {k: encoded[k] for k in ("tiles", "nodes")}


def action_json(action):
    from catanatron.json import GameEncoder
    return json.loads(json.dumps({"type": action.action_type.value,
                                  "value": action.value}, cls=GameEncoder))


def wilson(wins, games):
    if not games:
        return None
    z = 1.959963984540054
    p, z2 = wins / games, z * z
    mid = (p + z2 / (2 * games)) / (1 + z2 / games)
    half = z * ((p * (1 - p) / games + z2 / (4 * games * games)) ** .5) / (1 + z2 / games)
    return [max(0, mid - half), min(1, mid + half)]


def import_legacy(log, directory):
    log = Path(log)
    pattern = re.compile(r"iter\s+(\d+)\s+\| selfplay\s+(\d+)s avg_turns\s+([\d.]+).*"
                         r"ploss ([\d.]+) vloss ([\d.]+).*vs weighted (\d+)% vs value (\d+)%.*total (\d+)s")
    rows = [pattern.search(line) for line in log.read_text().splitlines()]
    rows = [m.groups() for m in rows if m]
    if not rows:
        raise ValueError("No recognized AlphaZero metrics in log")
    writer = RunWriter(directory, "alphazero", {
        "evaluation": "legacy_vp_leader_at_cutoff", "games_per_evaluation": None,
        "historical_config": "not recorded", "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest()},
        title="AlphaZero · July training", source="legacy_log")
    for it, sp, turns, pl, vl, weighted, value, total in rows:
        writer.event("metrics", step=int(it), selfplay_seconds=int(sp),
                     avg_turns=float(turns), policy_loss=float(pl), value_loss=float(vl),
                     legacy_weighted_score=int(weighted) / 100,
                     legacy_value_score=int(value) / 100, iteration_seconds=int(total))
    writer.finish(notes="Imported historical metrics. Cutoff leads counted as wins; episode traces and evaluation sample counts were not saved.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("log")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    import_legacy(args.log, args.out)
