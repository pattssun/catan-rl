"""Fixed-seat evaluation with actual outcomes and public episode traces."""

import argparse
import hashlib
import random
import time
from pathlib import Path

from catanatron import Color, Game, RandomPlayer
from catanatron.json import GameEncoder
from catanatron.state_functions import get_actual_victory_points

from catan_rl.benchmark import resolve_factory
from catan_rl.randomness import isolated_random
from catan_rl.telemetry import RunWriter, action_json, public_board, public_state, wilson


def factory(spec, seed):
    if spec.startswith("policy:"):
        from catan_rl.az import load_checkpoint
        from catan_rl.imitation import PolicyAgent
        net, encoder = load_checkpoint(Path(spec.split(":", 1)[1]))
        return lambda color: PolicyAgent(color, net, encoder, seed=seed)
    return resolve_factory(spec, seed=seed)


def paired_interval(outcomes, seed=611, resamples=10000):
    """Resample complete board pairs, retaining their seat correlation."""
    import numpy as np
    if len(outcomes) < 4 or len(outcomes) % 2:
        return None
    pairs = np.array([o == "win" for o in outcomes], dtype=float).reshape(-1, 2).mean(axis=1)
    rng = np.random.default_rng(seed)
    means = pairs[rng.integers(0, len(pairs), (resamples, len(pairs)))].mean(axis=1)
    return np.quantile(means, [.025, .975]).tolist()


def play_episode(agent, opponent, seed, seat, turn_cap=400, action_cap=8000,
                 episode_id="0000", trace=True, agent_factory=None):
    game = Game([RandomPlayer(Color.RED), RandomPlayer(Color.BLUE)], seed=seed)
    color = game.state.colors[seat]
    game.state.players = [(agent_factory if c == color and agent_factory else
                          factory(agent if c == color else opponent, seed + 123))(c)
                         for c in game.state.colors]
    decision_rng = random.Random(seed + 982451653)
    board = public_board(game) if trace else None
    steps, count, selected, eligible, end_turn, total_seconds = [], 0, {}, 0, 0, 0.0
    while game.winning_color() is None and game.state.num_turns < turn_cap and count < action_cap:
        actions = game.playable_actions
        mover = game.state.current_color()
        before = public_state(game) if trace else None
        start = time.monotonic()
        with isolated_random(decision_rng):
            player = game.state.players[game.state.current_player_index]
            chosen = player.decide(game, actions) if len(actions) > 1 else actions[0]
        elapsed = time.monotonic() - start
        if chosen not in actions:
            raise ValueError(f"Agent selected illegal action: {chosen}")
        if mover == color and len(actions) > 1:
            kind = chosen.action_type.value
            selected[kind] = selected.get(kind, 0) + 1
            if any(a.action_type.value == "END_TURN" for a in actions):
                eligible += 1
                end_turn += kind == "END_TURN"
        record = game.execute(chosen)
        if trace:
            import json
            steps.append({"index": count, "state": before, "actor": mover.value,
                          "action": action_json(chosen), "legal_actions": [action_json(a) for a in actions],
                          "result": json.loads(json.dumps(record.result, cls=GameEncoder)),
                          "seconds": elapsed})
            if mover == color and len(actions) > 1 and hasattr(player, "last_decision"):
                steps[-1]["policy_output"] = player.last_decision
        count += 1
        total_seconds += elapsed
    winner = game.winning_color()
    outcome = "win" if winner == color else "loss" if winner is not None else "timeout"
    points = {c.value: get_actual_victory_points(game.state, c) for c in game.state.colors}
    other = next(c for c in game.state.colors if c != color)
    flags = []
    if outcome == "timeout":
        flags.append({"code": "timeout", "reason": "No actual winner before the evaluation cap."})
        if points[color.value] > points[other.value]:
            flags.append({"code": "cutoff_lead", "reason": "Legacy evaluation would count this cutoff lead as a win."})
    if eligible >= 30 and end_turn / eligible >= .9:
        flags.append({"code": "end_turn_concentration", "reason": "END_TURN selected in at least 90% of 30+ eligible decisions. Review alternatives; this alone is not a reward hack."})
    return {"id": episode_id, "seed": seed, "seat": seat, "agent_color": color.value,
            "outcome": outcome, "winner": winner.value if winner else None,
            "turns": game.state.num_turns, "actions": count, "points": points,
            "reward": 1 if outcome == "win" else -1 if outcome == "loss" else 0,
            "reward_definition": "actual_outcome_v1", "action_counts": selected,
            "end_turn_eligible": eligible, "end_turn_selected": end_turn,
            "seconds": total_seconds, "flags": flags, "board": board, "steps": steps,
            "final_state": public_state(game)}


def evaluate(agent, opponent, games, directory, seed=10000, turn_cap=400,
             action_cap=8000, title=None):
    if games < 2 or games % 2:
        raise ValueError("Use an even number of games for paired seat assignments")
    config = {"agent": agent, "opponent": opponent, "games": games, "seed": seed,
              "turn_cap": turn_cap, "action_cap": action_cap,
              "evaluation": "actual_outcome_v1", "seating": "both_seats_per_board",
              "randomness": "isolated_agent_decisions_v1"}
    for spec in (agent, opponent):
        if spec.startswith(("policy:", "az:")):
            p = Path(spec.split(":")[1])
            config["checkpoint_sha256"] = hashlib.sha256(p.read_bytes()).hexdigest()
    writer = RunWriter(directory, "evaluation", config, title=title)
    wins = losses = timeouts = 0
    outcomes = []
    start = time.monotonic()
    try:
        for i in range(games):
            episode = play_episode(agent, opponent, seed + i // 2, i % 2,
                                   turn_cap, action_cap, f"{i:04d}")
            writer.episode(episode)
            outcomes.append(episode["outcome"])
            wins += episode["outcome"] == "win"
            losses += episode["outcome"] == "loss"
            timeouts += episode["outcome"] == "timeout"
            writer.event("metrics", step=i + 1, games=i + 1, wins=wins,
                         losses=losses, timeouts=timeouts, win_rate=wins / (i + 1),
                         timeout_rate=timeouts / (i + 1),
                         win_interval=wilson(wins, i + 1), interval_method="Wilson descriptive; paired boards are not independent",
                         elapsed_seconds=time.monotonic() - start)
            print(f"{i + 1}/{games}: {episode['outcome']}, {episode['turns']} turns", flush=True)
        writer.event("final_evaluation", games=games, win_rate=wins / games,
                     timeout_rate=timeouts / games, paired_win_interval=paired_interval(outcomes),
                     interval_method="paired-board percentile bootstrap, 10000 resamples, seed 611",
                     target_met=games >= 100 and wins / games >= .6 and paired_interval(outcomes)[0] > .5)
        writer.finish()
    except BaseException as exc:
        writer.finish("failed", error=type(exc).__name__)
        raise
    return writer


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("agent")
    p.add_argument("opponent")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=10000)
    p.add_argument("--turn-cap", type=int, default=400)
    p.add_argument("--action-cap", type=int, default=8000)
    p.add_argument("--title")
    a = p.parse_args()
    evaluate(a.agent, a.opponent, a.games, a.out, a.seed, a.turn_cap, a.action_cap, a.title)


if __name__ == "__main__":
    main()
