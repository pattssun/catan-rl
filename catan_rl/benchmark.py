"""Run N games between any agents and report win rates.

Usage: uv run python -m catan_rl.benchmark random weighted -n 100
"""

import argparse
import time
from itertools import cycle

from catanatron import Color, Game, RandomPlayer
from catanatron.players.minimax import AlphaBetaPlayer
from catanatron.players.search import VictoryPointPlayer
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

from catan_rl.mcts import MCTSAgent

PLAYER_FACTORIES = {
    "random": RandomPlayer,
    "weighted": WeightedRandomPlayer,
    "vp": VictoryPointPlayer,
    "value": ValueFunctionPlayer,
    "alphabeta": AlphaBetaPlayer,
}


def resolve_factory(spec):
    """'mcts:200:weighted:150' -> factory. Parts: sims, rollout, horizon."""
    name, *args = spec.split(":")
    if name == "mcts":
        sims = int(args[0]) if len(args) > 0 else 100
        rollout = args[1] if len(args) > 1 else "random"
        horizon = int(args[2]) if len(args) > 2 else None
        return lambda color: MCTSAgent(color, num_simulations=sims,
                                       rollout=rollout, horizon=horizon)
    if name == "dmcts":  # dmcts:sims:worlds[:rollout[:horizon]]
        from catan_rl.determinize import DeterminizedMCTSAgent
        sims = int(args[0]) if len(args) > 0 else 100
        worlds = int(args[1]) if len(args) > 1 else 8
        rollout = args[2] if len(args) > 2 else "random"
        horizon = int(args[3]) if len(args) > 3 else None
        return lambda color: DeterminizedMCTSAgent(
            color, num_simulations=sims, worlds=worlds,
            rollout=rollout, horizon=horizon)
    if name == "az":  # az:runs/az/iter005.pt:60
        from catan_rl.az import AZAgent, load_checkpoint
        net, encoder = load_checkpoint(args[0])
        sims = int(args[1]) if len(args) > 1 else 60
        return lambda color: AZAgent(color, net, encoder, num_simulations=sims)
    return PLAYER_FACTORIES[name]

COLORS = [Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE]


def run_match(agent_names, n_games, seed=0):
    """Play n_games between the named agents, rotating seating each game.

    Returns per-agent stats keyed by a unique label (name#i on duplicates).
    """
    labels = [
        name if agent_names.count(name) == 1 else f"{name}#{i}"
        for i, name in enumerate(agent_names)
    ]
    stats = {label: {"wins": 0} for label in labels}
    stats["draws"] = 0
    turns, t0 = [], time.time()

    for i in range(n_games):
        # rotate seating so no agent always goes first
        order = [(j + i) % len(labels) for j in range(len(labels))]
        color_to_label = {}
        players = []
        for color, j in zip(COLORS, order):
            players.append(resolve_factory(agent_names[j])(color))
            color_to_label[color] = labels[j]
        game = Game(players, seed=seed + i)
        winner = game.play()
        if winner is None:
            stats["draws"] += 1
        else:
            stats[color_to_label[winner]]["wins"] += 1
        turns.append(game.state.num_turns)

    elapsed = time.time() - t0
    return {
        "stats": stats,
        "labels": labels,
        "n_games": n_games,
        "avg_turns": sum(turns) / len(turns),
        "elapsed": elapsed,
    }


def report(result):
    n = result["n_games"]
    print(f"\n{n} games | avg {result['avg_turns']:.0f} turns"
          f" | {result['elapsed']:.1f}s total ({result['elapsed'] / n * 1000:.0f}ms/game)")
    for label in result["labels"]:
        wins = result["stats"][label]["wins"]
        print(f"  {label:<12} {wins:>4} wins  {wins / n:6.1%}")
    draws = result["stats"]["draws"]
    if draws:
        print(f"  {'(draws)':<12} {draws:>4}       {draws / n:6.1%}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agents", nargs="+",
                        help=f"2-4 agent specs: {list(PLAYER_FACTORIES)} or mcts[:sims[:rollout[:horizon]]]")
    parser.add_argument("-n", "--n-games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not 2 <= len(args.agents) <= 4:
        parser.error("need 2-4 agents")
    report(run_match(args.agents, args.n_games, args.seed))


if __name__ == "__main__":
    main()
