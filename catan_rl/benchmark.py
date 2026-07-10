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

PLAYER_FACTORIES = {
    "random": RandomPlayer,
    "weighted": WeightedRandomPlayer,
    "vp": VictoryPointPlayer,
    "value": ValueFunctionPlayer,
    "alphabeta": AlphaBetaPlayer,
}

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
            players.append(PLAYER_FACTORIES[agent_names[j]](color))
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
    parser.add_argument("agents", nargs="+", choices=PLAYER_FACTORIES.keys(),
                        help="2-4 agent names")
    parser.add_argument("-n", "--n-games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if not 2 <= len(args.agents) <= 4:
        parser.error("need 2-4 agents")
    report(run_match(args.agents, args.n_games, args.seed))


if __name__ == "__main__":
    main()
