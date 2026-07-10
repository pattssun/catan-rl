"""Stage 4: 4-player opponent pool with Elo.

Each game samples 4 distinct agents from the pool with random seating;
ratings update pairwise from the final VP ranking (winner > ... > last),
so a 2nd-place finish against strong opponents still earns rating.
The Stage 1 MCTS enters unchanged: its per-mover backup is max^n, not
minimax, so nothing about it assumed 2 players — the minimax story ends
here, as the roadmap promised.

Usage: uv run python -m catan_rl.pool random weighted vp value -n 100
"""

import argparse
import random
import time
from itertools import combinations

from catanatron import Game
from catanatron.state_functions import get_actual_victory_points

from catan_rl.benchmark import COLORS, resolve_factory


def elo_update(ratings, ranking, k=20.0):
    """Pairwise Elo over a full ranking: [(label, vps), ...] best first."""
    n = len(ranking)
    deltas = {label: 0.0 for label, _ in ranking}
    for (la, vpa), (lb, vpb) in combinations(ranking, 2):
        expected = 1.0 / (1.0 + 10 ** ((ratings[lb] - ratings[la]) / 400.0))
        score = 1.0 if vpa > vpb else 0.5 if vpa == vpb else 0.0
        delta = k / (n - 1) * (score - expected)
        deltas[la] += delta
        deltas[lb] -= delta
    for label, delta in deltas.items():
        ratings[label] += delta


def run_tournament(agent_specs, n_games, seed=0, players_per_game=4, k=20.0):
    rng = random.Random(seed)
    ratings = {spec: 1000.0 for spec in agent_specs}
    games_played = {spec: 0 for spec in agent_specs}
    wins = {spec: 0 for spec in agent_specs}

    for i in range(n_games):
        table = rng.sample(agent_specs, players_per_game)
        players = [resolve_factory(spec)(color)
                   for spec, color in zip(table, COLORS)]
        game = Game(players, seed=seed + i)
        game.play()
        vps = [
            (spec, get_actual_victory_points(game.state, color))
            for spec, color in zip(table, COLORS)
        ]
        ranking = sorted(vps, key=lambda t: -t[1])
        elo_update(ratings, ranking, k=k)
        for spec, _ in vps:
            games_played[spec] += 1
        winner = game.winning_color()
        if winner is not None:
            wins[table[COLORS.index(winner)]] += 1

    return ratings, games_played, wins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("agents", nargs="+", help="4+ agent specs (see benchmark)")
    parser.add_argument("-n", "--n-games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if len(args.agents) < 4:
        parser.error("need at least 4 agent specs")

    t0 = time.time()
    ratings, games_played, wins = run_tournament(args.agents, args.n_games, args.seed)
    print(f"\n{args.n_games} four-player games | {time.time() - t0:.0f}s")
    print(f"  {'agent':<24} {'elo':>6} {'games':>6} {'wins':>5}")
    for spec in sorted(ratings, key=ratings.get, reverse=True):
        print(f"  {spec:<24} {ratings[spec]:>6.0f} {games_played[spec]:>6} {wins[spec]:>5}")


if __name__ == "__main__":
    main()
