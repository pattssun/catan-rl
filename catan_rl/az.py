"""AlphaZero loop for Catan (Stage 2: 1v1, perfect-information search).

PUCT-MCTS where the policy head prunes breadth (priors order
exploration) and the value head prunes depth (no rollouts — leaf value
is the net's estimate). Self-play labels: policy head learns the MCTS
visit distribution, value head learns the game outcome. Chance edges
(dice, dev draws, steals) sampled as in Stage 1.

Usage: uv run python -m catan_rl.az --iterations 5 --games 30 --sims 60
"""

import argparse
import math
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from catanatron import Color, Game, RandomPlayer
from catanatron.game import TURNS_LIMIT
from catanatron.models.enums import ActionRecord, ActionType
from catanatron.models.player import Player
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

from catanatron.state_functions import get_actual_victory_points

from catan_rl.mcts import DICE_PAIRS, is_chance
from catan_rl.net import Encoder, PolicyValueNet


def game_winner(game):
    """Winning color; at a turn cutoff the VP leader wins (tie -> None).

    Catan under aimless play does not self-terminate the way Go does —
    without this, cold-start self-play games all hit the turn limit,
    z carries no signal, and the net learns passivity.
    """
    winner = game.winning_color()
    if winner is not None:
        return winner
    vps = {c: get_actual_victory_points(game.state, c) for c in game.state.colors}
    top = max(vps.values())
    leaders = [c for c, vp in vps.items() if vp == top]
    return leaders[0] if len(leaders) == 1 else None


def _mover_value(value, mover, leaf_color, state):
    """Leaf value from `mover`'s perspective. Scalar value = zero-sum 1v1
    (negate for the opponent); vector value = per-seat, in relative order
    from the leaf's current color."""
    if isinstance(value, float):
        return value if mover == leaf_color else -value
    rel = (state.color_to_index[mover] - state.color_to_index[leaf_color]) % len(
        state.colors
    )
    return float(value[rel])


class AZNode:
    __slots__ = ("game", "mover", "N", "W", "priors", "children")

    def __init__(self, game, mover):
        self.game = game
        self.mover = mover
        self.N = 0
        self.W = 0.0
        self.priors = None  # dict[action -> P]; None until net-evaluated
        self.children = {}  # action -> {outcome_key: AZNode}

    @property
    def is_terminal(self):
        return (
            self.game.winning_color() is not None
            or self.game.state.num_turns >= TURNS_LIMIT
        )


class AZMCTS:
    def __init__(self, net, encoder, c_puct=1.5, seed=None):
        self.net = net
        self.encoder = encoder
        self.c_puct = c_puct
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

    def search(self, game, num_simulations, root_noise=None):
        """Returns {action: visit_count} at the root."""
        root = AZNode(game.copy(), None)
        self._evaluate(root)
        if root_noise is not None:
            alpha, frac = root_noise
            noise = self.np_rng.dirichlet([alpha] * len(root.priors))
            root.priors = {
                a: (1 - frac) * p + frac * n
                for (a, p), n in zip(root.priors.items(), noise)
            }

        for _ in range(num_simulations):
            path = [root]
            node = root
            while node.priors is not None and not node.is_terminal:
                node = self._step(node, self._select_action(node))
                path.append(node)

            if node.is_terminal:
                winner = game_winner(node.game)
                for n in path:
                    n.N += 1
                    if n.mover is not None:
                        n.W += 1.0 if winner == n.mover else 0.0 if winner is None else -1.0
            else:
                value = self._evaluate(node)  # from node's current color's view
                leaf_state = node.game.state
                leaf_color = leaf_state.current_color()
                for n in path:
                    n.N += 1
                    if n.mover is not None:
                        n.W += _mover_value(value, n.mover, leaf_color, leaf_state)

        return {
            action: sum(c.N for c in bucket.values())
            for action, bucket in root.children.items()
        }

    def _evaluate(self, node):
        """Set priors from the policy head; return the value head's estimate."""
        game = node.game
        color = game.state.current_color()
        actions = list(game.playable_actions)
        features = self.encoder.encode(game, color)
        legal = [self.encoder.action_to_index(a) for a in actions]
        priors, value = self.net.predict(features, legal)
        node.priors = dict(zip(actions, priors.tolist()))
        return value

    def _select_action(self, node):
        sqrt_n = math.sqrt(node.N + 1)
        best_action, best_score = None, -math.inf
        for action, prior in node.priors.items():
            bucket = node.children.get(action)
            if bucket:
                n_a = sum(c.N for c in bucket.values())
                q = sum(c.W for c in bucket.values()) / n_a
            else:
                n_a, q = 0, 0.0
            score = q + self.c_puct * prior * sqrt_n / (1 + n_a)
            if score > best_score:
                best_action, best_score = action, score
        return best_action

    def _step(self, node, action):
        bucket = node.children.setdefault(action, {})
        mover = node.game.state.current_color()

        if action.action_type == ActionType.ROLL:
            pair = self.rng.choice(DICE_PAIRS)
            key = pair[0] + pair[1]
            child = bucket.get(key)
            if child is None:
                game = node.game.copy()
                game.execute(action, action_record=ActionRecord(action, pair))
                child = bucket[key] = AZNode(game, mover)
            return child

        if is_chance(action):
            game = node.game.copy()
            record = game.execute(action)
            child = bucket.get(record.result)
            if child is None:
                child = bucket[record.result] = AZNode(game, mover)
            return child

        child = bucket.get(None)
        if child is None:
            game = node.game.copy()
            game.execute(action)
            child = bucket[None] = AZNode(game, mover)
        return child


class AZAgent(Player):
    def __init__(self, color, net, encoder, num_simulations=60, seed=None):
        super().__init__(color)
        self.num_simulations = num_simulations
        self.mcts = AZMCTS(net, encoder, seed=seed)

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        visits = self.mcts.search(game, self.num_simulations)
        return max(visits, key=visits.get)


def self_play_game(net, encoder, num_simulations, seed, temp_moves=30,
                   noise=(0.3, 0.25), turn_cap=400, opponents=None):
    """Play one self-play game; return (samples, winner, num_turns).

    Samples are (features, visit_dist, z) with z from that state's
    current player's perspective. Games hitting turn_cap are scored by
    VP leader (see game_winner). `opponents` maps colors to Player
    instances for seats not driven by the net (league-style population
    play); those seats yield no training samples.
    """
    rng = random.Random(seed)
    opponents = opponents or {}
    game = Game([RandomPlayer(c) for c in encoder.colors], seed=seed)
    mcts = AZMCTS(net, encoder, seed=seed)
    records = []
    decision = 0
    while game.winning_color() is None and game.state.num_turns < turn_cap:
        actions = game.playable_actions
        if len(actions) == 1:
            game.execute(actions[0])
            continue
        color = game.state.current_color()
        if color in opponents:
            game.execute(opponents[color].decide(game, actions))
            continue
        visits = mcts.search(game, num_simulations, root_noise=noise)
        pi = np.zeros(encoder.num_actions, dtype=np.float32)
        for action, n in visits.items():
            pi[encoder.action_to_index(action)] += n
        pi /= pi.sum()
        records.append((encoder.encode(game, color), pi, color))

        items = list(visits.items())
        if decision < temp_moves:
            weights = [n for _, n in items]
            action = rng.choices([a for a, _ in items], weights)[0]
        else:
            action = max(visits, key=visits.get)
        game.execute(action)
        decision += 1

    winner = game_winner(game)
    colors = game.state.colors
    index = game.state.color_to_index

    def z_for(color):
        if net.num_values == 1:
            return 1.0 if color == winner else 0.0 if winner is None else -1.0
        z = np.zeros(net.num_values, dtype=np.float32)
        if winner is not None:
            for c in colors:
                rel = (index[c] - index[color]) % len(colors)
                z[rel] = 1.0 if c == winner else -1.0
        return z

    samples = [(f, pi, z_for(color)) for f, pi, color in records]
    return samples, winner, game.state.num_turns


def train_steps(net, buffer, steps, batch_size=256, lr=1e-3, device="cpu"):
    net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    losses = []
    for _ in range(steps):
        batch = random.sample(buffer, min(batch_size, len(buffer)))
        x = torch.from_numpy(np.stack([b[0] for b in batch])).to(device)
        pi = torch.from_numpy(np.stack([b[1] for b in batch])).to(device)
        z = torch.from_numpy(np.asarray([b[2] for b in batch], dtype=np.float32)).to(device)
        logits, v = net(x)
        policy_loss = -(pi * torch.log_softmax(logits, dim=1)).sum(1).mean()
        value_loss = torch.nn.functional.mse_loss(v, z)
        loss = policy_loss + value_loss
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append((policy_loss.item(), value_loss.item()))
    net.to("cpu").eval()
    return tuple(np.mean(losses, axis=0))


def evaluate(net, encoder, opponent_cls, n_games, num_simulations, seed=0,
             turn_cap=400):
    """AZ agent vs a table of opponent_cls, rotating the AZ seat.
    Games hitting turn_cap are scored by VP leader — bounds eval time
    against passive early nets and still measures who is ahead."""
    wins = 0
    colors = list(encoder.colors)
    for i in range(n_games):
        az_color = colors[i % len(colors)]
        players = [AZAgent(az_color, net, encoder, num_simulations, seed=seed + i)]
        players += [opponent_cls(c) for c in colors if c != az_color]
        game = Game(players, seed=seed + i)
        while game.winning_color() is None and game.state.num_turns < turn_cap:
            game.play_tick()
        if game_winner(game) == az_color:
            wins += 1
    return wins / n_games


def save_checkpoint(net, encoder, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": net.state_dict(),
        "num_features": encoder.num_features,
        "num_actions": encoder.num_actions,
        "num_values": net.num_values,
        "colors": [c.value for c in encoder.colors],
    }, path)


def load_checkpoint(path):
    ckpt = torch.load(path, weights_only=False)
    encoder = Encoder([Color(v) for v in ckpt["colors"]])
    net = PolicyValueNet(ckpt["num_features"], ckpt["num_actions"],
                         num_values=ckpt.get("num_values", 1))
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net, encoder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--games", type=int, default=30, help="self-play games per iteration")
    parser.add_argument("--sims", type=int, default=60)
    parser.add_argument("--train-steps", type=int, default=150)
    parser.add_argument("--turn-cap", type=int, default=300)
    parser.add_argument("--temp-moves", type=int, default=60)
    parser.add_argument("--eval-games", type=int, default=20)
    parser.add_argument("--eval-sims", type=int, default=60)
    parser.add_argument("--buffer", type=int, default=100_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--players", type=int, default=2, choices=[2, 3, 4])
    parser.add_argument("--pool", type=str, default=None,
                        help="comma-separated agent specs; each self-play game "
                             "fills each non-net seat from the pool with prob 0.5")
    parser.add_argument("--out", type=str, default="runs/az")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.resume:
        net, encoder = load_checkpoint(Path(args.resume))
        print(f"resumed from {args.resume}")
    else:
        encoder = Encoder([Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE][:args.players])
        num_values = 1 if args.players == 2 else args.players
        net = PolicyValueNet(encoder.num_features, encoder.num_actions,
                             num_values=num_values)
        net.eval()
    out = Path(args.out)
    buffer = deque(maxlen=args.buffer)
    pool_specs = args.pool.split(",") if args.pool else []
    pool_rng = random.Random(args.seed + 1)

    for it in range(1, args.iterations + 1):
        t0 = time.time()
        turns, wins = [], {c: 0 for c in encoder.colors}
        for g in range(args.games):
            opponents = {}
            if pool_specs:
                from catan_rl.benchmark import resolve_factory
                # keep >=1 net seat; every other seat drawn from pool half the time
                for color in list(encoder.colors)[1:]:
                    if pool_rng.random() < 0.5:
                        spec = pool_rng.choice(pool_specs)
                        opponents[color] = resolve_factory(spec)(color)
            samples, winner, num_turns = self_play_game(
                net, encoder, args.sims, seed=args.seed + it * 10_000 + g,
                temp_moves=args.temp_moves, turn_cap=args.turn_cap,
                opponents=opponents)
            buffer.extend(samples)
            turns.append(num_turns)
            if winner is not None:
                wins[winner] += 1
        sp_time = time.time() - t0

        ploss, vloss = train_steps(net, buffer, args.train_steps)

        wr_weighted = evaluate(net, encoder, WeightedRandomPlayer,
                               args.eval_games, args.eval_sims, seed=it * 777)
        wr_value = evaluate(net, encoder, ValueFunctionPlayer,
                            args.eval_games, args.eval_sims, seed=it * 777)

        save_checkpoint(net, encoder, out / f"iter{it:03d}.pt")
        print(f"iter {it:2d} | selfplay {sp_time:5.0f}s avg_turns {np.mean(turns):5.0f} "
              f"{dict((c.value, w) for c, w in wins.items())} | buffer {len(buffer):6d} "
              f"| ploss {ploss:.3f} vloss {vloss:.3f} "
              f"| vs weighted {wr_weighted:.0%} vs value {wr_value:.0%} "
              f"| total {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
