"""Winner-imitation baseline (Stage 2 control experiment).

The REINFORCE-style loop Eric Jang describes plateauing: play games by
sampling the policy directly (no search), then train the policy to
imitate every move the *winner* made. One genuinely better move's
gradient drowns in thousands of neutral move-labels — no per-state
improvement signal. Run against the AlphaZero loop's eval curve to see
the credit-assignment sidestep from both sides.

Usage: uv run python -m catan_rl.imitation --iterations 15 --games 200
"""

import argparse
import random
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

from catanatron import Color, Game, RandomPlayer
from catanatron.models.player import Player
from catanatron.players.value import ValueFunctionPlayer
from catanatron.players.weighted_random import WeightedRandomPlayer

from catan_rl.az import game_winner, save_checkpoint
from catan_rl.net import Encoder, PolicyValueNet


def sample_policy_action(net, encoder, game, rng, greedy=False):
    actions = list(game.playable_actions)
    if len(actions) == 1:
        return actions[0]
    features = encoder.encode(game, game.state.current_color())
    legal = [encoder.action_to_index(a) for a in actions]
    priors, _ = net.predict(features, legal)
    if greedy:
        return actions[int(np.argmax(priors))]
    return rng.choices(actions, weights=priors.tolist())[0]


class PolicyAgent(Player):
    def __init__(self, color, net, encoder, greedy=True, seed=None):
        super().__init__(color)
        self.net = net
        self.encoder = encoder
        self.greedy = greedy
        self.rng = random.Random(seed)

    def decide(self, game, playable_actions):
        return sample_policy_action(self.net, self.encoder, game, self.rng,
                                    greedy=self.greedy)


def imitation_game(net, encoder, seed, turn_cap=300):
    """Both seats sample the policy; return the winner's (features, action) moves."""
    rng = random.Random(seed)
    game = Game([RandomPlayer(c) for c in encoder.colors], seed=seed)
    moves = {c: [] for c in encoder.colors}
    while game.winning_color() is None and game.state.num_turns < turn_cap:
        actions = game.playable_actions
        if len(actions) == 1:
            game.execute(actions[0])
            continue
        color = game.state.current_color()
        action = sample_policy_action(net, encoder, game, rng)
        moves[color].append((encoder.encode(game, color),
                             encoder.action_to_index(action)))
        game.execute(action)
    winner = game_winner(game)
    return (moves[winner] if winner is not None else []), winner, game.state.num_turns


def train_policy(net, buffer, steps, batch_size=256, lr=1e-3, device="cpu"):
    net.to(device).train()
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
    losses = []
    for _ in range(steps):
        batch = random.sample(buffer, min(batch_size, len(buffer)))
        x = torch.from_numpy(np.stack([b[0] for b in batch])).to(device)
        y = torch.tensor([b[1] for b in batch], dtype=torch.long, device=device)
        logits, _ = net(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    net.to("cpu").eval()
    return float(np.mean(losses))


def evaluate_policy(net, encoder, opponent_cls, n_games, seed=0, turn_cap=400):
    wins = 0
    colors = list(encoder.colors)
    for i in range(n_games):
        my_color = colors[i % len(colors)]
        players = [PolicyAgent(my_color, net, encoder, seed=seed + i)]
        players += [opponent_cls(c) for c in colors if c != my_color]
        game = Game(players, seed=seed + i)
        while game.winning_color() is None and game.state.num_turns < turn_cap:
            game.play_tick()
        if game_winner(game) == my_color:
            wins += 1
    return wins / n_games


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=15)
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--train-steps", type=int, default=150)
    parser.add_argument("--eval-games", type=int, default=30)
    parser.add_argument("--buffer", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="runs/imitation")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    encoder = Encoder([Color.RED, Color.BLUE])
    net = PolicyValueNet(encoder.num_features, encoder.num_actions)
    net.eval()
    buffer = deque(maxlen=args.buffer)
    out = Path(args.out)

    for it in range(1, args.iterations + 1):
        t0 = time.time()
        turns = []
        for g in range(args.games):
            moves, winner, num_turns = imitation_game(
                net, encoder, seed=args.seed + it * 10_000 + g)
            buffer.extend(moves)
            turns.append(num_turns)
        loss = train_policy(net, buffer, args.train_steps)
        wr_weighted = evaluate_policy(net, encoder, WeightedRandomPlayer,
                                      args.eval_games, seed=it * 777)
        wr_value = evaluate_policy(net, encoder, ValueFunctionPlayer,
                                   args.eval_games, seed=it * 777)
        save_checkpoint(net, encoder, out / f"iter{it:03d}.pt")
        print(f"iter {it:2d} | games {args.games} avg_turns {np.mean(turns):5.0f} "
              f"| buffer {len(buffer):6d} | loss {loss:.3f} "
              f"| vs weighted {wr_weighted:.0%} vs value {wr_value:.0%} "
              f"| {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
