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
from catan_rl.randomness import isolated_random
from catan_rl.telemetry import RunWriter


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
        self.environment_rng = random.Random(seed)

    def search(self, game, num_simulations, root_noise=None):
        """Returns {action: visit_count} at the root."""
        with isolated_random(self.environment_rng):
            return self._search(game, num_simulations, root_noise)

    def _search(self, game, num_simulations, root_noise=None):
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
                   noise=(0.3, 0.25), turn_cap=400, opponents=None, diagnostics=None,
                   action_cap=8000):
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
    ticks = 0
    action_counts = {}
    while game.winning_color() is None and game.state.num_turns < turn_cap and ticks < action_cap:
        ticks += 1
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
        kind = action.action_type.value
        action_counts[kind] = action_counts.get(kind, 0) + 1
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
    if diagnostics is not None:
        diagnostics.update(actual_winner=game.winning_color().value if game.winning_color() else None,
                           cutoff=game.winning_color() is None, actions=ticks,
                           action_counts=action_counts)
    return samples, winner, game.state.num_turns


def train_steps(net, buffer, steps, batch_size=256, lr=1e-3, device="cpu", optimizer=None):
    net.to(device).train()
    opt = optimizer or torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4)
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
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite AlphaZero loss")
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append((policy_loss.item(), value_loss.item()))
    net.to("cpu").eval()
    return tuple(np.mean(losses, axis=0))


def evaluate(net, encoder, opponent_cls, n_games, num_simulations, seed=0,
             turn_cap=400):
    """Actual wins only. Cutoff leads are not wins."""
    wins = 0
    colors = list(encoder.colors)
    for i in range(n_games):
        az_color = colors[i % len(colors)]
        players = [AZAgent(az_color, net, encoder, num_simulations, seed=seed + i)]
        players += [opponent_cls(c) for c in colors if c != az_color]
        game = Game(players, seed=seed + i)
        while game.winning_color() is None and game.state.num_turns < turn_cap:
            game.play_tick()
        if game.winning_color() == az_color:
            wins += 1
    return wins / n_games


def save_checkpoint(net, encoder, path, training=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": net.state_dict(),
        "num_features": encoder.num_features,
        "num_actions": encoder.num_actions,
        "num_values": net.num_values,
        "colors": [c.value for c in encoder.colors],
    }
    if training is not None:
        payload["training"] = training
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


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
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--eval-every", type=int, default=5)
    parser.add_argument("--curriculum", default=None, help="comma-separated turn caps, equal iteration stages")
    parser.add_argument("--final-eval-games", type=int, default=0)
    args = parser.parse_args()
    if min(args.iterations, args.games, args.sims, args.train_steps, args.eval_every, args.eval_sims, args.buffer, args.eval_games) <= 0 or args.hours <= 0:
        parser.error("Training budgets must be positive")
    if args.players == 2 and args.eval_games % 2:
        parser.error("Development evaluation requires an even game count")
    if args.final_eval_games < 0 or (args.final_eval_games and (args.players != 2 or args.final_eval_games % 2)):
        parser.error("Final paired evaluation requires two players and an even game count")
    caps = [int(c) for c in args.curriculum.split(",")] if args.curriculum else [args.turn_cap]
    if min(caps) <= 0:
        parser.error("Turn caps must be positive")
    out = Path(args.out)
    if out.exists() and (list(out.glob("*.pt")) or (out / "manifest.json").exists()):
        parser.error("Choose a fresh --out directory; existing runs are never overwritten")

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if args.resume:
        net, encoder = load_checkpoint(Path(args.resume))
        if len(encoder.colors) != args.players:
            parser.error("Checkpoint player count does not match --players")
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

    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4)
    start_iteration, elapsed_before = 0, 0.0
    resume_kind = "new"
    initial_checkpoint = args.resume
    if args.resume:
        checkpoint = torch.load(args.resume, weights_only=False)
        state = checkpoint.get("training")
        resume_kind = "full_state" if state else "weights_only"
        if state:
            initial_checkpoint = state.get("initial_checkpoint", state["config"].get("resume"))
            for key in ("seed", "games", "sims", "train_steps", "buffer", "curriculum", "iterations", "turn_cap", "temp_moves", "pool", "hours", "eval_games", "eval_sims", "eval_every", "final_eval_games"):
                if state["config"].get(key) != vars(args).get(key):
                    parser.error(f"Full resume must preserve {key}")
            buffer.extend(state["replay"])
            optimizer.load_state_dict(state["optimizer"])
            torch.set_rng_state(state["torch_rng"])
            random.setstate(state["python_rng"])
            pool_rng.setstate(state["pool_rng"])
            start_iteration = state["iteration"]
            elapsed_before = state["elapsed_seconds"]
    config = {**vars(args), "resume_kind": resume_kind, "evaluation": "actual_outcome_v1",
              "training_reward": "vp_leader_at_curriculum_cap", "optimizer": "persistent_adam",
              "caps": caps, "interval_games": args.eval_games}
    if args.resume:
        import hashlib
        config["initial_checkpoint_sha256"] = hashlib.sha256(Path(args.resume).read_bytes()).hexdigest()
    writer = RunWriter(out, "alphazero", config, title=out.name)
    started = time.monotonic()
    deadline = started + max(0, args.hours * 3600 - elapsed_before)
    final_path = None
    completed_iteration = start_iteration
    try:
        for it in range(start_iteration + 1, args.iterations + 1):
            if time.monotonic() >= deadline:
                break
            t0 = time.monotonic()
            cap = caps[min((it - 1) * len(caps) // args.iterations, len(caps) - 1)]
            turns, diagnostics = [], []
            for g in range(args.games):
                if time.monotonic() >= deadline:
                    break
                opponents = {}
                if pool_specs:
                    from catan_rl.benchmark import resolve_factory
                    for color in list(encoder.colors)[1:]:
                        if pool_rng.random() < .5:
                            opponents[color] = resolve_factory(pool_rng.choice(pool_specs), seed=args.seed + it * 10000 + g)(color)
                detail = {}
                samples, winner, num_turns = self_play_game(
                    net, encoder, args.sims, seed=args.seed + it * 10000 + g,
                    temp_moves=args.temp_moves, turn_cap=cap,
                    opponents=opponents, diagnostics=detail)
                buffer.extend(samples)
                turns.append(num_turns)
                diagnostics.append(detail)
            if not turns:
                break
            if not buffer:
                raise RuntimeError("No non-forced decisions collected")
            sp_time = time.monotonic() - t0
            ploss, vloss = train_steps(net, buffer, args.train_steps, optimizer=optimizer)
            final_path = out / f"iter{it:03d}.pt"
            save_checkpoint(net, encoder, final_path)
            values = {"step": it, "policy_loss": ploss, "value_loss": vloss,
                      "selfplay_seconds": sp_time, "avg_turns": float(np.mean(turns)),
                      "turn_cap": cap, "buffer_size": len(buffer), "selfplay_games": len(turns),
                      "selfplay_timeout_rate": sum(d["cutoff"] for d in diagnostics) / len(diagnostics)}
            if it % args.eval_every == 0 or it == args.iterations:
                if len(encoder.colors) == 2:
                    from catan_rl.evaluation import play_episode
                    outcomes = []
                    for j in range(args.eval_games):
                        if j % 2 == 0 and time.monotonic() >= deadline:
                            break
                        episode = play_episode(f"az:{final_path}:{args.eval_sims}", "weighted",
                            80000 + j // 2, j % 2, 400, 8000, f"{it:03d}-{j:04d}")
                        writer.episode(episode)
                        outcomes.append(episode["outcome"])
                    if outcomes:
                        values.update(win_rate=outcomes.count("win") / len(outcomes),
                                      timeout_rate=outcomes.count("timeout") / len(outcomes),
                                      games=len(outcomes), evaluation_step=it,
                                      evaluation_complete=len(outcomes) == args.eval_games)
                else:
                    values["win_rate"] = evaluate(net, encoder, WeightedRandomPlayer,
                                                  args.eval_games, args.eval_sims, seed=80000)
            values["iteration_seconds"] = time.monotonic() - t0
            values["elapsed_seconds"] = elapsed_before + time.monotonic() - started
            writer.event("metrics", **values)
            save_checkpoint(net, encoder, out / "resume.pt", training={
                "iteration": it, "optimizer": optimizer.state_dict(), "replay": list(buffer),
                "python_rng": random.getstate(), "torch_rng": torch.get_rng_state(),
                "pool_rng": pool_rng.getstate(), "elapsed_seconds": values["elapsed_seconds"],
                "initial_checkpoint": initial_checkpoint,
                "config": vars(args)})
            completed_iteration = it
            print(f"iter {it}: loss {ploss:.3f}/{vloss:.3f}, cap {cap}, "
                  f"selfplay timeouts {values['selfplay_timeout_rate']:.0%}, "
                  f"actual win rate {values.get('win_rate', 'not evaluated')}", flush=True)
        if final_path is None:
            writer.finish("budget_exhausted", notes="No new iteration completed")
            return
        writer.finish("completed" if completed_iteration == args.iterations else "budget_exhausted",
                      final_checkpoint=final_path.name)
    except BaseException as exc:
        writer.finish("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    if args.final_eval_games:
        from catan_rl.evaluation import evaluate as evaluate_run
        evaluations = [("search", f"az:{final_path}:{args.eval_sims}"), ("policy", f"policy:{final_path}")]
        if initial_checkpoint:
            evaluations += [("starting-search", f"az:{initial_checkpoint}:{args.eval_sims}"),
                            ("starting-policy", f"policy:{initial_checkpoint}")]
        for name, agent in evaluations:
            evaluate_run(agent, "weighted", args.final_eval_games,
                         out.parent / f"{out.name}-final-{name}", seed=90000,
                         title=f"{out.name} · final {name}")


if __name__ == "__main__":
    main()
