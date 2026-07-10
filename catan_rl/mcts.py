"""Hand-written MCTS for Catan (Stage 1: 1v1, no player trading).

UCT over decision nodes. Stochastic actions — dice ROLLs, dev-card
draws, robber steals — are chance edges: each traversal samples an
outcome and descends into the child keyed by it, so action values
converge to expectations over outcomes (sampled expectimax) instead of
a single determinized future. Rollouts are uniform-random with no
domain evaluation; the engine supplies only transitions and legality.
"""

import math
import random

from catanatron.game import TURNS_LIMIT, Game
from catanatron.models.enums import ActionRecord, ActionType
from catanatron.models.player import Player
from catanatron.state_functions import get_actual_victory_points

DICE_PAIRS = [(a, b) for a in range(1, 7) for b in range(1, 7)]


def is_chance(action):
    return (
        action.action_type == ActionType.ROLL
        or action.action_type == ActionType.BUY_DEVELOPMENT_CARD
        or (action.action_type == ActionType.MOVE_ROBBER and action.value[1] is not None)
    )


def immediate_win(game, playable_actions, color):
    """Decisive-move check. Horizon-capped rollouts scored by VP leader
    make every action from a winning position look equally good (the
    leader wins every rollout by tiebreak), so the search can dither at
    the doorstep; taking a certain win is game logic, not evaluation."""
    for action in playable_actions:
        if is_chance(action):
            continue
        probe = game.copy()
        probe.execute(action, validate_action=False)
        if probe.winning_color() == color:
            return action
    return None


class Node:
    __slots__ = ("game", "mover", "N", "W", "children", "untried")

    def __init__(self, game, mover, rng):
        self.game = game
        self.mover = mover  # color whose action led here; None at root
        self.N = 0
        self.W = 0.0
        self.children = {}  # action -> {outcome_key: Node}
        self.untried = [] if self.is_terminal else list(game.playable_actions)
        rng.shuffle(self.untried)

    @property
    def is_terminal(self):
        return (
            self.game.winning_color() is not None
            or self.game.state.num_turns >= TURNS_LIMIT
        )


class MCTS:
    def __init__(self, c=math.sqrt(2), rollout="random", horizon=None, seed=None):
        self.c = c
        self.rollout = rollout
        self.horizon = horizon
        self.rng = random.Random(seed)

    def search(self, game, num_simulations):
        visits = self.search_visits(game, num_simulations)
        return max(visits, key=visits.get)

    def search_visits(self, game, num_simulations):
        """Returns {action: visit_count} at the root."""
        root = Node(game.copy(), None, self.rng)
        for _ in range(num_simulations):
            path = [root]
            node = root

            while not node.untried and not node.is_terminal:
                node = self._step(node, self._select_action(node))
                path.append(node)

            if node.untried:
                node = self._step(node, node.untried.pop())
                path.append(node)

            winner = self._rollout(node)
            for n in path:
                n.N += 1
                if n.mover is not None:
                    n.W += 1.0 if winner == n.mover else 0.5 if winner is None else 0.0

        return {
            action: sum(c.N for c in bucket.values())
            for action, bucket in root.children.items()
        }

    def _select_action(self, node):
        log_n = math.log(node.N)
        best_action, best_ucb = None, -1.0
        for action, bucket in node.children.items():
            n_a = sum(c.N for c in bucket.values())
            w_a = sum(c.W for c in bucket.values())
            ucb = w_a / n_a + self.c * math.sqrt(log_n / n_a)
            if ucb > best_ucb:
                best_action, best_ucb = action, ucb
        return best_action

    def _step(self, node, action):
        """Descend through an edge, sampling the outcome if it's a chance edge."""
        bucket = node.children.setdefault(action, {})
        mover = node.game.state.current_color()

        if action.action_type == ActionType.ROLL:
            # sample dice ourselves so existing outcome-children cost no copy
            pair = self.rng.choice(DICE_PAIRS)
            key = pair[0] + pair[1]
            child = bucket.get(key)
            if child is None:
                game = node.game.copy()
                game.execute(action, action_record=ActionRecord(action, pair))
                child = bucket[key] = Node(game, mover, self.rng)
            return child

        if is_chance(action):  # dev-card draw, robber steal: let the engine sample
            game = node.game.copy()
            record = game.execute(action)
            child = bucket.get(record.result)
            if child is None:
                child = bucket[record.result] = Node(game, mover, self.rng)
            return child

        child = bucket.get(None)
        if child is None:
            game = node.game.copy()
            game.execute(action)
            child = bucket[None] = Node(game, mover, self.rng)
        return child

    def _rollout(self, node):
        game = node.game
        if node.is_terminal:
            return game.winning_color()
        game = game.copy()
        end = TURNS_LIMIT
        if self.horizon is not None:
            end = min(end, game.state.num_turns + self.horizon)
        while game.winning_color() is None and game.state.num_turns < end:
            game.execute(self._rollout_action(game), validate_action=False)
        winner = game.winning_color()
        if winner is None:  # horizon/turn-limit cutoff: most VPs wins, tie = draw
            vps = {
                color: get_actual_victory_points(game.state, color)
                for color in game.state.colors
            }
            top = max(vps.values())
            leaders = [color for color, vp in vps.items() if vp == top]
            winner = leaders[0] if len(leaders) == 1 else None
        return winner

    def _rollout_action(self, game):
        actions = game.playable_actions
        if self.rollout == "weighted":
            weights = [
                10000 if a.action_type == ActionType.BUILD_CITY
                else 1000 if a.action_type == ActionType.BUILD_SETTLEMENT
                else 100 if a.action_type == ActionType.BUY_DEVELOPMENT_CARD
                else 1
                for a in actions
            ]
            return self.rng.choices(actions, weights)[0]
        return self.rng.choice(actions)


class MCTSAgent(Player):
    def __init__(self, color, num_simulations=100, c=math.sqrt(2),
                 rollout="random", horizon=None, seed=None):
        super().__init__(color)
        self.num_simulations = num_simulations
        self.mcts = MCTS(c=c, rollout=rollout, horizon=horizon, seed=seed)

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        win = immediate_win(game, playable_actions, self.color)
        if win is not None:
            return win
        return self.mcts.search(game, self.num_simulations)

    def __repr__(self):
        return f"MCTSAgent({self.num_simulations}sims,{self.mcts.rollout}):{self.color.value}"
