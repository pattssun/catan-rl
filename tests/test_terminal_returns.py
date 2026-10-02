from types import SimpleNamespace
import random

import pytest
from catanatron import Color, Game
from catanatron.models.enums import Action, ActionType
from catanatron.players.value import ValueFunctionPlayer
from catan_rl.mcts import MCTS, Node, RolloutResult, backed_up_value


class Chain:
    def __init__(self, depth=0, winner=Color.RED, length=3):
        self.depth, self.winner, self.length = depth, winner, length
        color = Color.RED if depth % 2 == 0 else Color.BLUE
        self.state = SimpleNamespace(num_turns=0, current_color=lambda: color)
        self.playable_actions = [Action(color, ActionType.END_TURN, None)]

    def copy(self):
        return Chain(self.depth, self.winner, self.length)

    def execute(self, action, **kwargs):
        assert action in self.playable_actions
        self.__init__(self.depth + 1, self.winner, self.length)

    def winning_color(self):
        return self.winner if self.depth >= self.length else None


def test_backup_arithmetic_both_perspectives_and_neutral_cutoff():
    # Two actions in the tree and one rollout action after the chosen action.
    outcome = RolloutResult(Color.RED, 1, False)
    assert backed_up_value(outcome, Color.RED, 2, .5) == .5625
    assert backed_up_value(outcome, Color.BLUE, 2, .5) == .4375
    assert backed_up_value(RolloutResult(Color.RED, 0, False), Color.RED, 0, .5) == 1
    assert backed_up_value(RolloutResult(None, 5, True), Color.RED, 2, .5) == .5


@pytest.mark.parametrize("simulations", [1, 2, 3, 10])
@pytest.mark.parametrize("winner,expected", [(Color.RED, .625), (Color.BLUE, .375)])
def test_discount_counts_tree_and_rollout_edges_once(simulations, winner, expected):
    game = Chain(winner=winner)
    search = MCTS(seed=0, terminal_returns=True, discount=.5, rollout_action_cap=10)
    stats = search.search_statistics(game, simulations)
    assert stats[game.playable_actions[0]]["value"] == expected
    assert stats[game.playable_actions[0]]["visits"] == simulations
    assert search.rollout_counts["cutoff"] == 0


def test_action_cap_stops_a_game_that_never_advances_turns():
    game = Chain(length=10000)
    search = MCTS(seed=0, terminal_returns=True, discount=.995, rollout_action_cap=7)
    outcome = search._rollout(Node(game, None, random.Random(1)))
    assert outcome == RolloutResult(None, 7, True)


def test_real_engine_immediate_win_and_cutoff_lead():
    game = Game([ValueFunctionPlayer(Color.RED), ValueFunctionPlayer(Color.BLUE)], seed=3)
    for _ in range(8000):
        before = game.copy()
        record = game.play_tick()
        if game.winning_color() is not None:
            break
    else:
        pytest.fail("Fixture never reached an engine-declared win")
    winner = game.winning_color()
    winning = before.copy()
    winning.execute(record.action)
    terminal = MCTS(terminal_returns=True, discount=.995)
    outcome = terminal._rollout(Node(winning, winner, random.Random(0)))
    assert outcome == RolloutResult(winner, 0, False)
    assert backed_up_value(outcome, winner, discount=.995) == 1
    assert backed_up_value(outcome, next(c for c in game.state.colors if c != winner), discount=.995) == 0
    passed = before.copy()
    passed.execute(next(a for a in passed.playable_actions if a.action_type == ActionType.END_TURN))
    assert passed.winning_color() is None
    candidate = MCTS(horizon=0, terminal_returns=True, discount=.995)
    control = MCTS(horizon=0)
    candidate_result = candidate._rollout(Node(passed, winner, random.Random(0)))
    control_result = control._rollout(Node(passed, winner, random.Random(0)))
    assert candidate_result == RolloutResult(None, 0, True)
    assert control_result == RolloutResult(winner, 0, True)
    assert backed_up_value(candidate_result, winner) == .5
    assert backed_up_value(control_result, winner) == 1
