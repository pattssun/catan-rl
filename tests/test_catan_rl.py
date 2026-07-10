import random
from collections import Counter

import numpy as np
import pytest

from catanatron import Color, Game, RandomPlayer
from catanatron.models.enums import DEVELOPMENT_CARDS
from catanatron.players.value import ValueFunctionPlayer
from catanatron.state_functions import get_actual_victory_points, player_key

from catan_rl.determinize import determinize
from catan_rl.mcts import MCTS, MCTSAgent
from catan_rl.net import Encoder, PolicyValueNet
from catan_rl.pool import elo_update


def midgame(seed=1, min_turns=30):
    game = Game([ValueFunctionPlayer(Color.RED), ValueFunctionPlayer(Color.BLUE)],
                seed=seed)
    while game.state.num_turns < min_turns and game.winning_color() is None:
        game.play_tick()
    return game


def test_mcts_visits_sum_to_simulations():
    game = midgame()
    visits = MCTS(seed=0, horizon=60).search_visits(game, 40)
    assert sum(visits.values()) == 40
    assert set(visits) <= set(game.playable_actions)


def test_mcts_plays_full_game():
    game = Game([MCTSAgent(Color.RED, num_simulations=15, horizon=60, seed=0),
                 RandomPlayer(Color.BLUE)], seed=2)
    game.play()
    assert game.state.num_turns > 0


def test_mcts_takes_winning_move():
    # replay to one decision before a win; the agent must take a winning action
    game = Game([ValueFunctionPlayer(Color.RED), ValueFunctionPlayer(Color.BLUE)],
                seed=3)
    prev = None
    while game.winning_color() is None:
        prev = game.copy()
        game.play_tick()
    color = prev.state.current_color()
    agent = MCTSAgent(color, num_simulations=50, horizon=40, seed=0)
    chosen = agent.decide(prev, prev.playable_actions)
    probe = prev.copy()
    probe.execute(chosen)
    assert probe.winning_color() == color


def test_determinize_conserves_hidden_pool():
    game = midgame(seed=1, min_turns=45)
    state = game.state
    rng = random.Random(0)
    key = player_key(state, Color.BLUE)
    for _ in range(10):
        world = determinize(game, Color.RED, rng)
        ws = world.state
        pool = Counter(state.development_listdeck)
        wpool = Counter(ws.development_listdeck)
        for card in DEVELOPMENT_CARDS:
            pool[card] += state.player_state[f"{key}_{card}_IN_HAND"]
            wpool[card] += ws.player_state[f"{key}_{card}_IN_HAND"]
        assert pool == wpool
        assert get_actual_victory_points(ws, Color.BLUE) - get_actual_victory_points(
            state, Color.BLUE
        ) == (ws.player_state[f"{key}_VICTORY_POINT_IN_HAND"]
              - state.player_state[f"{key}_VICTORY_POINT_IN_HAND"])


def test_determinize_keeps_pov_untouched():
    game = midgame(seed=1, min_turns=45)
    world = determinize(game, Color.RED, random.Random(0))
    key = player_key(game.state, Color.RED)
    for card in DEVELOPMENT_CARDS:
        assert (world.state.player_state[f"{key}_{card}_IN_HAND"]
                == game.state.player_state[f"{key}_{card}_IN_HAND"])


def test_encoder_indexes_all_playable_actions():
    for seed in range(5):
        encoder = Encoder([Color.RED, Color.BLUE])
        game = midgame(seed=seed, min_turns=25)
        for action in game.playable_actions:
            idx = encoder.action_to_index(action)
            assert 0 <= idx < encoder.num_actions


def test_net_predict_shapes():
    encoder = Encoder([Color.RED, Color.BLUE])
    net = PolicyValueNet(encoder.num_features, encoder.num_actions).eval()
    game = midgame()
    feats = encoder.encode(game, game.state.current_color())
    legal = [encoder.action_to_index(a) for a in game.playable_actions]
    priors, value = net.predict(feats, legal)
    assert len(priors) == len(legal)
    assert np.isclose(priors.sum(), 1.0, atol=1e-5)
    assert -1.0 <= value <= 1.0


def test_net_vector_value():
    encoder = Encoder([Color.RED, Color.BLUE, Color.WHITE, Color.ORANGE])
    net = PolicyValueNet(encoder.num_features, encoder.num_actions,
                         num_values=4).eval()
    game = Game([RandomPlayer(c) for c in encoder.colors], seed=1)
    feats = encoder.encode(game, game.state.current_color())
    legal = [encoder.action_to_index(a) for a in game.playable_actions]
    _, value = net.predict(feats, legal)
    assert value.shape == (4,)


def test_elo_zero_sum_and_direction():
    ratings = {"a": 1000.0, "b": 1000.0, "c": 1000.0, "d": 1000.0}
    elo_update(ratings, [("a", 10), ("b", 8), ("c", 5), ("d", 2)])
    assert ratings["a"] > 1000 > ratings["d"]
    assert ratings["a"] > ratings["b"] > ratings["c"] > ratings["d"]
    assert abs(sum(ratings.values()) - 4000.0) < 1e-9
