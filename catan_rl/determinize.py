"""Stage 3: hidden information via determinization.

In 1v1 Catan the genuinely hidden state is the opponent's unplayed dev
cards and the deck order — resource hands are deducible by card
counting (every steal involves you). Perfect-info search cheats on
exactly that: the copied state carries the opponent's true dev hand
(so simulated opponents play knights they "shouldn't know" they have)
and the true deck order (so BUY_DEVELOPMENT_CARD "draws" are known in
advance). The net's features were already honest — enemy hands appear
only as counts — the tree wasn't.

Determinization: sample K worlds consistent with the root player's
information set (reshuffle the unseen pool = deck + opponent's unplayed
devs), search each world independently, sum root visit counts.
"""

import math
import random
from collections import Counter

from catanatron.models.enums import DEVELOPMENT_CARDS, VICTORY_POINT
from catanatron.models.player import Player
from catanatron.state_functions import player_key

from catan_rl.mcts import MCTS, immediate_win


def determinize(game, pov_color, rng):
    """A copy of `game` with hidden state resampled from pov's info set."""
    world = game.copy()
    state = world.state
    pool = list(state.development_listdeck)
    hand_sizes = {}
    for color in state.colors:
        if color == pov_color:
            continue
        key = player_key(state, color)
        hand_sizes[color] = 0
        for card in DEVELOPMENT_CARDS:
            count = state.player_state[f"{key}_{card}_IN_HAND"]
            pool.extend([card] * count)
            hand_sizes[color] += count
    rng.shuffle(pool)

    for color, n in hand_sizes.items():
        key = player_key(state, color)
        new_hand, pool = pool[:n], pool[n:]
        counts = Counter(new_hand)
        old_vp = state.player_state[f"{key}_{VICTORY_POINT}_IN_HAND"]
        for card in DEVELOPMENT_CARDS:
            state.player_state[f"{key}_{card}_IN_HAND"] = counts[card]
            flag = f"{key}_{card}_OWNED_AT_START"
            if flag in state.player_state:
                state.player_state[flag] = counts[card] > 0
        state.player_state[f"{key}_ACTUAL_VICTORY_POINTS"] += (
            counts[VICTORY_POINT] - old_vp
        )
    state.development_listdeck = pool
    return world


class DeterminizedMCTSAgent(Player):
    """Stage 1 MCTS run over K sampled worlds, root visits summed."""

    def __init__(self, color, num_simulations=100, worlds=8,
                 rollout="random", horizon=None, seed=None):
        super().__init__(color)
        self.num_simulations = num_simulations
        self.worlds = worlds
        self.rng = random.Random(seed)
        self.mcts = MCTS(rollout=rollout, horizon=horizon, seed=seed)

    def decide(self, game, playable_actions):
        if len(playable_actions) == 1:
            return playable_actions[0]
        win = immediate_win(game, playable_actions, self.color)
        if win is not None:
            return win
        sims_per_world = math.ceil(self.num_simulations / self.worlds)
        totals = Counter()
        for _ in range(self.worlds):
            world = determinize(game, self.color, self.rng)
            totals.update(self.mcts.search_visits(world, sims_per_world))
        return max(totals, key=totals.get)

    def __repr__(self):
        return (f"DeterminizedMCTSAgent({self.num_simulations}sims,"
                f"{self.worlds}worlds):{self.color.value}")
