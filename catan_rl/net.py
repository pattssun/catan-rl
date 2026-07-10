"""Policy+value network and the game<->tensor encoding (Stage 2).

The encoding consumes Catanatron's feature extraction and fixed action
enumeration (state representation is the engine's job); the network,
search, and training are ours.
"""

import numpy as np
import torch
import torch.nn as nn

from catanatron.features import create_sample_vector, get_feature_ordering
from catanatron.gym.envs.action_space import get_action_array
from catanatron.models.enums import ActionType


class Encoder:
    def __init__(self, colors, map_type="BASE"):
        self.colors = tuple(colors)
        self.num_features = len(get_feature_ordering(len(colors)))
        self.action_array = get_action_array(self.colors, map_type)
        self.num_actions = len(self.action_array)
        self.action_index = {key: i for i, key in enumerate(self.action_array)}

    def encode(self, game, color):
        return np.array(create_sample_vector(game, color), dtype=np.float32)

    def action_to_index(self, action):
        value = action.value
        if action.action_type == ActionType.BUILD_ROAD:
            value = tuple(sorted(value))
        elif isinstance(value, list):
            value = tuple(value)
        return self.action_index[(action.action_type, value)]


class PolicyValueNet(nn.Module):
    """num_values=1: scalar value, zero-sum 1v1. num_values=n: one value
    per seat in relative order (index 0 = the perspective player) — with
    3+ players there is no single adversary to negate against, so the
    search needs every seat's value, not v and -v."""

    def __init__(self, num_features, num_actions, hidden=512, num_values=1):
        super().__init__()
        self.num_values = num_values
        self.trunk = nn.Sequential(
            nn.Linear(num_features, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.policy_head = nn.Linear(hidden, num_actions)
        self.value_head = nn.Sequential(nn.Linear(hidden, 64), nn.ReLU(),
                                        nn.Linear(64, num_values), nn.Tanh())

    def forward(self, x):
        h = self.trunk(x)
        values = self.value_head(h)
        if self.num_values == 1:
            values = values.squeeze(-1)
        return self.policy_head(h), values

    @torch.no_grad()
    def predict(self, features, legal_indices):
        """Single-state inference: masked priors over legal actions + value
        (float if num_values == 1, else relative-seat vector)."""
        x = torch.from_numpy(features).unsqueeze(0)
        logits, value = self(x)
        legal_logits = logits[0, legal_indices]
        priors = torch.softmax(legal_logits, dim=0).numpy()
        if self.num_values == 1:
            return priors, value.item()
        return priors, value[0].numpy()
