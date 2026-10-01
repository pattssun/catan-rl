import math

import pytest
import torch

from catan_rl.grpo import group_advantages, grpo_loss
from catan_rl.llm import completion_mask, parse_action, prompt_for, sample_states


def test_equal_rewards_have_no_policy_signal():
    assert torch.equal(group_advantages(torch.tensor([[1., 1., 1., 1.]])), torch.zeros(1, 4))


def test_group_normalization_is_per_prompt():
    a = group_advantages(torch.tensor([[0., 2.], [100., 102.]]))
    assert a.tolist()[0] == pytest.approx([-1 / 1.0001, 1 / 1.0001])
    assert torch.equal(a[0], a[1])


def test_categorical_policy_gradient_from_independent_derivative():
    logits = torch.tensor([math.log(.25), math.log(.75)], requires_grad=True)
    logp = logits.log_softmax(0).reshape(2, 1)
    advantage = torch.tensor([-1., 1.])
    loss, metrics = grpo_loss(logp, logp.detach(), logp.detach(), advantage, torch.ones(2, 1), beta=0)
    loss.backward()
    # For equally sampled opposite advantages, dL/d(logit_0) = 1/2.
    assert logits.grad.tolist() == pytest.approx([.5, -.5])
    assert metrics["kl"] == 0
    assert metrics["clip_fraction"] == 0


@pytest.mark.parametrize("advantage,expected_loss,expected_grad", [(1., -1.2, 0.), (-1., 2., 2.)])
def test_clipping_handles_both_advantage_signs(advantage, expected_loss, expected_grad):
    new = torch.tensor([[math.log(.4)]], requires_grad=True)
    old = torch.tensor([[math.log(.2)]])
    loss, _ = grpo_loss(new, old, new.detach(), torch.tensor([advantage]), torch.ones(1, 1), beta=0)
    loss.backward()
    assert loss.item() == pytest.approx(expected_loss)
    assert new.grad.item() == pytest.approx(expected_grad)


def test_padding_has_no_gradient_and_reference_is_frozen():
    logp = torch.tensor([[-1., -2.]], requires_grad=True)
    ref = torch.tensor([[-1.2, -2.5]], requires_grad=True)
    old = logp.detach().clone().requires_grad_()
    loss, _ = grpo_loss(logp, old, ref, torch.tensor([1.]), torch.tensor([[1., 0.]]))
    loss.backward()
    assert logp.grad[0, 1] == 0
    assert ref.grad is None
    assert old.grad is None


def test_eos_is_scored_once():
    mask = completion_mask(torch.tensor([[2, 9, 9, 9], [1, 2, 3, 4]]), eos=9)
    assert mask.tolist() == [[1, 1, 0, 0], [1, 1, 1, 1]]


@pytest.mark.parametrize("text", ["A01", "A-1", "A3 then A2", "<answer>A2</answer>", "A999", "A1\nA2", "1", "A1.0"])
def test_parser_rejects_ambiguous_or_out_of_range_output(text):
    assert parse_action(text, 10) is None


def test_parser_accepts_exact_legal_id():
    assert parse_action(" A2\n", 3) == 2
    assert parse_action("A3", 3) is None


def test_prompt_hides_deck_order():
    game = sample_states(1)[0]
    first = prompt_for(game)
    game.state.development_listdeck.reverse()
    assert prompt_for(game) == first
    assert "Legal actions:" in first


def test_prompt_port_access_and_robber_tile_are_unambiguous():
    import json
    from catanatron.models.map import Port, EdgeRef
    game = sample_states(1)[0]
    state = json.loads(prompt_for(game).splitlines()[1])
    expected = {}
    for tile in game.state.board.map.tiles.values():
        if isinstance(tile, Port):
            edge = tile.edges[EdgeRef(tile.direction.value)]
            expected.setdefault(tile.resource, set()).update(edge)
    assert {p["resource"]: set(p["nodes"]) for p in state["ports"]} == expected
    assert sum(t["coordinate"] == state["public"]["robber"] for t in state["tiles"]) == 1
