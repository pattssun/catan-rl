"""Information-limited Catan prompts and a measured MPS preflight."""

import argparse
import json
import random
import re
import time

import torch
from catanatron import Color, Game
from catanatron.models.enums import DEVELOPMENT_CARDS, RESOURCES
from catanatron.players.value import ValueFunctionPlayer
from catanatron.state_functions import player_key

from catan_rl.telemetry import RunWriter, action_json, public_state

MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"


def prompt_for(game):
    color = game.state.current_color()
    key = player_key(game.state, color)
    state = public_state(game)
    own = {r: game.state.player_state[f"{key}_{r}_IN_HAND"] for r in RESOURCES + DEVELOPMENT_CARDS}
    tiles = []
    ports = [{"resource": resource, "nodes": sorted(nodes)}
             for resource, nodes in game.state.board.map.port_nodes.items()]
    for coordinate, tile in game.state.board.map.tiles.items():
        if hasattr(tile, "number"):
            tiles.append({"coordinate": list(coordinate),
                          "resource": tile.resource, "number": tile.number,
                          "nodes": list(tile.nodes.values())})
    observation = {"you": color.value, "target_vp": game.vps_to_win, "hand": own,
                   "public": state, "tiles": tiles, "ports": ports}
    actions = [action_json(a) for a in game.playable_actions]
    return ("Choose one legal Catan action. Reach 10 victory points. Roads connect settlements; "
            "settlements and cities produce resources on their tile numbers. "
            "Reply with only its ID, such as A3. Do not explain.\n"
            + json.dumps(observation, separators=(",", ":")) + "\nLegal actions:\n"
            + "\n".join(f"A{i} {a['type']} {json.dumps(a['value'],separators=(',',':'))}"
                         for i, a in enumerate(actions)))


def parse_action(text, count):
    match = re.fullmatch(r"A(0|[1-9][0-9]*)", text.strip())
    if not match:
        return None
    index = int(match[1])
    return index if index < count else None


def sample_states(count=4, seed=73000):
    states = []
    for i in range(count * 10):
        game = Game([ValueFunctionPlayer(Color.RED), ValueFunctionPlayer(Color.BLUE)], seed=seed + i)
        target = [0, 20, 60, 100][i % 4]
        ticks = 0
        while game.winning_color() is None and game.state.num_turns < target and ticks < 2000:
            game.play_tick()
            ticks += 1
        while game.winning_color() is None and len(game.playable_actions) == 1 and ticks < 2000:
            game.play_tick()
            ticks += 1
        if game.winning_color() is None and len(game.playable_actions) > 1:
            states.append(game)
        if len(states) == count:
            return states
    raise RuntimeError("Could not collect enough decision states")


def load_policy(device="mps", revision=None):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import LoraConfig, get_peft_model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=revision)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.float32 if device == "cpu" else torch.float16
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=revision,
                                               torch_dtype=dtype, attn_implementation="sdpa")
    model = get_peft_model(model, LoraConfig(r=8, lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0, bias="none", task_type="CAUSAL_LM"))
    model.to(device).eval()
    return model, tokenizer


def encode_prompt(tokenizer, prompt, device, max_prompt_tokens=2048):
    text = tokenizer.apply_chat_template([{"role": "user", "content": prompt}],
                                        tokenize=False, add_generation_prompt=True)
    tokens = tokenizer(text, return_tensors="pt")
    if tokens.input_ids.shape[1] > max_prompt_tokens:
        raise ValueError(f"Prompt is {tokens.input_ids.shape[1]} tokens; limit {max_prompt_tokens}. No truncation allowed.")
    return {k: v.to(device) for k, v in tokens.items()}


def completion_mask(completion, eos):
    hits = (completion == eos).cumsum(dim=1)
    return ((hits == 0) | ((completion == eos) & (hits == 1))).float()


def token_log_probs(model, sequences, prompt_length, mask):
    attention = torch.cat([torch.ones_like(sequences[:, :prompt_length]), mask.long()], dim=1)
    logits = model(input_ids=sequences, attention_mask=attention, use_cache=False).logits
    logits = logits[:, prompt_length - 1:-1, :].float()
    return logits.log_softmax(dim=-1).gather(-1, sequences[:, prompt_length:].unsqueeze(-1)).squeeze(-1)


def synchronize(device):
    if device == "mps":
        torch.mps.synchronize()


def preflight(directory, count=4, group=4, device="mps", revision=None, backward=False):
    torch.manual_seed(611)
    random.seed(611)
    model, tokenizer = load_policy(device, revision)
    states = sample_states(count)
    writer = RunWriter(directory, "llm_preflight", {
        "model": MODEL_ID, "revision": model.config._commit_hash, "group": group,
        "prompt_version": "catan_action_v2_ports",
        "max_new_tokens": 16, "max_prompt_tokens": 2048, "device": device,
        "optimizer_updates": 0, "backward_probe": backward,
        "purpose": "throughput and format only; no training"},
        title="Qwen 0.5B · gradient preflight" if backward else "Qwen 0.5B · MPS preflight")
    try:
        for i, game in enumerate(states):
            tokens = encode_prompt(tokenizer, prompt_for(game), device)
            synchronize(device)
            start = time.monotonic()
            with torch.no_grad():
                sequences = model.generate(**tokens, num_return_sequences=group, do_sample=True,
                    temperature=1.0, top_p=1.0, top_k=0, max_new_tokens=16,
                    pad_token_id=tokenizer.pad_token_id)
            synchronize(device)
            elapsed = time.monotonic() - start
            completion = sequences[:, tokens["input_ids"].shape[1]:]
            mask = completion_mask(completion, tokenizer.eos_token_id)
            texts = tokenizer.batch_decode(completion, skip_special_tokens=True)
            chosen = [parse_action(t, len(game.playable_actions)) for t in texts]
            probe = {}
            if backward:
                start_backward = time.monotonic()
                model.zero_grad(set_to_none=True)
                for j in range(group):
                    logp = token_log_probs(model, sequences[j:j+1], tokens["input_ids"].shape[1], mask[j:j+1])
                    loss = -(logp * mask[j:j+1]).sum() / mask[j:j+1].sum() / group
                    loss.backward()
                synchronize(device)
                gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
                probe = {"backward_seconds": time.monotonic() - start_backward,
                         "gradients_finite": all(torch.isfinite(g).all().item() for g in gradients),
                         "gradient_max": max(g.abs().max().item() for g in gradients)}
                model.zero_grad(set_to_none=True)
            writer.event("metrics", step=i + 1, prompt_tokens=tokens["input_ids"].shape[1],
                         generated_tokens=int(mask.sum().item()), generation_seconds=elapsed,
                         tokens_per_second=mask.sum().item() / elapsed,
                         valid_rate=sum(c is not None for c in chosen) / group,
                         outputs=texts, chosen_actions=chosen,
                         memory_bytes=torch.mps.driver_allocated_memory() if device == "mps" else None,
                         **probe)
            print(f"prompt {i+1}: {tokens['input_ids'].shape[1]} tokens, {elapsed:.2f}s, {texts}", flush=True)
        writer.finish()
    except BaseException as exc:
        writer.finish("failed", error=f"{type(exc).__name__}: {exc}")
        raise


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True)
    p.add_argument("--prompts", type=int, default=4)
    p.add_argument("--group", type=int, default=4)
    p.add_argument("--device", default="mps", choices=["cpu", "mps"])
    p.add_argument("--revision")
    p.add_argument("--backward", action="store_true", help="Time gradients without updating any weights")
    a = p.parse_args()
    preflight(a.out, a.prompts, a.group, a.device, a.revision, a.backward)
