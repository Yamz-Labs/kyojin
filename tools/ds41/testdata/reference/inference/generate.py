import os
import json
import sys
from argparse import ArgumentParser
from typing import List

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from safetensors.torch import load_model

from model import Transformer, ModelArgs

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(current_dir, "../encoding")))

from encoding import (
    encode_case,
    encode_messages,
    load_cases,
    parse_message_from_completion_text,
    parse_tagged_text,
    to_json,
)
from image_processor import TEXT, prepare_vl_inputs


@torch.inference_mode()
def generate(
    model: Transformer,
    prompt_tokens: List[List[int]],
    max_new_tokens: int,
    eos_id: int,
    prompt_token_types: List[List[int]] | None = None,
    images=None,
) -> List[List[int]]:
    """Batch generation with right-padded prompts.

    The first forward pass processes [:min_prompt_len] tokens (prefill phase).
    Subsequent passes generate one token at a time (decode phase). For positions
    still within a prompt, the ground-truth token overrides the model's prediction.

    `prompt_token_types` and `images` come from image_processor.prepare_vl_inputs. Image spans are
    only visible to the prefill pass, so they must end before the shortest prompt does.
    """
    prompt_lens = [len(t) for t in prompt_tokens]
    assert max(prompt_lens) <= model.max_seq_len, (
        f"Prompt length exceeds model maximum sequence length (max_seq_len={model.max_seq_len})"
    )
    total_len = min(model.max_seq_len, max_new_tokens + max(prompt_lens))
    tokens = torch.full((len(prompt_tokens), total_len), -1, dtype=torch.long)
    for i, t in enumerate(prompt_tokens):
        tokens[i, : len(t)] = torch.tensor(t, dtype=torch.long)

    token_types = None
    if images is not None:
        token_types = torch.full((len(prompt_tokens), total_len), TEXT, dtype=torch.long)
        for i, types in enumerate(prompt_token_types):
            token_types[i, : len(types)] = torch.tensor(types, dtype=torch.long)
        for sample in images:
            for img in sample or ():
                assert img.start + img.types.numel() <= min(prompt_lens), "image spans must fit in the prefill chunk"

    prev_pos = 0
    finished = torch.tensor([False] * len(prompt_tokens))
    prompt_mask = tokens != -1
    for cur_pos in range(min(prompt_lens), total_len):
        with_images = images is not None and prev_pos == 0
        next_token = model.forward(
            tokens[:, prev_pos:cur_pos],
            prev_pos,
            images=images if with_images else None,
            token_types=token_types[:, prev_pos:cur_pos] if with_images else None,
        )[0]
        next_token = torch.where(prompt_mask[:, cur_pos], tokens[:, cur_pos], next_token)
        tokens[:, cur_pos] = next_token
        finished |= torch.logical_and(~prompt_mask[:, cur_pos], next_token == eos_id)
        prev_pos = cur_pos
        if finished.all():
            break
    completion_tokens = []
    for i, toks in enumerate(tokens.tolist()):
        toks = toks[prompt_lens[i] : prompt_lens[i] + max_new_tokens]
        if eos_id in toks:
            toks = toks[: toks.index(eos_id)]
        completion_tokens.append(toks)
    return completion_tokens


def prepare_case(case, thinking_mode, tokenizer, args):
    """Encode one message case and expand any image placeholders."""
    if case.get("context"):
        raise ValueError("Standalone inference does not support context without a prefilled KV cache")
    prompt, image_records = encode_case(case, thinking_mode)
    tokens, token_types, images = prepare_vl_inputs(prompt, image_records, tokenizer, args)
    return prompt, tokens, token_types, images


def main(
    ckpt_path: str,
    config: str,
    input_file: str = "",
    interactive: bool = True,
    max_new_tokens: int = 100,
    temperature: float = 1.0,
    thinking_mode: str = "chat",
) -> None:
    world_size = int(os.getenv("WORLD_SIZE", "1"))
    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if world_size > 1:
        dist.init_process_group("nccl")
    global print
    if rank != 0:
        print = lambda *_, **__: None
    torch.cuda.set_device(local_rank)
    torch.cuda.memory._set_allocator_settings("expandable_segments:True")
    torch.set_default_dtype(torch.bfloat16)
    torch.set_num_threads(8)
    torch.manual_seed(33377335)
    with open(config) as f:
        args = ModelArgs(**json.load(f))
        args.temperature = temperature
    if interactive:
        args.max_batch_size = 1
        args.max_seq_len = 64 * 1024
    print(args)
    tokenizer = AutoTokenizer.from_pretrained(ckpt_path)
    print("build model")
    with torch.device("cuda"):
        model = Transformer(args, tokenizer)
    print("load model")
    load_model(model, os.path.join(ckpt_path, f"model{rank}-mp{world_size}.safetensors"))
    torch.set_default_device("cuda")
    print("I'm DeepSeek 👋")

    if interactive:
        messages = []
        while True:
            if world_size == 1:
                prompt = input(">>> ")
            elif rank == 0:
                prompt = input(">>> ")
                objects = [prompt]
                dist.broadcast_object_list(objects, 0)
            else:
                objects = [None]
                dist.broadcast_object_list(objects, 0)
                prompt = objects[0]
            if prompt == "/exit":
                break
            elif prompt == "/clear":
                messages.clear()
                continue
            messages.append({"role": "user", "content": prompt})
            prompt_tokens = tokenizer.encode(encode_messages(messages, thinking_mode=thinking_mode))
            completion_tokens = generate(model, [prompt_tokens], max_new_tokens, tokenizer.eos_token_id)
            completion = tokenizer.decode(completion_tokens[0])
            print(completion)
            messages.append(parse_message_from_completion_text(completion, thinking_mode=thinking_mode))
    else:
        if input_file.endswith(".json"):
            # Harmony input: a JSON file with one or more OpenAI-format cases
            # ({"messages": [...], "tools": [...]} or a bare message list).
            cases = load_cases(input_file)
            raw_prompts = [to_json(case["messages"]) for case in cases]
        else:
            # Plain-text input: blank-line-separated prompts, optionally with
            # <image>path</image> tags.
            with open(input_file) as f:
                raw_prompts = f.read().rstrip("\n").split("\n\n")
            cases = [{"messages": [{"role": "user", "content": parse_tagged_text(prompt)}]} for prompt in raw_prompts]

        prompt_tokens, prompt_token_types, images = [], [], []
        for case in cases:
            _, tokens, token_types, image_inputs = prepare_case(case, thinking_mode, tokenizer, args)
            prompt_tokens.append(tokens)
            prompt_token_types.append(token_types)
            images.append(image_inputs)

        if any(images):
            # image spans must be prefilled in one chunk, so VL prompts are generated one at a time
            completion_tokens = [
                generate(model, [tok], max_new_tokens, tokenizer.eos_token_id, [types], [image])[0]
                for tok, types, image in zip(prompt_tokens, prompt_token_types, images)
            ]
        else:
            completion_tokens = generate(model, prompt_tokens, max_new_tokens, tokenizer.eos_token_id)
        completions = tokenizer.batch_decode(completion_tokens)
        for raw_prompt, completion in zip(raw_prompts, completions):
            print("Prompt:", raw_prompt)
            print("Completion:", completion)
            print()

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--ckpt-path", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--input-file", type=str, default="")
    parser.add_argument("--interactive", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--thinking-mode", type=str, default="chat", choices=["chat", "thinking"])
    args = parser.parse_args()
    assert args.input_file or args.interactive, "Either input-file or interactive mode must be specified"
    main(
        args.ckpt_path,
        args.config,
        args.input_file,
        args.interactive,
        args.max_new_tokens,
        args.temperature,
        args.thinking_mode,
    )
