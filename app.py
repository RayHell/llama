import gradio as gr
import os
import sys
import torch
import fire
import time
import json
from pathlib import Path
from typing import Tuple
from fairscale.nn.model_parallel.initialize import initialize_model_parallel
from llama import ModelArgs, Transformer, Tokenizer, LLaMA


def setup_model_parallel() -> Tuple[int, int]:
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "7861"
    if torch.distributed.is_initialized():
        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
    else:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
        initialize_model_parallel(world_size)
        torch.cuda.set_device(local_rank)
        # seed must be the same in all processes
        torch.manual_seed(1)
    return local_rank, world_size


def load(
    ckpt_dir: str,
    tokenizer_path: str,
    local_rank: int,
    world_size: int,
    max_seq_len: int,
    max_batch_size: int,
) -> LLaMA:
    start_time = time.time()
    checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
    assert world_size == len(
        checkpoints
    ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {world_size}"
    ckpt_path = checkpoints[local_rank]
    print("Loading")
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    with open(Path(ckpt_dir) / "params.json", "r") as f:
        params = json.loads(f.read())

    model_args: ModelArgs = ModelArgs(
        max_seq_len=max_seq_len, max_batch_size=max_batch_size, **params
    )
    tokenizer = Tokenizer(model_path=tokenizer_path)
    model_args.vocab_size = tokenizer.n_words
    torch.set_default_tensor_type(torch.cuda.HalfTensor)
    model = Transformer(model_args)
    torch.set_default_tensor_type(torch.FloatTensor)
    model.load_state_dict(checkpoint, strict=False)

    generator = LLaMA(model, tokenizer)
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return generator


def generate(prompt, temperature, top_p, max_seq_len, max_batch_size):
    ckpt_dir = "models/7B"
    tokenizer_path = "models/tokenizer.model"
    local_rank, world_size = setup_model_parallel()
    if local_rank > 0:
        sys.stdout = open(os.devnull, "w")
    generator = load(
        ckpt_dir, tokenizer_path, local_rank, world_size, max_seq_len, max_batch_size
    )
    results = generator.generate(
        [prompt], max_gen_len=256, temperature=temperature, top_p=top_p
    )
    return results[0]


# Setup Gradio interface
input_prompt = gr.inputs.Textbox(label="Enter Prompt", default="The amazing thing about Reddit is")
temperature = gr.inputs.Slider(minimum=0.0, maximum=2.0, step=0.1, default=0.8, label="Temperature")
top_p = gr.inputs.Slider(minimum=0.0, maximum=1.0, step=0.01, default=0.95, label="Top-p")
max_seq_len = gr.inputs.Slider(minimum=1, maximum=2048, default=512, step=1, label="Max Sequence Length")
max_batch_size = gr.inputs.Slider(minimum=1, maximum=32, default=16, step=1, label="Max Batch Size")
output_text = gr.outputs.Textbox(label="Output Text")

# Launch Gradio web app
interface = gr.Interface(fn=generate, inputs=[input_prompt, temperature, top_p, max_seq_len, max_batch_size], outputs=output_text, title="LLaMA UI", description="Enter a prompt and the LLaMA model will generate a continuation of the prompt.", language="en")
interface.launch()
