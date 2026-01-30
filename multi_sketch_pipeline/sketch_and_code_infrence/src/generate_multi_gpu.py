import argparse
import json
import os
import time
import traceback

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils import read_jsonl, write_jsonl_line


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--gpus", type=int, default=1)
    p.add_argument("--nos", type=int, default=None)
    p.add_argument("--max_new_tokens", type=int, default=600)
    p.add_argument(
        "--model_name", type=str, default="deepseek-ai/deepseek-coder-1.3b-base"
    )
    p.add_argument("--num_sequences", type=int, default=3)
    p.add_argument("--max_length", type=int, default=2500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Currently batch_size>1 is experimental",
    )
    return p.parse_args()


def build_fim_prompt(prompt: str) -> str:
    return (
        "<｜fim▁begin｜>"
        + prompt.replace("<FILL_FUNCTION_BODY>", "<｜fim▁hole｜>")
        + "<｜fim▁end｜>"
    )


def worker(rank: int, world_size: int, args):
    # set device for this worker
    device = f"cuda:{rank}"
    torch.cuda.set_device(rank)

    # load tokenizer + model on this worker
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True
    ).to(device)
    model.eval()
    if args.nos < 1:
        data = read_jsonl(args.input_path)
    else:
        data = read_jsonl(args.input_path)[: args.nos]
    n = len(data)

    # create per-worker output file
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"predictions_{rank}.jsonl")

    assigned_indices = list(range(rank, n, world_size))

    # use tqdm with position so multiple progress bars can show
    pbar = tqdm(assigned_indices, desc=f"GPU {rank}", position=rank, leave=True)

    for idx in pbar:
        item = data[idx]
        try:
            prompt = item.get("prompt", "")
            fim_prompt = build_fim_prompt(prompt)

            # single-example generation (batch_size=1)
            inputs = tokenizer(fim_prompt, return_tensors="pt", truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    num_return_sequences=args.num_sequences,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                )

            predictions = []
            for out in outputs:
                decoded = tokenizer.decode(out, skip_special_tokens=True)
                # strip prompt prefix
                predictions.append(decoded[len(fim_prompt) :])

            item["predictions_code"] = predictions

            # append to per-worker file
            write_jsonl_line(out_path, item)

        except Exception as e:
            tb = traceback.format_exc()
            print(f"Error on GPU {rank} idx {idx}: {e}\n{tb}")
            # still write a record for traceability
            item.setdefault("predictions_code", [])
            item.setdefault("error", str(e))
            write_jsonl_line(out_path, item)

    print(f"GPU {rank} finished. Wrote: {out_path}")


def main():
    args = parse_args()

    # spawn processes, each will run 'worker'
    world_size = args.gpus

    if world_size <= 1:
        # single-GPU fallback
        worker(0, 1, args)
        return

    # torch.multiprocessing.spawn will start `world_size` processes
    torch.multiprocessing.spawn(
        worker, args=(world_size, args), nprocs=world_size, join=True
    )


if __name__ == "__main__":
    main()
