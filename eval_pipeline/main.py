# main_parallel.py
import os
import json
import math
import argparse
import multiprocessing as mp
from pathlib import Path
from tqdm import tqdm

# Import your ModelLoader and generation util
from model_loader import ModelLoader
from sketch_gen import generate_skethces_and_save_jsonl  # optional if you want single-process fallback

def read_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))
    return items

def write_jsonl(path, items):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def chunked(items, n_chunks):
    """Split list into n_chunks (roughly equal) preserving order."""
    length = len(items)
    chunk_size = math.ceil(length / n_chunks)
    return [items[i:i+chunk_size] for i in range(0, length, chunk_size)]

def worker_process(rank, config_path, items_slice, output_part_path, gen_args, force_one_device):
    """
    rank              : integer worker id
    config_path       : str path to config json
    items_slice       : list of JSON objects (each should contain 'prompt' field)
    output_part_path  : where this worker writes its part
    gen_args          : dict with generation settings (batch_size, max_new_tokens, temp, top_p, etc.)
    force_one_device  : if True, pin model to the GPU assigned by CUDA_VISIBLE_DEVICES
    """
    import os
    # Set visible GPU for this process: map physical GPU rank -> device 0 inside process.
    # This is optional; if you prefer not to change env, set force_one_device False and pass device name.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    # Now import loader (safe to import here after setting env)
    loader = ModelLoader(config_path)
    tokenizer = loader.get_tokenizer()

    # Choose device string for forcing model to a single GPU inside this process.
    # If CUDA_VISIBLE_DEVICES is set to "rank" above, inside process its CUDA:0 maps to physical GPU rank.
    device_for_model = None
    if force_one_device and torch.cuda.is_available():
        # pin full model to "cuda:0" inside this process
        device_for_model = "cuda:0"
    # Load model (this will be on device_for_model if provided)
    model = loader.get_model(force_device=device_for_model)

    # ensure pad token exists
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    batch_size = gen_args.get("batch_size", 1)
    max_new_tokens = gen_args.get("max_new_tokens", 256)
    temperature = gen_args.get("temperature", 0.3)
    top_p = gen_args.get("top_p", 0.95)
    min_new_tokens = gen_args.get("min_new_tokens", 1)

    device = model.device  # device where model lives

    out_items = []
    # create batches
    for i in range(0, len(items_slice), batch_size):
        batch = items_slice[i:i+batch_size]
        prompts = [x["prompt"] for x in batch]

        # Tokenize batch with padding
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        # Move to model device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        input_lens = inputs["attention_mask"].sum(dim=1).tolist()

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=pad_id,
            )

        # outputs shape: (batch_size, seq_len_out)
        for j in range(outputs.shape[0]):
            out_ids = outputs[j]
            prompt_len = input_lens[j]
            # ensure prompt_len is int and <= out_ids length
            if prompt_len >= out_ids.shape[-1]:
                generated_text = ""
            else:
                generated_text = tokenizer.decode(out_ids[prompt_len:], skip_special_tokens=True)

            item = batch[j]
            item["prompt_predictions"] = generated_text
            out_items.append(item)

    # write worker part
    write_jsonl(output_part_path, out_items)
    print(f"[worker {rank}] wrote {len(out_items)} items to {output_part_path}")

def merge_parts(output_path, n_parts):
    # collect parts in order
    merged = []
    for r in range(n_parts):
        part_path = f"{output_path}.part{r}"
        if not Path(part_path).exists():
            continue
        merged.extend(read_jsonl(part_path))
        # optionally remove part file
        # os.remove(part_path)
    write_jsonl(output_path, merged)
    print(f"Merged {n_parts} parts into {output_path}")

if __name__ == "__main__":
    import torch
    from pathlib import Path

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input_jsonl", required=True)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--num_workers", type=int, default=1)   # set via config too
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--min_new_tokens", type=int, default=1)
    parser.add_argument("--force_one_device", action="store_true",
                        help="Pin full model to one GPU inside each worker (for when model fits on single GPU)")
    args = parser.parse_args()

    items = read_jsonl(args.input_jsonl)
    n_workers = args.num_workers

    # split dataset into n workers
    slices = chunked(items, n_workers)

    processes = []
    for rank, slice_items in enumerate(slices):
        out_part = f"{args.output_jsonl}.part{rank}"
        p = mp.Process(
            target=worker_process,
            args=(rank, args.config, slice_items, out_part,
                  {"batch_size": args.batch_size,
                   "max_new_tokens": args.max_new_tokens,
                   "temperature": args.temperature,
                   "top_p": args.top_p,
                   "min_new_tokens": args.min_new_tokens},
                  args.force_one_device)
        )
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    # merge parts
    merge_parts(args.output_jsonl, len(slices))
