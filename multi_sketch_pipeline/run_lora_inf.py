# run_inference_and_store_jsonl.py
"""
Usage:
  python run_inference_and_store_jsonl.py \
    --model_dir ./encoder_lora_sketch_classifier_full \
    --input_path ram.jsonl \
    --output_path results_with_probs.jsonl
"""
import argparse
import json

import torch
import torch.nn.functional as F
from peft import PeftModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_MODEL = "microsoft/codebert-base"
ADAPTER_DIR = "/home/user/Desktop/Code_Bench/encoder_lora_sketch_classifier_ada_only"


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def extract_task_id(row):
    if (
        "metadata" in row
        and isinstance(row["metadata"], dict)
        and "task_id" in row["metadata"]
    ):
        return row["metadata"]["task_id"]
    return row.get("task_id")


def run_inference(rows, tokenizer, model, device, batch_size=32, max_length=512):
    texts = ["\n\n[SKETCH]\n" + str(r.get("sketch", "")) for r in rows]

    model.eval()
    probs_all = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = tokenizer(
                batch,
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = F.softmax(logits, dim=-1)[:, 1]
            probs_all.extend(probs.cpu().tolist())

    for row, p in zip(rows, probs_all):
        row["prob"] = float(p)
        row["_task_id_cache"] = extract_task_id(row)

    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--input_path", required=True, help="input ram JSONL")
    parser.add_argument("--output_path", default="results_with_probs.jsonl")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_length", type=int, default=512)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading model...")

    # 1) Load tokenizer (IMPORTANT: from adapter dir if you saved it there)
    tokenizer = AutoTokenizer.from_pretrained(ADAPTER_DIR, trust_remote_code=True)

    # 2) Load base model
    base = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=2,
        trust_remote_code=True,
    )

    base.to(DEVICE)

    # 3) Attach LoRA adapter
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)

    model.eval()
    print("Loading input data...")
    rows = load_jsonl(args.input_path)

    print("Running inference...")
    rows = run_inference(
        rows,
        tokenizer,
        model,
        device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    print(f"Saving JSONL → {args.output_path}")
    write_jsonl(args.output_path, rows)

    print("Done. Rows:", len(rows))


if __name__ == "__main__":
    main()
