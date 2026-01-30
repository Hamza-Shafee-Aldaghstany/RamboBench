#!/usr/bin/env python3
"""
train_encoder_lora_sketch_classifier_fixed.py

Train a binary sketch relevance classifier using an encoder-style model
(AutoModelForSequenceClassification) with LoRA (PEFT).

Example:
  python train_encoder_lora_sketch_classifier_fixed.py \
    --train train.jsonl --valid valid.jsonl \
    --model_name microsoft/codebert-base \
    --output_dir ./encoder_lora_sketch_classifier \
    --num_train_epochs 3 --per_device_batch_size 8 --fp16
"""
import argparse
import json
import os
from typing import Dict

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          set_seed)


# ---------------------------
# Simple JSONL loader (kept from your previous code)
# ---------------------------
def load_jsonl_to_hf(
    path, context_key="sketch_prompt", sketch_key="sketch", label_key="rel"
):
    """
    Load a JSONL where each line is a dict that must contain `context_key` and `sketch_key`.
    The label is read from `label_key`.
    Returns a HuggingFace Dataset with columns: text, label
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    rows = []
    leblesss = []
    with open(path, "r", encoding="utf-8") as f:
        coun = 0
        for line in f:
            coun += 1
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if context_key not in item or sketch_key not in item:
                # skip incomplete records
                continue

            # Normalize label robustly
            if label_key not in item:
                raise KeyError(f"Label key '{label_key}' missing in one record: {item}")
            label_raw = item[label_key]
            if isinstance(label_raw, bool):
                label = 1 if label_raw else 0
            else:
                try:
                    label = int(label_raw)
                    # print(label)
                except Exception:
                    label = 1 if str(label_raw).lower() in ("true", "1", "yes") else 0
            leblesss.append(label)

            text = "\n\n[SKETCH]\n" + str(item[sketch_key])
            # text =  "\n\n[SKETCH]\n" + str(item[sketch_key])
            rows.append({"text": text, "label": int(label)})
            # if coun==2:
            #     break
    print(len([0 for l in leblesss if l == 0]), len([0 for l in leblesss if l == 1]))
    if len(rows) == 0:
        raise ValueError(f"No valid rows found in {path}")
    return Dataset.from_list(rows)


# ---------------------------
# Metrics
# ---------------------------
def compute_metrics(pred):
    """pred is a transformers.EvalPrediction with .predictions and .label_ids"""
    logits = pred.predictions  # shape (N, num_labels)
    labels = pred.label_ids
    # convert to probabilities
    import torch.nn.functional as F

    probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()
    preds = np.argmax(probs, axis=1)
    from sklearn.metrics import (accuracy_score,
                                 precision_recall_fscore_support,
                                 roc_auc_score)

    acc = accuracy_score(labels, preds)
    prec, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    try:
        auc = (
            float(roc_auc_score(labels, probs[:, 1]))
            if len(np.unique(labels)) > 1
            else 0.0
        )
    except Exception:
        auc = 0.0
    return {"accuracy": acc, "precision": prec, "recall": recall, "f1": f1, "auc": auc}


# ---------------------------
# Main
# ---------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train",
        required=True,
        help="train jsonl path (must contain sketch_prompt, sketch, rel)",
    )
    parser.add_argument("--valid", required=True, help="valid jsonl path")
    parser.add_argument(
        "--model_name",
        default="microsoft/codebert-base",
        help="encoder-style model checkpoint",
    )
    parser.add_argument("--output_dir", default="./encoder_lora_sketch_classifier")
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="max token length for tokenizer (512 recommended)",
    )
    parser.add_argument("--per_device_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--num_train_epochs", type=int, default=3)
    parser.add_argument("--learning_rate", type=float, default=1e-04)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--fp16", action="store_true", help="use fp16")
    parser.add_argument("--logging_dir", default="./runs")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)

    # Load dataset
    print("Loading datasets...")
    train_ds = load_jsonl_to_hf(args.train)
    valid_ds = load_jsonl_to_hf(args.valid)

    # Tokenizer
    print(f"Loading tokenizer for {args.model_name} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True, max_length=args.max_length)

    # Tokenize datasets
    train_ds = train_ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    valid_ds = valid_ds.map(tokenize_fn, batched=True, remove_columns=["text"])

    # Rename label -> labels (Trainer / HF convention)
    def rename_label(batch):
        batch["labels"] = batch.pop("label")
        return batch

    train_ds = train_ds.map(rename_label, batched=True)
    valid_ds = valid_ds.map(rename_label, batched=True)

    # Set PyTorch format and ensure columns exist
    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    valid_ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    # Model
    print(f"Loading sequence classification model {args.model_name} ...")
    # We use num_labels=2 and CrossEntropy (classic)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=2,
        trust_remote_code=True,
        # low_cpu_mem_usage=True,
    )

    # Resize token embeddings if tokenizer changed (pad token added)
    vocab_size_model = model.get_input_embeddings().weight.shape[0]
    if len(tokenizer) != vocab_size_model:
        model.resize_token_embeddings(len(tokenizer))

    # LoRA config (PEFT) - use TaskType.TOKEN_CLS for encoder classification
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        # typical attention/projection module names in Roberta-like models:
        target_modules=["query", "key", "value", "dense"],
        lora_dropout=args.lora_dropout,
        bias="none",
        modules_to_save=["classifier"],
        task_type=TaskType.TOKEN_CLS,
    )
    print("Applying PEFT LoRA...")
    model = get_peft_model(model, lora_config)
    for param in model.classifier.parameters():
        param.requires_grad = True
    # Data collator
    data_collator = DataCollatorWithPadding(tokenizer)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="steps",
        logging_steps=50,
        learning_rate=args.learning_rate,
        logging_dir=args.logging_dir,
        fp16=args.fp16,
        report_to="tensorboard",
        save_total_limit=3,
        load_best_model_at_end=False,
        remove_unused_columns=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=valid_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train
    print("Starting training...")
    trainer.train()

    # Evaluate
    print("Final evaluation:")
    print(trainer.evaluate())

    # Save adapter + tokenizer (PEFT adapter weights saved by model.save_pretrained)
    # Save adapter + tokenizer (optional if you want PEFT adapter only)
    os.makedirs(args.output_dir + "_ada_only", exist_ok=True)
    model.save_pretrained(args.output_dir + "_ada_only")
    tokenizer.save_pretrained(args.output_dir + "_ada_only")
    print("Saved PEFT adapter + tokenizer.")

    # --- NEW: merge LoRA and save full model for inference ---
    os.makedirs(args.output_dir + "_full", exist_ok=True)
    model.merge_and_unload()  # merges LoRA into base model
    model.save_pretrained(args.output_dir + "_full")  # saves full model
    tokenizer.save_pretrained(args.output_dir + "_full")

    # print("Saved full model + tokenizer for inference in Jupyter.")

    # Small inference helper: returns positive-class probabilities for each row in a JSONL file
    def predict_probs(
        path_jsonl, context_key="sketch_prompt", sketch_key="sketch", label_key="rel"
    ):
        ds = load_jsonl_to_hf(
            path_jsonl,
            context_key=context_key,
            sketch_key=sketch_key,
            label_key=label_key,
        )
        ds = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
        ds = ds.map(rename_label, batched=True)
        ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
        preds = trainer.predict(ds)
        logits = np.array(preds.predictions)  # shape (N,2)
        import torch.nn.functional as F

        probs = F.softmax(torch.from_numpy(logits), dim=-1).numpy()
        return probs[:, 1]  # positive-class prob

    # Example usage (uncomment)
    # sample_probs = predict_probs(args.valid)
    # print("sample positive-class probs:", sample_probs[:10])


if __name__ == "__main__":
    main()
