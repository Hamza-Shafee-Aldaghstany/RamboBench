import json
import torch
from pathlib import Path
from tqdm import tqdm


def generate_and_save_jsonl(
    model,
    tokenizer,
    input_jsonl_path,
    output_jsonl_path,
    max_new_tokens=600,
    temperature=0.3,
    top_p=0.95,
    min_new_tokens=2,
):
    """
    Reads a JSONL file with a 'prompt' field, generates model predictions,
    and writes a new JSONL file with an added 'prompt_predictions' field.
    """

    Path(output_jsonl_path).parent.mkdir(parents=True, exist_ok=True)

    device = model.device

    with open(input_jsonl_path, "r", encoding="utf-8") as fin, \
         open(output_jsonl_path, "w", encoding="utf-8") as fout:

        for line in tqdm(fin, desc="Generating predictions"):
            item = json.loads(line)

            prompt = item["prompt"]

            inputs = tokenizer(prompt, return_tensors="pt").to(device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    min_new_tokens=min_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=True,
                    pad_token_id=tokenizer.eos_token_id,
                )

            generated_text = tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[-1]:],
                skip_special_tokens=True
            )

            # Add new column
            item["prompt_predictions"] = generated_text

            fout.write(json.dumps(item, ensure_ascii=False) + "\n")
