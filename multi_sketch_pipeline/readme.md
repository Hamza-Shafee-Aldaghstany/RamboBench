# RAMBO Multi‑Sketch Pipeline

This document describes an experimental extension to the **RAMBO** repository‑level method body completion framework.  
Unlike the vanilla RAMBO pipeline—which retrieves essential repository elements and feeds them into a single code generation pass—the **multi‑sketch** extension generates multiple candidate sketches per prompt and uses a binary classifier to select the most promising sketch.  The classifier is fine‑tuned using LoRA and trained on pass@1 outcomes from Defects4J, so that sketches likely to produce correct code are ranked higher.

## Overview

RAMBO identifies essential classes, methods, fields and their usages to provide context for large language models.  The original paper reports significant improvements over state‑of‑the‑art MBC techniques【251449669295767†L292-L307】.  This extension builds on that idea by:

1. **Generating multiple sketches** for each prompt using a generative code model.
2. **Training a sketch relevance classifier** using LoRA on top of an encoder‑style model.  The classifier predicts whether a sketch will pass unit tests (pass@1).
3. **Selecting the best sketch** according to the classifier and generating full method bodies from it.
4. **Evaluating** the final code against Defects4J tests.

## Prerequisites

- **Python 3.8**.  All scripts have been tested under Python 3.8.
- **GPU(s)**.  Sketch generation and classification benefit from at least one GPU.  Multi‑GPU generation is supported.
- **Defects4J**.  Install and build the Defects4J project and note the path to its `bin` directory.  Edit `pipeline/testing/run_tests.py` to set the `DEFECTS4J_BIN` variable to the full path of your installation【739552546115183†L70-L82】.  Also set the `DEFECTS4J_HOME` environment variable.
- **Dependencies**.  Install required Python packages:

```sh
conda create -n rambo-multi python=3.8
conda activate rambo-multi
pip install -r requirements.txt
pip install -r pipeline/requirements.txt
```

## Pipeline Steps

### 1. Prepare Prompts

Follow the main repository’s README to extract repository elements (EEI and RUE) and create the JSONL file of prompts.  Each record should include at least a `prompt` field containing the infill context with a `<FILL_FUNCTION_BODY>` placeholder.

### 2. Generate Sketches

Use the script in `pipeline/sketch_and_code_infrence/src/generate_multi_gpu.py` to generate candidate sketches.  The script accepts the following arguments【525476501738965†L10-L30】:

- `--input_path`: path to the JSONL file with prompts.
- `--output_dir`: directory where per‑GPU outputs will be written.
- `--gpus`: number of GPUs to use (defaults to 1).  For multi‑GPU generation the script will spawn one process per GPU.
- `--nos`: optional limit on the number of prompts to process.
- `--max_new_tokens`: maximum tokens to generate per sketch (default 600).
- `--model_name`: hugging‑face model name (default `deepseek-ai/deepseek-coder-1.3b-base`).
- `--num_sequences`: number of sketches per prompt (default 3).
- `--max_length`: maximum input length (default 2500).
- `--temperature` and `--top_p`: sampling parameters.
- `--batch_size`: generation batch size (batch sizes > 1 are experimental).

Example command to generate sketches with four GPUs and three sketches per prompt:

```sh
python multi_sketch_pipeline/sketch_and_code_infrence/src/generate_multi_gpu.py \
  --input_path prompts.jsonl \
  --output_dir sketches_out \
  --gpus 4 \
  --num_sequences 3 \
  --model_name deepseek-ai/deepseek-coder-1.3b-base \
  --max_new_tokens 600 \
  --temperature 0.8 \
  --top_p 0.95 \
  --batch_size 1
```

After generation, each GPU writes its predictions to `sketches_out/predictions_<GPU>.jsonl`.  Merge these files into a single JSONL using `merge_outputs.py`:

```sh
python multi_sketch_pipeline/sketch_and_code_infrence/src/merge_outputs.py \
  --output_dir sketches_out \
  --merged_file sketches_all.jsonl
```

### 3. Generate Code From Sketches

Run the official RAMBO pipeline to generate full method bodies from each sketch.  You may reuse the scripts used for method body completion in the original repo.  The input JSONL should contain a `prompt` and a `sketch` field for each candidate.

### 4. Label Sketches for Training

To train the classifier, you need labels indicating whether a sketch leads to a passing solution.  Run the evaluation harness (step 7) on the generated code and record `rel = 1` for sketches whose top completion passes all tests and `rel = 0` otherwise.  Each record in your training JSONL must contain `sketch_prompt`, `sketch` and `rel` fields.

### 5. Train the Sketch Relevance Classifier

The classifier is implemented in `pipeline/lora_training/train.py`.  It loads a hugging‑face encoder model, wraps it with LoRA and fine‑tunes it on the labelled sketches.  Key command‑line arguments include【698361307651735†L118-L150】:

- `--train`: path to training JSONL (must contain `sketch_prompt`, `sketch` and `rel`).
- `--valid`: path to validation JSONL.
- `--model_name`: base model (default `microsoft/codebert-base`).
- `--output_dir`: where to save the fine‑tuned model.
- `--max_length`: maximum token length (default 512).
- `--per_device_batch_size`: batch size per GPU.
- `--num_train_epochs`: number of epochs.
- `--learning_rate`, `--lora_r`, `--lora_alpha`, `--lora_dropout`, and `--fp16`: LoRA hyperparameters.

Example training command:

```sh
python pipeline/lora_training/train.py \
  --train data/train.jsonl \
  --valid data/valid.jsonl \
  --model_name microsoft/codebert-base \
  --output_dir experiments/encoder_lora_sketch_classifier \
  --num_train_epochs 3 \
  --per_device_batch_size 8 \
  --lora_r 16 \
  --lora_alpha 16 \
  --lora_dropout 0.05 \
  --fp16
```

The script produces two sets of weights: `_ada_only` (LoRA adapter) and `_full` (merged model).  Use the merged model for inference.

### 6. Rank Sketches Using the Classifier

To score new sketches, run `pipeline/run_lora_inf.py`.  Specify the directory containing the full model, the input JSONL with sketches and the output file name.  The script attaches the LoRA adapter to the base model, computes probabilities for each sketch and writes the results with a `prob` field:

```sh
python pipeline/run_lora_inf.py \
  --model_dir experiments/encoder_lora_sketch_classifier_full \
  --input_path sketches_all.jsonl \
  --output_path sketches_ranked.jsonl \
  --batch_size 8 \
  --max_length 512
```

For each prompt, sort the sketches by `prob` (descending) and choose the highest‑scoring one for final code generation.

### 7. Evaluate with Defects4J

After generating full method bodies for the selected sketches, evaluate them using the Defects4J harness at `pipeline/testing/run_tests.py`.  Ensure that the `DEFECTS4J_BIN` variable in this script points to your local installation and that the `DEFECTS4J_HOME` environment variable is set【739552546115183†L70-L82】.  Then run:

```sh
python pipeline/testing/run_tests.py \
  --pred final_completions.jsonl \
  --out eval_results.jsonl \
  --workers 6 \
  --max 50
```

The harness runs tests in parallel and outputs pass/fail information.  Use these results both as your final evaluation and to label data for classifier retraining.

## Notes and Tips

- **Multiple iterations**: You can iteratively improve the classifier by retraining on new data from subsequent runs.
- **Hyperparameters**: The number of sketches (`num_sequences`), generation temperature, top‑p sampling and LoRA hyperparameters all influence performance.  Feel free to experiment.
- **Dataset format**: Training/validation JSONL files must include keys `sketch_prompt`, `sketch` and `rel` (0/1)【698361307651735†L118-L150】.  Additional fields will be ignored.
- **Limitations**: The evaluation harness currently resides in a Jupyter notebook and has some limitations in how functions are selected from Defects4J.  These issues are being investigated.

