#!/usr/bin/env python3
"""
Run three baselines (simple, medium, strong) with Qwen2.5-1.5B-Instruct.
Single command: installs deps, downloads data, trains, evaluates, outputs metrics.
Usage: python run_baselines.py [--skip-install] [--skip-download] [--data-dir DIR] [--output-dir DIR]
"""

import argparse
import csv
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path
from urllib.request import urlretrieve

# -----------------------------------------------------------------------------
# Setup: install deps and download data (before importing heavy libs)
# -----------------------------------------------------------------------------
DATA_URLS = {
    "gsm8k_train.jsonl": "https://www.csie.ntu.edu.tw/~b10902031/gsm8k_train.jsonl",
    "gsm8k_train_self-instruct.jsonl": "https://www.csie.ntu.edu.tw/~b10902031/gsm8k_train_self-instruct.jsonl",
    "gsm8k_test_public.jsonl": "https://www.csie.ntu.edu.tw/~b10902031/gsm8k_test_public.jsonl",
    "gsm8k_test_private.jsonl": "https://www.csie.ntu.edu.tw/~b10902031/gsm8k_test_private.jsonl",
    "ailuminate_test.csv": "https://www.csie.ntu.edu.tw/~b10902031/ailuminate_test.csv",
}


def do_install():
    subprocess.check_call([
        sys.executable, "-m", "pip", "install", "-U", "-q",
        "datasets", "trl", "bitsandbytes", "transformers", "accelerate", "peft", "tqdm",
    ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def do_download(data_dir: Path):
    data_dir.mkdir(parents=True, exist_ok=True)
    for fname, url in DATA_URLS.items():
        path = data_dir / fname
        if path.exists():
            print(f"  Data exists: {path.name}")
            continue
        print(f"  Downloading {fname} ...")
        urlretrieve(url, path)
    print("  Data ready.")


# -----------------------------------------------------------------------------
# Imports (after optional install)
# -----------------------------------------------------------------------------
def ensure_imports():
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
    from tqdm import tqdm
    from trl import SFTConfig, SFTTrainer
    import csv
    return torch, Dataset, LoraConfig, PeftModel, get_peft_model, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, tqdm, SFTConfig, SFTTrainer


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
SEED = 42
TRAIN_SEED = 1126
GSM8K_FIXED_FEWSHOT_INDICES = [0, 500, 1000, 1500, 2000]
GSM8K_PUBLIC_LIMIT = 100
GSM8K_PRIVATE_LIMIT = 100
AILUMINATE_PUBLIC_SLICE = (0, 40)
AILUMINATE_PRIVATE_SLICE = (120, 160)
PORTION = 1.0 / 3.0  # longest 1/3


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------
def load_jsonlines(path):
    with open(path, "r") as f:
        return [json.loads(line) for line in f]


def load_csv(path):
    with open(path) as f:
        rows = csv.DictReader(f)
        return [row["prompt_text"] for row in rows]


# -----------------------------------------------------------------------------
# Chat formatting (supports fixed or random few-shot)
# -----------------------------------------------------------------------------
def nshot_chats(nshot_data, n, question, answer, mode, fixed_indices=None):
    if mode not in ("train", "test"):
        raise ValueError("mode must be 'train' or 'test'")
    chats = []
    if fixed_indices is not None and len(fixed_indices) <= len(nshot_data):
        sample = [nshot_data[i] for i in fixed_indices[:n]]
    else:
        sample = random.sample(nshot_data, min(n, len(nshot_data)))
    for qna in sample:
        chats.append({"role": "user", "content": f'Q: {qna["question"]}'})
        chats.append({"role": "assistant", "content": f'A: {qna["answer"]}'})
    chats.append({
        "role": "user",
        "content": f"Q: {question} Let's think step by step. At the end, you MUST write the answer as an integer after '####'.",
    })
    if mode == "train":
        chats.append({"role": "assistant", "content": f"A: {answer}"})
    return chats


# -----------------------------------------------------------------------------
# Format training data: full text (simple) or prompt+completion (medium/strong)
# -----------------------------------------------------------------------------
def format_gsm8k_full_text(gsm8k_train, tokenizer, train_n_shot=1, fixed_indices=None):
    """Like simple.ipynb: full sequence for SFT (no assistant-only masking)."""
    formatted = []
    for qna in gsm8k_train:
        chats = nshot_chats(gsm8k_train, train_n_shot, qna["question"], qna["answer"], "train", fixed_indices)
        text = tokenizer.apply_chat_template(chats, tokenize=False)
        # Qwen has no <|eot_id|>; use full template
        formatted.append({"text": text})
    return Dataset.from_list(formatted)


def format_gsm8k_prompt_completion(gsm8k_train, tokenizer, train_n_shot=1, exclude_indices=None, fixed_indices=None):
    """Prompt + completion for assistant-only loss."""
    if exclude_indices is not None:
        exclude_set = set(exclude_indices)
        gsm8k_train = [x for i, x in enumerate(gsm8k_train) if i not in exclude_set]
    formatted = []
    for qna in gsm8k_train:
        chats = nshot_chats(gsm8k_train, train_n_shot, qna["question"], qna["answer"], "train", fixed_indices)
        chats_prompt = chats[:-1]
        prompt = tokenizer.apply_chat_template(chats_prompt, tokenize=False, add_generation_prompt=True)
        completion = f"A: {qna['answer']}"
        formatted.append({"prompt": prompt, "completion": completion})
    return Dataset.from_list(formatted)


def filter_longest_portion(dataset, portion=PORTION, use_longest=True):
    def _letters(s):
        s = "" if s is None else (s if isinstance(s, str) else str(s))
        return sum(1 for ch in s if ch.isalpha())
    cols = getattr(dataset, "column_names", None) or []
    if "prompt" in cols:
        fields = ("prompt", "completion")
    elif "text" in cols:
        fields = ("text",)
    else:
        fields = ("question", "answer")
    n = len(dataset)
    k = max(1, int(round(n * portion)))
    lengths = [sum(_letters(dataset[i].get(f, "")) for f in fields) for i in range(n)]
    top_idx = sorted(range(n), key=lambda i: lengths[i], reverse=use_longest)[:k]
    return dataset.select(top_idx)


# -----------------------------------------------------------------------------
# Answer extraction and inference
# -----------------------------------------------------------------------------
def extract_ans_from_response(answer: str) -> str:
    if not answer or not isinstance(answer, str):
        return ""
    text = answer.strip()
    for c in [",", "$", "%"]:
        text = text.replace(c, "")
    if "####" in answer:
        after = answer.split("####")[-1].strip()
        for c in [",", "$", "%"]:
            after = after.replace(c, "")
        # last number
        matches = re.findall(r"-?\d+\.?\d*", after)
        if matches:
            s = matches[-1].strip()
            if s.endswith("g") and len(s) > 1 and s[:-1].replace(".", "").replace("-", "").isdigit():
                s = s[:-1]
            return s.strip()
        return after.strip()
    matches = re.findall(r"-?\d+\.?\d*", text)
    if matches:
        s = matches[-1]
        if s.endswith("g") and len(s) > 1 and s[:-1].replace(".", "").replace("-", "").isdigit():
            s = s[:-1]
        return s.strip()
    return text


def get_response(generator, chats, max_new_tokens=256, do_sample=True, temperature=0.6, top_p=0.9):
    out = generator(
        chats,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else None,
        top_p=top_p if do_sample else None,
    )[0]
    return out["generated_text"][-1]["content"]


# -----------------------------------------------------------------------------
# Evaluation
# -----------------------------------------------------------------------------
def eval_gsm8k(generator, gsm8k_test_public, gsm8k_train, test_n_shot, fixed_indices,
              max_new_tokens, do_sample, temperature, top_p, limit=100):
    test = gsm8k_test_public[:limit]
    correct = 0
    predictions = []
    for qna in tqdm(test, desc="GSM8K Eval"):
        messages = nshot_chats(gsm8k_train, test_n_shot, qna["question"], None, "test", fixed_indices)
        response = get_response(generator, messages, max_new_tokens=max_new_tokens,
                                do_sample=do_sample, temperature=temperature, top_p=top_p)
        pred_ans = extract_ans_from_response(response)
        true_ans = extract_ans_from_response(qna["answer"])
        if pred_ans == true_ans:
            correct += 1
        predictions.append(pred_ans)
    accuracy = correct / len(test) if test else 0.0
    return accuracy, predictions


def run_gsm8k_private(generator, gsm8k_test_private, gsm8k_train, test_n_shot, fixed_indices,
                     max_new_tokens, do_sample, temperature, top_p, limit=100):
    test = gsm8k_test_private[:limit]
    predictions = []
    for qna in tqdm(test, desc="GSM8K Private"):
        messages = nshot_chats(gsm8k_train, test_n_shot, qna["question"], None, "test", fixed_indices)
        response = get_response(generator, messages, max_new_tokens=max_new_tokens,
                                do_sample=do_sample, temperature=temperature, top_p=top_p)
        predictions.append(extract_ans_from_response(response))
    return predictions


def run_ailuminate(generator, ailuminate_test, max_new_tokens, do_sample, temperature, top_p):
    predictions = []
    for q in tqdm(ailuminate_test, desc="AILuminate"):
        message = [{"role": "user", "content": q}]
        response = get_response(generator, message, max_new_tokens=max_new_tokens,
                                do_sample=do_sample, temperature=temperature, top_p=top_p)
        predictions.append(response)
    return predictions


# -----------------------------------------------------------------------------
# Baseline runner
# -----------------------------------------------------------------------------
def run_baseline(
    name,
    data_dir,
    output_base,
    gsm8k_train,
    gsm8k_train_si,
    gsm8k_test_public,
    gsm8k_test_private,
    ailuminate_test,
    config,
):
    torch, Dataset, LoraConfig, PeftModel, get_peft_model, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline, tqdm, SFTConfig, SFTTrainer = ensure_imports()
    random.seed(SEED)
    torch.manual_seed(SEED)

    output_dir = Path(output_base) / f"sft_{name}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Model & tokenizer
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.model_max_length = 10000
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    model.resize_token_embeddings(len(tokenizer))

    # Training data
    train_data = config["train_data"]
    train_n_shot = config["train_n_shot"]
    use_prompt_completion = config.get("use_prompt_completion", False)
    fixed_fewshot = config.get("fixed_fewshot", None)
    exclude_fewshot = config.get("exclude_fewshot", None)

    if use_prompt_completion:
        formatted = format_gsm8k_prompt_completion(
            train_data, tokenizer, train_n_shot=train_n_shot,
            exclude_indices=exclude_fewshot, fixed_indices=fixed_fewshot,
        )
    else:
        formatted = format_gsm8k_full_text(
            train_data, tokenizer, train_n_shot=train_n_shot, fixed_indices=fixed_fewshot,
        )
    use_longest = config.get("filter_longest", True)
    formatted = filter_longest_portion(formatted, portion=PORTION, use_longest=use_longest)

    # LoRA
    peft_config = LoraConfig(
        r=config.get("lora_r", 8),
        lora_alpha=config.get("lora_alpha", 16),
        lora_dropout=config.get("lora_dropout", 0.0),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["up_proj", "down_proj", "gate_proj", "k_proj", "q_proj", "v_proj", "o_proj"],
    )
    peft_model = get_peft_model(model, peft_config).to(dtype=torch.bfloat16)

    # SFT config
    sft_kw = {
        "seed": TRAIN_SEED,
        "data_seed": TRAIN_SEED,
        "output_dir": str(output_dir),
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "optim": "paged_adamw_32bit",
        "num_train_epochs": config.get("num_train_epochs", 1),
        "logging_strategy": "steps",
        "logging_steps": 0.1,
        "save_strategy": "steps",
        "save_steps": config.get("save_steps", 0.1),
        "lr_scheduler_type": config.get("lr_scheduler_type", "linear"),
        "learning_rate": config.get("learning_rate", 3e-4),
        "warmup_ratio": config.get("warmup_ratio", 0.0),
        "weight_decay": config.get("weight_decay", 0.0),
        "bf16": True,
        "report_to": "none",
    }
    if config.get("max_seq_length"):
        sft_kw["max_seq_length"] = config["max_seq_length"]
    if use_prompt_completion:
        # no dataset_text_field => prompt+completion
        pass
    else:
        sft_kw["dataset_text_field"] = "text"

    training_args = SFTConfig(**sft_kw)
    trainer = SFTTrainer(
        model=peft_model,
        train_dataset=formatted,
        processing_class=tokenizer,
        args=training_args,
    )
    trainer.train()
    del trainer
    peft_model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Checkpoints to evaluate
    checkpoints = sorted([p for p in output_dir.iterdir() if p.is_dir() and p.name.startswith("checkpoint-")],
                        key=lambda p: int(p.name.split("-")[1]))
    if not checkpoints:
        checkpoints = [output_dir]
    eval_checkpoints = config.get("eval_checkpoints", "last")
    if eval_checkpoints == "all" and len(checkpoints) > 1:
        to_eval = checkpoints
    else:
        to_eval = [checkpoints[-1]] if checkpoints else [output_dir]

    # Inference config
    max_new_tokens = config.get("max_new_tokens", 256)
    do_sample = config.get("do_sample", True)
    temperature = config.get("temperature", 0.6)
    top_p = config.get("top_p", 0.9)
    test_n_shot = config.get("test_n_shot", 1)
    fixed_test = config.get("fixed_fewshot_for_test", None)
    test_fewshot_data = config.get("test_fewshot_data", gsm8k_train)

    best_acc = -1.0
    best_ckpt = None
    all_metrics = []

    for ckpt_path in to_eval:
        # Load adapter
        base_for_inference = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        base_for_inference.resize_token_embeddings(len(tokenizer))
        if (ckpt_path / "adapter_config.json").exists() or (ckpt_path / "adapter_model.safetensors").exists():
            model_inf = PeftModel.from_pretrained(base_for_inference, str(ckpt_path), torch_dtype=torch.bfloat16)
        else:
            model_inf = base_for_inference
        gen = pipeline(
            "text-generation",
            model=model_inf,
            tokenizer=tokenizer,
            pad_token_id=tokenizer.eos_token_id,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
        )
        acc, gsm8k_pub_preds = eval_gsm8k(
            gen, gsm8k_test_public, test_fewshot_data, test_n_shot, fixed_test,
            max_new_tokens, do_sample, temperature, top_p, limit=GSM8K_PUBLIC_LIMIT,
        )
        gsm8k_priv_preds = run_gsm8k_private(
            gen, gsm8k_test_private, test_fewshot_data, test_n_shot, fixed_test,
            max_new_tokens, do_sample, temperature, top_p, limit=GSM8K_PRIVATE_LIMIT,
        )
        ailuminate_preds = run_ailuminate(
            gen, ailuminate_test, max_new_tokens, do_sample, temperature, top_p,
        )
        if acc > best_acc:
            best_acc = acc
            best_ckpt = ckpt_path
            best_gsm8k_pub = gsm8k_pub_preds
            best_gsm8k_priv = gsm8k_priv_preds
            best_ailuminate = ailuminate_preds
        all_metrics.append({"checkpoint": str(ckpt_path), "gsm8k_public_accuracy": round(acc, 4)})
        del base_for_inference
        if hasattr(model_inf, "base_model"):
            del model_inf
        del gen
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Use best checkpoint's predictions for submission
    metrics = {
        "baseline": name,
        "gsm8k_public_accuracy": round(best_acc, 4),
        "checkpoint_evaluated": str(best_ckpt),
        "all_checkpoint_accuracies": all_metrics,
    }
    submission = best_gsm8k_pub + best_gsm8k_priv + best_ailuminate
    return metrics, submission


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run simple, medium, strong baselines (Qwen2.5-1.5B)")
    parser.add_argument("--skip-install", action="store_true", help="Skip pip install")
    parser.add_argument("--skip-download", action="store_true", help="Skip data download")
    parser.add_argument("--data-dir", type=str, default=".", help="Data directory")
    parser.add_argument("--output-dir", type=str, default=".", help="Output directory for checkpoints and results")
    parser.add_argument("--baseline", type=str, choices=["simple", "medium", "strong", "all"], default="all",
                        help="Which baseline(s) to run")
    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    output_base = Path(args.output_dir)

    print("=" * 60)
    print("BASELINES: Simple / Medium / Strong (Qwen2.5-1.5B-Instruct)")
    print("=" * 60)

    if not args.skip_install:
        print("Installing dependencies...")
        do_install()
        print("Done.")
    if not args.skip_download:
        print("Downloading data...")
        do_download(data_dir)
    else:
        data_dir.mkdir(parents=True, exist_ok=True)

    # Load data once
    gsm8k_train = load_jsonlines(data_dir / "gsm8k_train.jsonl")
    gsm8k_train_si = load_jsonlines(data_dir / "gsm8k_train_self-instruct.jsonl")
    gsm8k_test_public = load_jsonlines(data_dir / "gsm8k_test_public.jsonl")
    gsm8k_test_private = load_jsonlines(data_dir / "gsm8k_test_private.jsonl")
    ailuminate_all = load_csv(data_dir / "ailuminate_test.csv")
    ailuminate_test = ailuminate_all[AILUMINATE_PUBLIC_SLICE[0]:AILUMINATE_PUBLIC_SLICE[1]] + ailuminate_all[AILUMINATE_PRIVATE_SLICE[0]:AILUMINATE_PRIVATE_SLICE[1]]

    n_train = len(gsm8k_train)
    fewshot_indices = [i for i in GSM8K_FIXED_FEWSHOT_INDICES if i < n_train][:5]
    if len(fewshot_indices) < 5:
        fewshot_indices = list(range(min(5, n_train)))
    n_train_si = len(gsm8k_train_si)
    fewshot_indices_si = [i for i in GSM8K_FIXED_FEWSHOT_INDICES if i < n_train_si][:5]
    if len(fewshot_indices_si) < 5:
        fewshot_indices_si = list(range(min(5, n_train_si)))

    # Baseline configs
    simple_config = {
        "train_data": gsm8k_train,
        "test_fewshot_data": gsm8k_train,
        "train_n_shot": 1,
        "use_prompt_completion": False,
        "fixed_fewshot": None,
        "exclude_fewshot": None,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "num_train_epochs": 1,
        "save_steps": 0.1,
        "learning_rate": 3e-4,
        "warmup_ratio": 0.0,
        "weight_decay": 0.0,
        "eval_checkpoints": "last",
        "max_new_tokens": 256,
        "do_sample": True,
        "temperature": 0.6,
        "top_p": 0.9,
        "test_n_shot": 1,
        "fixed_fewshot_for_test": None,
        "filter_longest": False,
    }

    medium_config = {
        "train_data": gsm8k_train,
        "test_fewshot_data": gsm8k_train,
        "train_n_shot": 5,
        "use_prompt_completion": True,
        "fixed_fewshot": None,
        "exclude_fewshot": None,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.0,
        "num_train_epochs": 1,
        "save_steps": 30,
        "learning_rate": 5e-5,
        "warmup_ratio": 0.1,
        "weight_decay": 0.0,
        "max_seq_length": 2480,
        "eval_checkpoints": "all",
        "max_new_tokens": 512,
        "do_sample": False,
        "temperature": 0.6,
        "top_p": 0.9,
        "test_n_shot": 5,
        "fixed_fewshot_for_test": fewshot_indices,
        "filter_longest": True,
    }

    strong_config = {
        "train_data": gsm8k_train_si,
        "test_fewshot_data": gsm8k_train_si,
        "train_n_shot": 5,
        "use_prompt_completion": True,
        "fixed_fewshot": fewshot_indices_si,
        "exclude_fewshot": fewshot_indices_si,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.1,
        "num_train_epochs": 3,
        "save_steps": 30,
        "learning_rate": 5e-5,
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "max_seq_length": 2480,
        "eval_checkpoints": "all",
        "max_new_tokens": 512,
        "do_sample": False,
        "temperature": 0.6,
        "top_p": 0.9,
        "test_n_shot": 5,
        "fixed_fewshot_for_test": fewshot_indices_si,
        "filter_longest": True,
    }

    results = {}
    to_run = []
    if args.baseline in ("simple", "all"):
        to_run.append(("simple baseline", "simple", simple_config))
    if args.baseline in ("medium", "all"):
        to_run.append(("medium baseline", "medium", medium_config))
    if args.baseline in ("strong", "all"):
        to_run.append(("strong baseline", "strong", strong_config))

    for label, name, config in to_run:
        print("\n" + "=" * 60)
        print(f"Running: {label}")
        print("=" * 60)
        metrics, submission = run_baseline(
            name, data_dir, output_base,
            gsm8k_train, gsm8k_train_si,
            gsm8k_test_public, gsm8k_test_private,
            ailuminate_test,
            config,
        )
        results[label] = metrics
        out_txt = output_base / f"submission_{name}.txt"
        with open(out_txt, "w") as f:
            print(submission, file=f)
        print(f"  Submission saved: {out_txt}")

    # Summary
    print("\n" + "=" * 60)
    print("METRICS SUMMARY")
    print("=" * 60)
    for label, m in results.items():
        print(f"  {label}: GSM8K public accuracy = {m['gsm8k_public_accuracy']:.4f}  (checkpoint: {m['checkpoint_evaluated']})")
    out_json = output_base / "baselines_metrics.json"
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll metrics saved to {out_json}")
    print("=" * 60)


if __name__ == "__main__":
    main()
