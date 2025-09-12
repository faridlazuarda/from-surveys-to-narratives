#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import argparse
import shutil
from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
    logging,
)
import wandb
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig  # Import SFTConfig
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_in_model, dispatch_model
import numpy as np
from huggingface_hub import login
import gc
import random
from dotenv import load_dotenv


# ------------------------------------------------------------------
#  House-keeping
# ------------------------------------------------------------------
torch.cuda.empty_cache()
gc.collect()
load_dotenv()

access_token = ""
if access_token:
    login(token=access_token)
else:
    print("Warning: HUGGINGFACE_TOKEN not found in environment variables.")


def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def eval_model(model, tokenizer):
    """Quick sanity-check generation."""
    prompt = "Who is Leonardo Da Vinci?"
    pipe = pipeline(
        task="text-generation",
        model=model,
        tokenizer=tokenizer,
        max_length=200,
    )
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print(result[0]["generated_text"])


# ------------------------------------------------------------------
#  Dataset helpers
# ------------------------------------------------------------------
def load_and_prepare_survey_data(data_files):
    """
    World-Values-Survey (or other survey) → prompt-formatted.
    Expected original field: `text` containing "### Question: … ### Answer: …"
    """
    survey_dataset = load_dataset("json", data_files=data_files, split="train")

    def add_survey_prompt(example):
        try:
            question_part = (
                example["text"].split("### Question:")[1].split("### Answer:")[0].strip()
            )
            answer_part = example["text"].split("### Answer:")[1].strip()
            new_text = (
                f"### Task: Survey Question-Answer\n"
                f"### Question: {question_part}\n"
                f"### Answer: {answer_part}"
            )
        except IndexError:
            new_text = (
                f"### Task: Survey Question-Answer\n"
                f"### Content: {example['text']}"
            )
        return {"text": new_text}

    return survey_dataset.map(add_survey_prompt, remove_columns=survey_dataset.column_names)


def load_and_prepare_cultural_context_data(cultural_context_dir, cultures):
    """
    Wikipedia / curated cultural snippets.
    Each line/sentence becomes one example with a cultural-context-prompt wrapper.
    """
    culture_entries = []

    print(f"Looking for cultural context files in: {cultural_context_dir}")
    for culture in cultures:
        culture_file = os.path.join(cultural_context_dir, f"{culture.lower()}.txt")
        if not os.path.isfile(culture_file):
            print(
                f"Warning: Cultural context file for '{culture}' not found at '{culture_file}'. Skipping."
            )
            continue

        with open(culture_file, "r", encoding="utf-8") as f:
            culture_text = f.read().strip()
            if not culture_text:
                print(
                    f"Warning: Cultural context file for '{culture}' is empty. Skipping."
                )
                continue

            import re

            sentences = re.split(r"(?<=[.!?]) +", culture_text)
            for sentence in sentences:
                if not sentence.strip():
                    continue
                culture_entries.append(
                    {
                        "text": (
                            "### Task: Cultural Context\n"
                            f"### Culture: {culture.capitalize()}\n"
                            f"### Description: {sentence.strip()}"
                        )
                    }
                )

    if not culture_entries:
        raise ValueError(
            "No valid cultural context data found. Please check the cultural context files."
        )

    return Dataset.from_list(culture_entries)


def load_and_prepare_normad_data(cultures):
    """
    NormAd → cultural context style prompts.
    We map language → list of countries (NormAd index is by country).
    """
    language_to_countries = {
        "arabic": [
            "egypt",
            "lebanon",
            "sudan",
            "iraq",
            "saudi_arabia",
            "palestinian_territories",
            "syria",
        ],
        "bengali": ["bangladesh"],
        "chinese": ["china", "hong_kong", "taiwan", "singapore"],
        "english": [
            "united_kingdom",
            "united_states_of_america",
            "australia",
            "new_zealand",
            "canada",
            "ireland",
        ],
        "german": ["germany", "austria"],
        "greek": ["greece", "cyprus"],
        "korean": ["south_korea"],
        "portuguese": ["portugal", "brazil"],
        "spanish": [
            "spain",
            "argentina",
            "colombia",
            "chile",
            "venezuela",
            "mexico",
            "peru",
        ],
        "turkish": ["türkiye"],
    }

    selected_countries = []
    for culture in cultures:
        if culture in language_to_countries:
            selected_countries.extend(language_to_countries[culture])
        else:
            print(f"Warning: No country mapping found for culture '{culture}'. Skipping.")

    selected_countries = list(set(selected_countries))
    if not selected_countries:
        raise ValueError(
            "No countries selected for NormAd data. Please check the culture and mappings."
        )

    normad_dataset = load_dataset("akhilayerukola/NormAd", split="train")
    normad_dataset = normad_dataset.filter(lambda x: x["Country"] in selected_countries)

    normad_entries = []
    for example in normad_dataset:
        culture_str = ", ".join(cultures)
        normad_entries.append(
            {
                "text": (
                    "### Task: NormAd Cultural Context\n"
                    f"### Culture: {culture_str.capitalize()}\n"
                    f"### Country: {example['Country']}\n"
                    f"### Background: {example.get('Background', '')}\n"
                    f"### Rule-of-Thumb: {example.get('Rule-of-Thumb', '')}\n"
                    f"### Story: {example.get('Story', '')}\n"
                    f"### Explanation: {example.get('Explanation', '')}"
                )
            }
        )

    if not normad_entries:
        raise ValueError("No NormAd context data found after filtering.")

    return Dataset.from_list(normad_entries)


# ------------------------------------------------------------------
#  Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Train a LoRA module per culture for LLM with optional multi-task / "
            "context augmentation."
        )
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Base HF model.",
    )
    parser.add_argument(
        "--culture",
        type=str,
        default="arabic",
        help=(
            "One or more comma-separated cultures/languages (e.g. 'arabic,english')."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Only train on first 10 samples for a quick smoke test.",
    )
    parser.add_argument(
        "--data_files",
        type=str,
        default="/workspace/value-adapters/culturellm/data/Arabic/Finetune/WVQ_arabic_Iraq_Jordan_llama.jsonl",
        help="Comma-separated JSONL paths for survey (WVS) data.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="./output/xnli/lora",
        help="Directory for checkpoints, logs, etc.",
    )
    parser.add_argument(
        "--wandb_api_key", type=str, default=None, help="Weights & Biases API key."
    )
    parser.add_argument(
        "--wandb_offline",
        action="store_true",
        help="Run W&B in offline mode (no network).",
    )
    parser.add_argument(
        "--override_results",
        action="store_true",
        help="Delete any existing content in --out_path.",
    )
    parser.add_argument("--wandb_name", type=str, default="valadapt", help="Run name.")
    parser.add_argument(
        "--training_scenario",
        type=str,
        # default="normal",
        # >>> NEW / CHANGED <<<
        choices=[
            "normal",
            "cultural_context",
            "normad_context",
            "normad_only",
            "wiki_only",
            "all_combined",  # new
        ],
        help=(
            "normal: only WVS/survey | cultural_context: WVS+Wiki | "
            "normad_context: WVS+NormAd | normad_only: only NormAd | "
            "wiki_only: only Wiki/cultural | all_combined: WVS+Wiki+NormAd"
        ),
    )
    args = parser.parse_args()

    print("Arguments:", args)
    set_seed()

    # ------------------------------------------------------------------
    #  Output dir
    # ------------------------------------------------------------------
    if len(args.out_path.strip()) == 0:
        args.out_path = f"./output/valadapt/{args.culture}/"

    if os.path.exists(args.out_path):
        assert (
            args.debug or args.override_results
        ), f"Output directory {args.out_path} exists! Use --override_results."
        shutil.rmtree(args.out_path)
    os.makedirs(args.out_path, exist_ok=True)

    # ------------------------------------------------------------------
    #  W&B
    # ------------------------------------------------------------------
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "VALADAPT"),
        name=args.wandb_name,
        dir="/workspace/value-adapters/culturellm/wandb",
    )

    # ------------------------------------------------------------------
    #  Load datasets
    # ------------------------------------------------------------------
    cultures = [c.strip().lower() for c in args.culture.split(",")]
    print(f"Training scenario       : {args.training_scenario}")
    print(f"Training on cultures    : {cultures}")

    # -- WVS / survey --
    survey_dataset = None
    survey_data_files = args.data_files.split(",")
    if args.training_scenario not in {"normad_only", "wiki_only"}:
        survey_datasets = []
        for fpath in survey_data_files:
            if not os.path.isfile(fpath):
                print(f"Warning: Survey data file '{fpath}' not found. Skipping.")
                continue
            survey_datasets.append(load_and_prepare_survey_data(fpath))
        if survey_datasets:
            survey_dataset = concatenate_datasets(survey_datasets)
            print(f"Survey dataset size     : {len(survey_dataset)}")
        else:
            print("No valid survey data loaded.")

    # -- Scenario branching --------------------------------------------------
    train_dataset = None
    CULTURAL_CONTEXT_DIR = "/workspace/value-adapters/culturellm/data/cultural_context"

    if args.training_scenario == "cultural_context":
        cultural_dataset = load_and_prepare_cultural_context_data(
            CULTURAL_CONTEXT_DIR, cultures
        )
        print(f"Cultural ctx size       : {len(cultural_dataset)}")

        train_dataset = (
            concatenate_datasets([survey_dataset, cultural_dataset])
            if survey_dataset is not None
            else cultural_dataset
        )

    elif args.training_scenario == "normad_context":
        normad_dataset = load_and_prepare_normad_data(cultures)
        print(f"NormAd ctx size         : {len(normad_dataset)}")

        train_dataset = (
            concatenate_datasets([survey_dataset, normad_dataset])
            if survey_dataset is not None
            else normad_dataset
        )

    elif args.training_scenario == "normad_only":
        normad_dataset = load_and_prepare_normad_data(cultures)
        print(f"NormAd ctx size         : {len(normad_dataset)}")
        train_dataset = normad_dataset

    elif args.training_scenario == "wiki_only":
        cultural_dataset = load_and_prepare_cultural_context_data(
            CULTURAL_CONTEXT_DIR, cultures
        )
        print(f"Cultural ctx size       : {len(cultural_dataset)}")
        train_dataset = cultural_dataset

    elif args.training_scenario == "all_combined":
        # Load everything: survey (WVS), Wikipedia cultural context, NormAd.
        cultural_dataset = load_and_prepare_cultural_context_data(
            CULTURAL_CONTEXT_DIR, cultures
        )
        normad_dataset = load_and_prepare_normad_data(cultures)

        print(f"Cultural ctx size       : {len(cultural_dataset)}")
        print(f"NormAd ctx size         : {len(normad_dataset)}")

        datasets_to_concat = [cultural_dataset, normad_dataset]
        if survey_dataset is not None:
            datasets_to_concat.insert(0, survey_dataset)
        else:
            print("Warning: No survey data – 'all_combined' will omit WVS.")

        train_dataset = concatenate_datasets(datasets_to_concat)
        print(f"Combined dataset size   : {len(train_dataset)}")

    else:  # normal
        if survey_dataset is None:
            raise ValueError(
                "No survey data loaded for 'normal' scenario – check --data_files."
            )
        train_dataset = survey_dataset

    # ------------------------------------------------------------------
    #  Debug slice
    # ------------------------------------------------------------------
    if args.debug:
        print("DEBUG MODE: using only 10 samples.")
        train_dataset = train_dataset.select(range(10))

    # ------------------------------------------------------------------
    #  Quantisation & model
    # ------------------------------------------------------------------
    compute_dtype = torch.float16
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        device_map={"": torch.cuda.current_device()},
    )
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # ------------------------------------------------------------------
    #  PEFT / LoRA
    # ------------------------------------------------------------------
    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )

    # Batch size heuristics
    if "7b" in args.model_name.lower():
        bs = 6
    elif "8b" in args.model_name.lower():
        bs = 6
    elif "9b" in args.model_name.lower():
        bs = 2
    else:
        bs = 8

    # ------------------------------------------------------------------
    training_params = TrainingArguments(
        output_dir=args.out_path,
        num_train_epochs=6,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=1,
        optim="paged_adamw_8bit",
        logging_steps=250,
        save_steps=250,
        save_strategy="steps",
        learning_rate=2e-4,
        weight_decay=0.001,
        fp16=False,
        bf16=False,
        max_grad_norm=0.3,
        max_steps=-1,
        warmup_ratio=0.03,
        group_by_length=True,
        lr_scheduler_type="constant",
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        peft_config=peft_params,
        dataset_text_field="text",
        max_seq_length=None,
        tokenizer=tokenizer,
        args=training_params,
        packing=False,
    )

    trainer.train()

    trainer.model.save_pretrained(args.out_path)
    trainer.tokenizer.save_pretrained(args.out_path)
