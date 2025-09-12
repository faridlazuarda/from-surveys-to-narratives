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

# Clear CUDA cache and collect garbage
torch.cuda.empty_cache()
gc.collect()

# Load environment variables
load_dotenv()
access_token = os.getenv("HUGGINGFACE_TOKEN")
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
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def eval_model(model, tokenizer):
    """Evaluate the model with a sample prompt."""
    prompt = "Who is Leonardo Da Vinci?"
    pipe = pipeline(task="text-generation", model=model, tokenizer=tokenizer, max_length=200)
    result = pipe(f"<s>[INST] {prompt} [/INST]")
    print(result[0]['generated_text'])


def load_and_prepare_survey_data(data_files):
    """Load and prepare the survey dataset with task-specific prompts."""
    # Load the survey dataset
    survey_dataset = load_dataset('json', data_files=data_files, split='train')

    # Define a function to add task-specific prompt
    def add_survey_prompt(example):
        try:
            question_part = example['text'].split('### Question:')[1].split('### Answer:')[0].strip()
            answer_part = example['text'].split('### Answer:')[1].strip()
            return {
                'text': f"### Task: Survey Question-Answer\n### Question: {question_part}\n### Answer: {answer_part}"
            }
        except IndexError:
            # Handle cases where the expected format is not met
            return {
                'text': f"### Task: Survey Question-Answer\n### Content: {example['text']}"
            }

    # Apply the prompt
    survey_dataset = survey_dataset.map(add_survey_prompt, remove_columns=survey_dataset.column_names)

    return survey_dataset


def load_and_prepare_cultural_context_data(cultural_context_dir, cultures):
    """
    Load and prepare cultural context data with task-specific prompts.
    """
    culture_entries = []

    print(f"Looking for cultural context files in: {cultural_context_dir}")

    for culture in cultures:
        culture_file = os.path.join(cultural_context_dir, f"{culture.lower()}.txt")
        if not os.path.isfile(culture_file):
            print(f"Warning: Cultural context file for '{culture}' not found at '{culture_file}'. Skipping.")
            continue

        with open(culture_file, 'r', encoding='utf-8') as f:
            culture_text = f.read().strip()
            if not culture_text:
                print(f"Warning: Cultural context file for '{culture}' is empty. Skipping.")
                continue

            import re
            sentences = re.split(r'(?<=[.!?]) +', culture_text)
            for sentence in sentences:
                entry = {
                    'text': f"### Task: Cultural Context\n### Culture: {culture.capitalize()}\n### Description: {sentence}"
                }
                culture_entries.append(entry)

    if not culture_entries:
        raise ValueError("No valid cultural context data found. Please check the cultural context files.")

    cultural_dataset = Dataset.from_list(culture_entries)
    return cultural_dataset


def load_and_concatenate_datasets(data_files):
    """Load and concatenate multiple JSONL datasets."""
    dataset_list = [load_dataset('json', data_files=file, split='train') for file in data_files.split(',')]
    return concatenate_datasets(dataset_list)


def load_and_prepare_normad_data(cultures):
    """
    Load and prepare NormAd data similar to cultural_context scenario but from the NormAd dataset.
    We will map the given culture (language proxy) to a list of countries and filter NormAd dataset accordingly.
    """

    # Mapping cultures (languages) to countries
    language_to_countries = {
        "arabic": ["egypt", "lebanon", "sudan", "iraq", "saudi_arabia", "palestinian_territories", "syria"],
        "bengali": ["bangladesh"],
        "chinese": ["china", "hong_kong", "taiwan", "singapore"],
        "english": ["united_kingdom", "united_states_of_america", "australia", "new_zealand", "canada", "ireland"],
        "german": ["germany", "austria"],
        "greek": ["greece", "cyprus"],
        "korean": ["south_korea"],
        "portuguese": ["portugal", "brazil"],
        "spanish": ["spain", "argentina", "colombia", "chile", "venezuela", "mexico", "peru"],
        "turkish": ["türkiye"]
    }

    # If multiple cultures are provided (like "arabic,bengali"), combine their country lists
    selected_countries = []
    for culture in cultures:
        if culture in language_to_countries:
            selected_countries.extend(language_to_countries[culture])
        else:
            print(f"Warning: No country mapping found for culture '{culture}'. Skipping.")

    # Remove duplicates if any
    selected_countries = list(set(selected_countries))

    if not selected_countries:
        raise ValueError("No countries selected for NormAd data. Please check the culture and mappings.")

    # Load the NormAd dataset
    normad_dataset = load_dataset("akhilayerukola/NormAd", split="train")

    # Filter by selected countries
    normad_dataset = normad_dataset.filter(lambda x: x["Country"] in selected_countries)

    # Prepare the text entries
    normad_entries = []
    for example in normad_dataset:
        culture_str = ", ".join(cultures)
        text = (
            f"### Task: NormAd Cultural Context\n"
            f"### Culture: {culture_str.capitalize()}\n"
            f"### Country: {example['Country']}\n"
            f"### Background: {example.get('Background', '')}\n"
            f"### Rule-of-Thumb: {example.get('Rule-of-Thumb', '')}\n"
            f"### Story: {example.get('Story', '')}\n"
            f"### Explanation: {example.get('Explanation', '')}"
        )
        normad_entries.append({"text": text})

    if not normad_entries:
        raise ValueError("No normad context data found after filtering.")

    normad_context_dataset = Dataset.from_list(normad_entries)
    return normad_context_dataset


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a LoRA module per culture for LLM with optional multitask learning.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Set the pre-trained model.")
    parser.add_argument("--culture", type=str, default="arabic", help="Choose the region(s), separate different regions with commas if training the SAME LoRA module on multiple cultures.")
    parser.add_argument("--debug", action="store_true", default=False, help="Only train on the first 10 examples.")
    parser.add_argument("--data_files", type=str, default="/tud/culturellm/data/Arabic/Finetune/WVQ_arabic_Iraq_Jordan_llama.jsonl", help="Path(s) to the survey data file(s), separated by commas if multiple.")
    parser.add_argument("--out_path", type=str, default="./output/xnli/lora", help="Set the output path.")
    parser.add_argument("--wandb_api_key", type=str, default=None, help="WandB API key.")
    parser.add_argument("--wandb_offline", default=False, action="store_true", help="Enable WandB offline mode.")
    parser.add_argument("--override_results", action="store_true", default=False, help="When enabled, remove the previous checkpoints/results.")
    parser.add_argument("--wandb_name", type=str, default="valadapt", help="WandB run name.")
    parser.add_argument(
        "--training_scenario", 
        type=str, 
        default="normal", 
        choices=["normal", "cultural_context", "normad_context", "normad_only"],  # <--- ADDED HERE
        help="Choose the training scenario: 'normal', 'cultural_context', 'normad_context', or 'normad_only'."
    )
    args = parser.parse_args()

    print("Arguments:", args)
    print("Model Name:", args.model_name)
    set_seed()

    # Ensure output path is correctly set
    if len(args.out_path.strip()) == 0:
        args.out_path = f"./output/valadapt/{args.culture}/"

    # Handle existing output directory
    if os.path.exists(args.out_path):
        assert (args.debug or args.override_results), f"Output directory {args.out_path} already exists! Use --override_results to remove it."
        shutil.rmtree(args.out_path)
    os.makedirs(args.out_path)

    # Configure WandB
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
    if args.wandb_api_key:
        os.environ["WANDB_API_KEY"] = args.wandb_api_key

    wandb.init(
        project=os.getenv("WANDB_PROJECT", "VALADAPT"),
        name=args.wandb_name,
        dir="tud/wandb"
    )

    cultures = [culture.strip().lower() for culture in args.culture.split(',')]
    print(f"Training scenario: {args.training_scenario}")
    print(f"Training on cultures: {cultures}")

    # Always load the survey dataset (unless we're skipping it entirely, see 'normad_only' scenario).
    survey_data_files = args.data_files.split(',')
    survey_datasets = []
    for file in survey_data_files:
        if not os.path.isfile(file):
            print(f"Warning: Survey data file '{file}' not found. Skipping.")
            continue
        survey_datasets.append(load_and_prepare_survey_data(file))
    if survey_datasets:
        survey_dataset = concatenate_datasets(survey_datasets)
        print(f"Survey dataset size: {len(survey_dataset)}")
    else:
        survey_dataset = None
        print("Warning: No valid survey data loaded. The 'survey_dataset' is empty or not found.")

    if args.training_scenario == "cultural_context":
        # Load cultural context dataset
        CULTURAL_CONTEXT_DIR = "tud/value-adapters/culturellm/data/cultural_context"
        try:
            cultural_dataset = load_and_prepare_cultural_context_data(CULTURAL_CONTEXT_DIR, cultures)
            print(f"Cultural context dataset size: {len(cultural_dataset)}")
        except ValueError as ve:
            print(str(ve))
            exit(1)

        if survey_dataset is not None:
            train_dataset = concatenate_datasets([survey_dataset, cultural_dataset])
            print(f"Combined dataset size: {len(train_dataset)}")
        else:
            train_dataset = cultural_dataset

    elif args.training_scenario == "normad_context":
        # Load normad context dataset
        try:
            normad_dataset = load_and_prepare_normad_data(cultures)
            print(f"NormAd context dataset size: {len(normad_dataset)}")
        except ValueError as ve:
            print(str(ve))
            exit(1)

        if survey_dataset is not None:
            train_dataset = concatenate_datasets([survey_dataset, normad_dataset])
            print(f"Combined dataset size: {len(train_dataset)}")
        else:
            train_dataset = normad_dataset

    elif args.training_scenario == "normad_only":
        # Load ONLY normad context dataset (skip survey data)
        try:
            normad_dataset = load_and_prepare_normad_data(cultures)
            print(f"NormAd context dataset size: {len(normad_dataset)}")
        except ValueError as ve:
            print(str(ve))
            exit(1)

        # Train only on NormAd data
        train_dataset = normad_dataset

    else:
        # Normal training without cultural or normad context
        print("Training scenario: Normal")
        if survey_dataset is None:
            raise ValueError("No survey data is loaded for normal scenario. Please check your `data_files` paths.")
        train_dataset = survey_dataset

    # Apply debug mode if enabled
    if args.debug:
        print("Debug mode is enabled. Using only the first 10 examples of the dataset.")
        train_dataset = train_dataset.select(range(10))

    # Define quantization configuration
    compute_dtype = torch.float16  # or torch.bfloat16 if supported

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    # Load the pre-trained model with quantization
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quant_config,
        device_map={'': torch.cuda.current_device()},
    )
    model.quantization_config = quant_config
    model.config.use_cache = False  # Disable cache for training

    # Enable gradient checkpointing to save memory
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    # Load and configure the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Define LoRA configuration
    peft_params = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=64,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"]
    )

    # Define training arguments
    training_params = TrainingArguments(
        output_dir=args.out_path,
        num_train_epochs=6,
        per_device_train_batch_size=32,
        auto_find_batch_size=True,
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
        lr_scheduler_type="constant"
    )

    # Initialize the trainer
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

    # Start training
    trainer.train()

    # Save the fine-tuned model and tokenizer
    trainer.model.save_pretrained(args.out_path)
    trainer.tokenizer.save_pretrained(args.out_path)
