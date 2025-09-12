#!/bin/bash
#SBATCH --job-name="sc-1"
#SBATCH --output="/tud/outputs/run_self_check.out"
#SBATCH --gres=gpu:1
#SBATCH -N 1 
#SBATCH -A nlp-dept 
#SBATCH -p nlp-dept 
#SBATCH -q nlp-pool

nvidia-smi

echo $CUDA_VISIBLE_DEVICES

# Define the base path to the root directory containing all language folders
BASE_PATH="/tud/value-adapters/culturellm/data"
OUTPUT_DIR="/tud/value-adapters/culturellm/eval_outputs/self-check"

# Define the languages and their corresponding filenames
declare -A cultures
cultures=(
    [arabic]="WVQ_arabic_Iraq_Jordan_llama.jsonl"
    [bengali]="WVQ_Bengali_llama.jsonl"
    [chinese]="WVQ_China_llama.jsonl"
    [english]="WVQ_English_llama.jsonl"
    [german]="WVQ_Germany_llama.jsonl"
    [greek]="WVQ_Greece_llama.jsonl"
    [korean]="WVQ_Korean_llama.jsonl"
    [portuguese]="WVQ_Portuguese_llama.jsonl"
    [spanish]="WVQ_Spanish_llama.jsonl"
    [turkish]="WVQ_Turkey_llama.jsonl"
)

# Define the models, including both vanilla and adapter-based models
models=("meta-llama/Meta-Llama-3.1-8B" "meta-llama/Meta-Llama-3.1-8B-Instruct")
base_adapter_paths=("/valadapt-meta-llama-3.1-8b" "/valadapt-llama-3.1-8B-it")

# Loop through each model
for model in "${models[@]}"; do
    # Determine if this is a vanilla model (no adapter)
    is_vanilla=false
    if [[ "$model" == *"vanilla"* ]]; then
        is_vanilla=true
    fi

    # Determine the corresponding base adapter path prefix if not vanilla
    if [[ "$model" == *"Instruct"* ]]; then
        base_adapter_path_prefix="${base_adapter_paths[1]}"
    else
        base_adapter_path_prefix="${base_adapter_paths[0]}"
    fi

    # Loop through each culture and run the test
    for lang in "${!cultures[@]}"; do
        data_file="${BASE_PATH}/${lang}/Finetune/${cultures[$lang]}"

        if [ "$is_vanilla" = true ]; then
            echo "Running vanilla model ${model} on ${lang} dataset"
            python -m src.self_check \
                --model_name "$model" \
                --data_file "$data_file" \
                --output_dir "$OUTPUT_DIR"
        else
            lora_path="${base_adapter_path_prefix}-${lang}"
            echo "Running model ${model} with adapter on ${lang} dataset"
            python -m src.self_check \
                --model_name "$model" \
                --lora_path "$lora_path" \
                --data_file "$data_file" \
                --output_dir "$OUTPUT_DIR"
        fi
    done
done
