#!/bin/bash

#SBATCH --job-name="mmlu_evaluation"
#SBATCH --output="/tud/outputs/mmlu_evaluation_%j.out"
#SBATCH --error="/tud/outputs/mmlu_evaluation_%j.err"
#SBATCH --gres=gpu:1
#SBATCH -N 1 
#SBATCH -A nlp-dept 
#SBATCH -p nlp-dept 
#SBATCH -q nlp-pool
#SBATCH --time=24:00:00

# Load modules or activate your conda environment if necessary
# module load cuda/11.1
# source activate your_env

# Print the available GPUs
nvidia-smi
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Define base output directory and adapters directory
BASE_OUTPUT_DIR="/tud/value-adapters/culturellm/eval_outputs/mmlu"
ADAPTERS_DIR="/tud/models/baseline"

# Define adapter languages (culture adapters)
adapter_languages=("arabic" "bengali" "chinese" "english" "german" "greek" "korean" "portuguese" "spanish" "turkish")
# adapter_languages=("turkish")

# Define model names
models=("meta-llama/Meta-Llama-3.1-8B")
# models=("meta-llama/Meta-Llama-3.1-8B-Instruct")


################################
# 2. Zero-Shot Evaluations    #
################################

echo "Starting Zero-Shot Evaluations..."

for MODEL_NAME in "${models[@]}"; do
    MODEL_TYPE="${MODEL_NAME#*/}"

    # Define output directory for storing zero-shot evaluation results
    OUTPUT_DIR="${BASE_OUTPUT_DIR}/zero_shot/${MODEL_TYPE}"

    mkdir -p "$OUTPUT_DIR"

    echo "Running Zero-Shot Evaluation for Model: $MODEL_NAME"

    # Run the evaluation script without adapters
    python -m src.test_mmlu \
        --adapter_lang "none" \
        --test_lang "english" \
        --task "mmlu_science" \
        --output_dir "$OUTPUT_DIR" \
        --context "" \
        --model_name "$MODEL_NAME" \
        --model_type "$MODEL_TYPE" \
        --eval_type zero_shot \
        # --debug 
done


################################
# 1. Normal Evaluations    #
################################

# Iterate over models
for MODEL_NAME in "${models[@]}"; do
    MODEL_TYPE="${MODEL_NAME#*/}"

    # Iterate over adapter languages
    for adapter_lang in "${adapter_languages[@]}"; do
        # Define the path to the adapter
        ADAPTER_PATH="${ADAPTERS_DIR}/${MODEL_TYPE}/${adapter_lang}"

        # Define output directory for storing evaluation results
        OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODEL_TYPE}/${adapter_lang}"

        mkdir -p "$OUTPUT_DIR"

        echo "Running evaluation for Model: $MODEL_NAME, Adapter: $adapter_lang"

        # Run the evaluation script
        python -m src.test_mmlu \
            --adapter_lang "$adapter_lang" \
            --test_lang "$adapter_lang" \
            --task "mmlu_science" \
            --output_dir "$OUTPUT_DIR" \
            --context "" \
            --model_name "$MODEL_NAME" \
            --model_type "$MODEL_TYPE" \
            --eval_type single \
            --lora_path "$ADAPTER_PATH" \
            # --debug False
    done
done



##############################################
# 3. Translated Adapter Cultures Evaluations #
##############################################

echo "Starting Translated Adapter Evaluations..."

TRANSLATED_ADAPTERS_DIR="/tud/models/translated"

for MODEL_NAME in "${models[@]}"; do
    MODEL_TYPE="${MODEL_NAME#*/}"

    for adapter_lang in "${adapter_languages[@]}"; do

        # Define the path to the translated adapter
        TRANSLATED_ADAPTER_PATH="${TRANSLATED_ADAPTERS_DIR}/${MODEL_TYPE}/${adapter_lang}"

        # Define output directory for storing translated adapter evaluation results
        OUTPUT_DIR="${BASE_OUTPUT_DIR}/translated/${MODEL_TYPE}/${adapter_lang}"

        mkdir -p "$OUTPUT_DIR"

        echo "Running Translated Adapter Evaluation for Model: $MODEL_NAME, Adapter: $adapter_lang"

        # Check if the translated adapter exists
        if [ -d "$TRANSLATED_ADAPTER_PATH" ]; then
            # Run the evaluation script with translated adapters
            python -m src.test_mmlu \
                --adapter_lang "$adapter_lang" \
                --test_lang "$adapter_lang" \
                --task "mmlu_science" \
                --output_dir "$OUTPUT_DIR" \
                --context "" \
                --model_name "$MODEL_NAME" \
                --model_type "$MODEL_TYPE" \
                --eval_type single \
                --lora_path "$TRANSLATED_ADAPTER_PATH" \
                # --debug False
        else
            echo "Translated adapter not found for language: $adapter_lang at path: $TRANSLATED_ADAPTER_PATH. Skipping..."
        fi
    done
done

echo "All Evaluations Completed."