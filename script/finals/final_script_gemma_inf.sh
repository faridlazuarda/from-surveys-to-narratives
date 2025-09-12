#!/bin/bash

#SBATCH --job-name=test_job               # Job name
#SBATCH --output=value_adapters/output_%a.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1                        # Run on a single CPU
#SBATCH --mem=40G                         # Total RAM to be used
#SBATCH --cpus-per-task=64                # Number of CPU cores
#SBATCH --gres=gpu:1                      # Number of GPUs (per node)
#SBATCH -p cscc-gpu-p                     # Use the gpu partition
#SBATCH --time=8:00:00                    # Specify the time needed for your job
#SBATCH -q cscc-gpu-qos                   # Specify the QoS

# =========================== Environment Setup ================================

echo "Starting Gemma Training and Testing Script"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Load necessary modules or activate environments if required
# module load python/3.8
# source /path/to/your/venv/bin/activate

# =========================== Variable Definitions =============================

# Define Model
declare -A models
models["gemma"]="google/gemma-2-9b-it"      # Replace with actual HuggingFace model name
models["qwen"]="Qwen/Qwen2.5-7B-Instruct"   # Replace with actual HuggingFace model name
# models["llama"]="meta-llama/Meta-Llama-3.1-8B"   # Replace with actual HuggingFace model name
models["llamainst"]="meta-llama/Meta-Llama-3.1-8B-Instruct"   # Replace with actual HuggingFace model name

# Define Training Scenarios
TRAINING_SCENARIOS=("normad_only")

# List of Cultures
cultures=(
    "arabic" 
    "bengali" 
    "chinese" 
    "english" 
    "german" 
    "greek" 
    "korean" 
    "portuguese" 
    "spanish" 
    "turkish" 
    "arabic,bengali,chinese,english,german,greek,korean,portuguese,spanish,turkish"
)

# WandB API Key (Ensure this is securely managed)
WANDB_API_KEY=""

# Base Directories
OUT_DIR="tud/models"
DATA_PATH="tud/value-adapters/culturellm/data"
WANDB_DIR="tud/wandb"

# Define Combined Tasks per Language for Testing
declare -A combined_tasks
combined_tasks[arabic]='vulgar_detect_mp spam_detect hate_detect_fine-grained'
combined_tasks[bengali]='hate_detect_religion offensive_detect_1 offensive_detect_2 offensive_detect_3 racism_detect threat_detect'
combined_tasks[chinese]='bias_on_gender_detect spam_detect'
combined_tasks[greek]='offensive_detect offensive_detect_g'
combined_tasks[english]='hate_detect_2 hate_offens_detect hostility_directness_detect offensive_detect_easy threat_detect toxicity_detect'
combined_tasks[german]='hate_detect hate_off_detect hate_detect_iwg_1 hate_detect_check offensive_detect_eval'
combined_tasks[korean]='abusive_detect abusive_detect_2 abusive_detect_4 hate_detect_3 hate_detect_6 hate_detect_7'
combined_tasks[portuguese]='homophobia_detect insult_detect misogyny_detect offensive_detect_2 offensive_detect_3'
combined_tasks[spanish]='offensive_detect_ami offensive_detect_mex_a3t offensive_detect_mex_offend hate_detect_eval hate_detect_haterNet stereotype_detect mockery_detect insult_detect improper_detect aggressiveness_detect negative_stance_detect'
combined_tasks[turkish]='offensive_detect offensive_detect_corpus offensive_detect_finegrained offensive_detect_kaggle offensive_detect_kaggle2 abusive_detect spam_detect'

# Define Test Languages
test_languages=("arabic" "bengali" "chinese" "english" "german" "greek" "korean" "portuguese" "spanish" "turkish")

# Define Models for Testing
testing_models=("llamainst" "gemma" "qwen")
# testing_models=("qwen")

# =========================== Function Definitions ==============================

# Function to perform testing
test_model() {
    local model_key=$1
    local model_name=${models[$model_key]}
    local model_type=${model_name#*/}
    local training_scenario=$2

    echo "============================================="
    echo "Testing Model: $model_name"
    echo "Scenario: $training_scenario"
    echo "============================================="

    # Define directories
    BASE_EVAL_DIR="tud/value-adapters/culturellm/eval_outputs"
    ADAPTERS_BASE_DIR="${OUT_DIR}/${training_scenario}"

    # Iterate over adapter languages
    for adapter_lang in "${cultures[@]}"; do
        echo "Evaluating Adapter Language: $adapter_lang"

        # Iterate over test languages
        for test_lang in "${test_languages[@]}"; do
            echo "Testing on Language: $test_lang"

            # Iterate over tasks
            for task in ${combined_tasks[$test_lang]}; do
                echo "Evaluating Task: $task"

                # Define adapter path
                adapter_path="${ADAPTERS_BASE_DIR}/${model_key}/${adapter_lang}"
                echo "Adapter Path: $adapter_path"

                # Define output directory for evaluation results
                output_dir="${BASE_EVAL_DIR}/${training_scenario}"
                mkdir -p "${output_dir}"

                # Check if files already exist
                result_file="${output_dir}/${model_type}/${adapter_lang,,}/${test_lang,,}/${task}_res.jsonl"
                prediction_file="${output_dir}/${model_type}/${adapter_lang,,}/${test_lang,,}/${task}_predictions.jsonl"

                if [ -f "$result_file" ] && [ -f "$prediction_file" ]; then
                    echo "Results already exist for task ${task} (${adapter_lang} -> ${test_lang}). Skipping..."
                    continue
                fi

                # Execute Evaluation
                CUDA_VISIBLE_DEVICES=3 python -m src.test_cross \
                    --adapter_lang "${adapter_lang}" \
                    --test_lang "${test_lang}" \
                    --task "${task}" \
                    --output_dir "${output_dir}" \
                    --context False \
                    --model_name "${model_name}" \
                    --model_type "${model_type}" \
                    --lora_path "${adapter_path}" \
                    --eval_type "${training_scenario}"
            done
        done
    done
}

# =========================== Testing Phase =====================================

echo "Starting Testing Phase"

# Iterate over each model (only Gemma in this case)
for model_key in "${!models[@]}"; do
    echo "==============================="
    echo "Testing Model: $model_key"
    echo "==============================="

    # Iterate over each training scenario
    for training_scenario in "${TRAINING_SCENARIOS[@]}"; do
        echo ">>> Scenario: $training_scenario <<<"

        test_model "${model_key}" "${training_scenario}"

        echo "Completed Testing for Scenario: $training_scenario on Model: $model_key"
    done

    echo "Completed Testing for Model: $model_key"
done

echo "Testing Phase Completed"
echo "Gemma Training and Testing Script Finished Successfully"
