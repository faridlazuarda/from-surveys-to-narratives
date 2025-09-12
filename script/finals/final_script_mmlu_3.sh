#!/bin/bash

#SBATCH --job-name=test_job               # Job name
#SBATCH --output=tud/slurm-outputs/mmlu/output_%A.txt
#SBATCH --nodes=1
#SBATCH --ntasks=1                        # Run on a single CPU
#SBATCH --mem=40G                         # Total RAM to be used
#SBATCH --cpus-per-task=64                # Number of CPU cores
#SBATCH --gres=gpu:1                      # Number of GPUs (per node)
#SBATCH -p cscc-gpu-p                     # Use the gpu partition
#SBATCH --time=12:00:00                    # Specify the time needed for your job
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
# models["gemma"]="google/gemma-2-9b-it"      # Replace with actual HuggingFace model name
models["qwen"]="Qwen/Qwen2.5-7B-Instruct"   # Replace with actual HuggingFace model name
# models["llama"]="meta-llama/Meta-Llama-3.1-8B"   # Replace with actual HuggingFace model name
# models["llamainst"]="meta-llama/Meta-Llama-3.1-8B-Instruct"   # Replace with actual HuggingFace model name

# Define Training Scenarios
TRAINING_SCENARIOS=("normal" "cultural_context")

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
    # "arabic,bengali,chinese,english,german,greek,korean,portuguese,spanish,turkish"
)

# WandB API Key (Ensure this is securely managed)
WANDB_API_KEY=""

# Base Directories
OUT_DIR="/workspace/value-adapters/culturellm/models"
DATA_PATH="/workspace/value-adapters/culturellm/data"
WANDB_DIR="/workspace/value-adapters/culturellm/wandb"

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
testing_models=("llama")

# =========================== Function Definitions ==============================


# Function to perform training
train_model() {
    local model_key=$1
    local model_name=${models[$model_key]}
    local training_scenario=$2
    local culture=$3

    echo "---------------------------------------------"
    echo "Training Model: $model_name"
    echo "Scenario: $training_scenario"
    echo "Culture: $culture"
    echo "---------------------------------------------"
    # Define output path
    # Define output path
    out_path="${OUT_DIR}/${training_scenario}/${model_key}/${culture}"
    
    # Check if directory exists and contains required files
    if [ -d "$out_path" ] && \
       [ -n "$(find "$out_path" -name "*.safetensors")" ] && \
       [ -f "$out_path/adapter_config.json" ] && \
       [ -f "$out_path/special_tokens_map.json" ] && \
       [ -f "$out_path/tokenizer_config.json" ] && \
       [ -f "$out_path/tokenizer.json" ]; then
        echo "Training output directory contains all required files: $out_path"
        echo "Found: "
        echo "- Safetensors: $(find "$out_path" -name "*.safetensors")"
        echo "- Adapter config"
        echo "- Tokenizer files"
        echo "Skipping training for this configuration..."
        return 0
    else
        echo "Directory missing required files. Starting training..."
    fi
    # Handle combined cultures
    if [[ "$culture" == *","* ]]; then
        IFS=',' read -ra CULTURE_ARRAY <<< "$culture"
        llama_files=""
        for sub_culture in "${CULTURE_ARRAY[@]}"; do
            llama_file=$(find "${DATA_PATH}/${sub_culture}/Finetune/" -name "*llama*" | head -n 1)
            if [ -n "$llama_file" ]; then
                llama_files+="${llama_file},"
            else
                echo "No files containing 'llama' found for culture: ${sub_culture}"
            fi
        done
        # Remove trailing comma
        llama_files=${llama_files%,}
    else
        # Handle single culture case
        llama_file=$(find "${DATA_PATH}/${culture}/Finetune/" -name "*llama*" | head -n 1)
        if [ -n "$llama_file" ]; then
            llama_files=$llama_file
        else
            echo "No files containing 'llama' found for culture: ${culture}"
            return
        fi
    fi

    echo "Data files to be used: $llama_files"

    # Define output path
    out_path="${OUT_DIR}/${training_scenario}/${model_key}/${culture}"
    mkdir -p "${out_path}"

    # Execute Training
    CUDA_VISIBLE_DEVICES=2 python -m src.llama_finetune \
        --model_name "${model_name}" \
        --culture "${culture}" \
        --wandb_api_key "${WANDB_API_KEY}" \
        --wandb_name "valadapt_${training_scenario}_${model_key}_${culture}" \
        --out_path "${out_path}" \
        --data_files "${llama_files}" \
        --training_scenario "${training_scenario}" \
        --override_results
}



# Function to perform testing
test_model() {
    local model_key=$1
    local model_name=${models[$model_key]}
    local model_type=${model_name#*/}
    local training_scenario=$2
    local batch_size=${3:-8}  # Default batch size of 8 if not provided

    echo "============================================="
    echo "Testing Model: $model_name"
    echo "Scenario: $training_scenario"
    echo "Batch Size: $batch_size"
    echo "============================================="

    # Define directories
    BASE_EVAL_DIR="/workspace/value-adapters/culturellm/eval_outputs/mmlu"
    ADAPTERS_BASE_DIR="${OUT_DIR}/${training_scenario}"

    # Iterate over adapter languages
    for adapter_lang in "${cultures[@]}"; do
        echo "Evaluating Adapter Language: $adapter_lang"
        
        # For MMLU, we use a fixed set of tasks
        task="mmlu"
        echo "Evaluating Task: $task"

        # Define adapter path
        adapter_path="${ADAPTERS_BASE_DIR}/${model_key}/${adapter_lang}"
        echo "Adapter Path: $adapter_path"

        # Define output directory for evaluation results
        output_dir="${BASE_EVAL_DIR}/${training_scenario}/${model_key}/${adapter_lang}"
        mkdir -p "${output_dir}"

        # Check if files already exist
        result_file="${output_dir}/${task}_res.jsonl"
        prediction_file="${output_dir}/${task}_predictions.jsonl"

        if [ -f "$result_file" ] && [ -f "$prediction_file" ]; then
            echo "Results already exist for task ${task} (${adapter_lang}). Skipping..."
            continue
        fi

        # Create necessary subdirectories
        # mkdir -p "${output_dir}/${model_type}/${adapter_lang,,}"

        # Execute Evaluation with batch processing
        CUDA_VISIBLE_DEVICES=2 python -m src.test_mmlu \
            --adapter_lang "${adapter_lang}" \
            --task "${task}" \
            --output_dir "${output_dir}" \
            --context False \
            --model_name "${model_name}" \
            --model_type "${model_type}" \
            --lora_path "${adapter_path}" \
            --eval_type "${training_scenario}" \
            # --batch_size "${batch_size}"

        # Check if the evaluation was successful
        if [ $? -ne 0 ]; then
            echo "Error occurred during evaluation of task ${task} (${adapter_lang})"
            # Optionally, you can add error handling here
        fi
    done
}




echo "Starting Training Phase"

# Iterate over each model (only Gemma in this case)
for model_key in "${!models[@]}"; do
    echo "==============================="
    echo "Processing Model: $model_key"
    echo "==============================="

    # Iterate over each training scenario
    for training_scenario in "${TRAINING_SCENARIOS[@]}"; do
        echo ">>> Scenario: $training_scenario <<<"

        # Iterate over each culture
        for culture in "${cultures[@]}"; do
            train_model "${model_key}" "${training_scenario}" "${culture}"
        done

        echo "Completed Scenario: $training_scenario for Model: $model_key"
    done

    echo "Completed Training for Model: $model_key"
done

echo "Training Phase Completed"

# =========================== Testing Phase =====================================

# Set default batch size
BATCH_SIZE=64  # You can adjust this value


# Iterate over each training scenario
for training_scenario in "${TRAINING_SCENARIOS[@]}"; do
    echo "==============================="
    echo ">>> Starting Scenario: $training_scenario <<<"
    # echo "Model name: ${models[$model_key]}"
    echo "Batch size: $BATCH_SIZE"
    echo "==============================="
        
    # Iterate over each model (only Gemma in this case)
    for model_key in "${!models[@]}"; do
        echo "Testing Model: $model_key"
        # Run test_model with batch processing
        test_model "${model_key}" "${training_scenario}" "${BATCH_SIZE}"
        
        # Check if test_model was successful
        if [ $? -eq 0 ]; then
            echo "✓ Successfully completed testing for scenario: $training_scenario"
        else
            echo "✗ Error occurred during testing for scenario: $training_scenario"
            # Optionally, you could exit here if you want to stop on first error
            # exit 1
        fi
        
        echo ">>> Completed Scenario: $training_scenario <<<"
    done

    echo "Completed all testing scenarios for model: $model_key"
done

echo "MMLU Testing Phase Completed"
echo "Script execution finished"