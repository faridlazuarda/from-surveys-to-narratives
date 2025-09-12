#!/bin/bash
#SBATCH --job-name=fsdp_valadapt
#SBATCH --output=indoaksara/slurm_output/fsdp_valadapt.%A.txt
#SBATCH --nodes=1
#SBATCH --mem=200G
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH -p long
#SBATCH --time=72:00:00
#SBATCH -q gpu-12

# =========================== Environment Setup ================================

echo "Starting FSDP LoRA Training and Testing Script"
echo "Number of GPUs allocated: $SLURM_GPUS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Verify GPU allocation
if [ -z "$SLURM_GPUS" ]; then
    echo "No GPUs allocated by SLURM"
    exit 1
fi

# Load necessary modules (uncomment and modify as needed)
# module load python/3.8
# source /path/to/your/venv/bin/activate

# =========================== Variable Definitions =============================

# Define Model
declare -A models
models["gemma"]="google/gemma-7b-it"        # Can use larger models with FSDP
models["qwen"]="Qwen/Qwen1.5-14B-Chat"      # Can use larger models with FSDP
models["llama"]="meta-llama/Llama-2-13b"    # Can use larger models with FSDP
models["llamainst"]="meta-llama/Llama-2-13b-chat"

# Define Training Scenarios
TRAINING_SCENARIOS=("normal" "cultural_context" "normad_context")

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

# Load WandB API Key from environment or use default
WANDB_API_KEY="${WANDB_API_KEY:-"f8885d645b26e9b2b2fac972e456d72947c5e436"}"

# Base Directories
OUT_DIR="tud/models"
DATA_PATH="tud/value-adapters/culturellm/data"
WANDB_DIR="tud/wandb"

# Create necessary directories
for dir in "$OUT_DIR" "$DATA_PATH" "$WANDB_DIR"; do
    if [ ! -d "$dir" ]; then
        echo "Creating directory: $dir"
        mkdir -p "$dir"
    fi
done

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
testing_models=("qwen" "llama" "llamainst" "gemma")

# =========================== Function Definitions ==============================

# Function to perform FSDP training
train_model() {
    local model_key=$1
    local model_name=${models[$model_key]}
    local training_scenario=$2
    local culture=$3

    echo "---------------------------------------------"
    echo "Training Model: $model_name"
    echo "Scenario: $training_scenario"
    echo "Culture: $culture"
    echo "Using FSDP with 4 GPUs"
    echo "---------------------------------------------"

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
        llama_files=${llama_files%,}
    else
        llama_file=$(find "${DATA_PATH}/${culture}/Finetune/" -name "*llama*" | head -n 1)
        if [ -n "$llama_file" ]; then
            llama_files=$llama_file
        else
            echo "No files containing 'llama' found for culture: ${culture}"
            return 1
        fi
    fi

    echo "Data files to be used: $llama_files"

    # Define output path
    out_path="${OUT_DIR}/${training_scenario}/${model_key}/${culture}"
    mkdir -p "${out_path}"

    # Execute Training with FSDP
    if ! torchrun --nproc_per_node=4 --master_port=$(shuf -i 29500-29999 -n 1) \
        -m src.llama_finetune \
        --model_name "${model_name}" \
        --culture "${culture}" \
        --wandb_api_key "${WANDB_API_KEY}" \
        --wandb_name "valadapt_${training_scenario}_${model_key}_${culture}" \
        --out_path "${out_path}" \
        --data_files "${llama_files}" \
        --override_results \
        --use_fsdp \
        --fsdp_config "full_shard" \
        --mixed_precision "fp16" \
        --gradient_checkpointing \
        --gradient_accumulation_steps 4 \
        --per_device_train_batch_size 4; then
        echo "Training failed for model: ${model_name}, culture: ${culture}"
        return 1
    fi
}

# Function to perform testing (single GPU is fine for inference)
test_model() {
    local model_key=$1
    local model_name=${models[$model_key]}
    local model_type=${model_name#*/}
    local training_scenario=$2
    local gpu_id=$3

    echo "============================================="
    echo "Testing Model: $model_name"
    echo "Scenario: $training_scenario"
    echo "Using GPU: $gpu_id"
    echo "============================================="

    # Define directories
    BASE_EVAL_DIR="tud/value-adapters/culturellm/eval_outputs/mmlu"
    ADAPTERS_BASE_DIR="${OUT_DIR}/${training_scenario}"

    # Iterate over adapter languages
    for adapter_lang in "${test_languages[@]}"; do
        echo "Evaluating Adapter Language: $adapter_lang"
        
        # Iterate over test languages
        for test_lang in "${test_languages[@]}"; do
            echo "Testing Language: $test_lang"
            
            # Get tasks for the test language
            IFS=' ' read -ra tasks <<< "${combined_tasks[$test_lang]}"
            
            # Iterate over tasks
            for task in "${tasks[@]}"; do
                echo "Evaluating Task: $task"

                adapter_path="${ADAPTERS_BASE_DIR}/${model_key}/${adapter_lang}"
                echo "Adapter Path: $adapter_path"

                output_dir="${BASE_EVAL_DIR}/${training_scenario}"
                mkdir -p "${output_dir}"

                result_file="${output_dir}/${model_type}/${adapter_lang,,}/${test_lang,,}/${task}_res.jsonl"
                prediction_file="${output_dir}/${model_type}/${adapter_lang,,}/${test_lang,,}/${task}_predictions.jsonl"

                if [ -f "$result_file" ] && [ -f "$prediction_file" ]; then
                    echo "Results already exist for task ${task} (${adapter_lang} -> ${test_lang}). Skipping..."
                    continue
                fi

                # Execute Evaluation with specific GPU
                if ! CUDA_VISIBLE_DEVICES=$gpu_id python -m src.test_mmlu \
                    --adapter_lang "${adapter_lang}" \
                    --test_lang "${test_lang}" \
                    --task "${task}" \
                    --output_dir "${output_dir}" \
                    --context False \
                    --model_name "${model_name}" \
                    --model_type "${model_type}" \
                    --lora_path "${adapter_path}" \
                    --eval_type "${training_scenario}"; then
                    echo "Testing failed for adapter_lang: ${adapter_lang}, test_lang: ${test_lang}"
                    return 1
                fi
            done
        done
    done
}

# =========================== Main Execution ===================================

echo "Starting Training Phase"

# Run training sequentially since each training job uses all GPUs with FSDP
for model_key in "${!models[@]}"; do
    echo "==============================="
    echo "Processing Model: $model_key"
    echo "==============================="

    for training_scenario in "${TRAINING_SCENARIOS[@]}"; do
        echo ">>> Scenario: $training_scenario <<<"

        for culture in "${cultures[@]}"; do
            train_model "${model_key}" "${training_scenario}" "${culture}"
        done
    done
done

echo "Training Phase Completed"

# =========================== Testing Phase ===================================

echo "Starting Testing Phase"

# Calculate total number of testing jobs
total_test_jobs=$((${#testing_models[@]} * ${#TRAINING_SCENARIOS[@]}))
current_test_job=0

# Distribute testing across GPUs (we can parallelize testing)
for model_key in "${testing_models[@]}"; do
    echo "==============================="
    echo "Testing Model: $model_key"
    echo "==============================="

    for training_scenario in "${TRAINING_SCENARIOS[@]}"; do
        # Calculate which GPU to use (0-3) based on job number
        gpu_id=$((current_test_job % 4))
        
        echo ">>> Scenario: $training_scenario <<<"

        test_model "${model_key}" "${training_scenario}" "${gpu_id}" &
        
        # Increment job counter
        ((current_test_job++))
        
        # If we've started jobs on all GPUs, wait for them to complete
        if (( current_test_job % 4 == 0 )); then
            wait
        fi
    done
done

# Wait for any remaining testing jobs
wait

echo "Testing Phase Completed"
echo "FSDP LoRA Training and Testing Script Finished Successfully"