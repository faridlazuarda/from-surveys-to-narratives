# =========================== Environment Setup ================================

DEVICES=1

echo "Starting Gemma Training and Testing Script"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

conda activate valadapt

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
TRAINING_SCENARIOS=("all_combined" "wiki_only")

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
OUT_DIR="/workspace/value-adapters/models"
DATA_PATH="/workspace/value-adapters/culturellm/data"
WANDB_DIR="/workspace/value-adapters/wandb"

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
testing_models=("qwen")
# =========================== Function Definitions =============================

# -----------------------------------------------------------------------------#
# Fine‑tuning adapters                                                         #
# -----------------------------------------------------------------------------#
train_model() {
  local model_key=$1
  local model_name=${models[$model_key]}
  local training_scenario=$2
  local culture=$3

  echo "---------------------------------------------"
  echo "Training → model: $model_name | scenario: $training_scenario | culture: $culture"
  echo "---------------------------------------------"

  local out_path="${OUT_DIR}/${training_scenario}/${model_key}/${culture}"
  if [[ -d "$out_path" ]]; then
    echo "✔ training already done → $out_path"
    return
  fi

  # -------- Collect llama files for culture(s) -----------
  local llama_files=""
  if [[ "$culture" == *","* ]]; then
    IFS=',' read -ra CULTURE_ARRAY <<< "$culture"
    for sub_culture in "${CULTURE_ARRAY[@]}"; do
      local f=$(find "${DATA_PATH}/${sub_culture}/Finetune/" -name "*llama*" | head -n 1)
      [[ -n "$f" ]] && llama_files+="${f},"
    done
    llama_files=${llama_files%,}    # trim trailing comma
  else
    local f=$(find "${DATA_PATH}/${culture}/Finetune/" -name "*llama*" | head -n 1)
    [[ -z "$f" ]] && { echo "✖ no llama files for $culture"; return; }
    llama_files=$f
  fi
  echo "Data files: $llama_files"

  mkdir -p "$out_path"
  CUDA_VISIBLE_DEVICES="$DEVICES" python -m src.llama_finetune \
      --model_name   "$model_name" \
      --culture      "$culture" \
      --wandb_api_key "$WANDB_API_KEY" \
      --training_scenario "$training_scenario" \
      --wandb_name   "valadapt_${training_scenario}_${model_key}_${culture}" \
      --out_path     "$out_path" \
      --data_files   "$llama_files" \
      --override_results
}

# -----------------------------------------------------------------------------#
# Cross‑lingual task evaluation (non‑MMLU)                                     #
# -----------------------------------------------------------------------------#
test_model_cross() {
  local model_key=$1
  local model_name=${models[$model_key]}
  local model_type=${model_name#*/}
  local training_scenario=$2

  echo "============================================="
  echo "Cross‑lingual eval → model: $model_name | scenario: $training_scenario"
  echo "============================================="

  local BASE_EVAL_DIR="/workspace/value-adapters/culturellm/eval_outputs"
  local ADAPTERS_BASE_DIR="${OUT_DIR}/${training_scenario}"

  for adapter_lang in "${cultures[@]}"; do
    for test_lang in "${test_languages[@]}"; do
      for task in ${combined_tasks[$test_lang]}; do
        local adapter_path="${ADAPTERS_BASE_DIR}/${model_key}/${adapter_lang}"
        local output_dir="${BASE_EVAL_DIR}/${training_scenario}/${model_type}/${adapter_lang}/${test_lang}"
        mkdir -p "$output_dir"

        local res="${output_dir}/${task}_res.jsonl"
        local pred="${output_dir}/${task}_predictions.jsonl"
        if [[ -f "$res" && -f "$pred" ]]; then
          echo "✔ ${adapter_lang}→${test_lang}/${task} exists – skipping"
          continue
        fi

        CUDA_VISIBLE_DEVICES="$DEVICES" python -m src.test_cross \
          --adapter_lang "$adapter_lang" \
          --test_lang    "$test_lang" \
          --task         "$task" \
          --output_dir   "$output_dir" \
          --context      False \
          --model_name   "$model_name" \
          --model_type   "$model_type" \
          --lora_path    "$adapter_path" \
          --eval_type    "$training_scenario"
      done
    done
  done
}

# -----------------------------------------------------------------------------#
# MMLU evaluation (single task)                                                #
# -----------------------------------------------------------------------------#
test_model_mmlu() {
  local model_key=$1
  local model_name=${models[$model_key]}
  local model_type=${model_name#*/}
  local training_scenario=$2
  local batch_size=${3:-8}

  echo "============================================="
  echo "MMLU eval → model: $model_name | scenario: $training_scenario | batch: $batch_size"
  echo "============================================="

  local BASE_EVAL_DIR="/workspace/value-adapters/culturellm/eval_outputs/mmlu"
  local ADAPTERS_BASE_DIR="${OUT_DIR}/${training_scenario}"

  for adapter_lang in "${cultures[@]}"; do
    local adapter_path="${ADAPTERS_BASE_DIR}/${model_key}/${adapter_lang}"
    local output_dir="${BASE_EVAL_DIR}/${training_scenario}/${model_key}/${adapter_lang}"
    mkdir -p "$output_dir"

    local res="${output_dir}/mmlu_res.jsonl"
    local pred="${output_dir}/mmlu_predictions.jsonl"
    if [[ -f "$res" && -f "$pred" ]]; then
      echo "✔ MMLU ${adapter_lang} exists – skipping"
      continue
    fi

    CUDA_VISIBLE_DEVICES="$DEVICES" python -m src.test_mmlu \
      --adapter_lang "$adapter_lang" \
      --task         "mmlu" \
      --output_dir   "$output_dir" \
      --context      False \
      --model_name   "$model_name" \
      --model_type   "$model_type" \
      --lora_path    "$adapter_path" \
      --eval_type    "$training_scenario" \
    #   --batch_size   "$batch_size"

    [[ $? -ne 0 ]] && echo "✖ error during MMLU eval for $adapter_lang"
  done
}

# =========================== Training Phase ===================================
# echo "================== Training Phase =================="
# for model_key in "${!models[@]}"; do
#   for training_scenario in "${TRAINING_SCENARIOS[@]}"; do
#     for culture in "${cultures[@]}"; do
#       train_model "$model_key" "$training_scenario" "$culture"
#     done
#   done
# done
# echo "Training Phase Completed"

# =========================== Testing Phase ====================================
echo "================== Testing Phase ==================="
for model_key in "${!models[@]}"; do
  for training_scenario in "${TRAINING_SCENARIOS[@]}"; do
    # test_model_cross "$model_key" "$training_scenario"
    test_model_mmlu  "$model_key" "$training_scenario" 8   # change batch here if desired
  done
done
echo "Testing Phase Completed"
echo "Gemma Training and Testing Script Finished Successfully"
