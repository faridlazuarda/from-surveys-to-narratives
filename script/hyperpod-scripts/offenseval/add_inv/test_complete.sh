#!/bin/bash

#SBATCH --job-name="cross1"
#SBATCH --output="/tud/outputs/run_test_cross1.out"
#SBATCH --gres=gpu:1
#SBATCH -N 1 
#SBATCH -A nlp-dept 
#SBATCH -p nlp-dept 
#SBATCH -q nlp-pool

# Set the visible device
VISIBLE_DEVICE=0

nvidia-smi
echo $CUDA_VISIBLE_DEVICES

# Define tasks for each language
declare -A combined_tasks=(
    [arabic]='vulgar_detect_mp spam_detect hate_detect_fine-grained'
    [bengali]='hate_detect_religion offensive_detect_1 offensive_detect_2 offensive_detect_3 racism_detect threat_detect'
    [chinese]='bias_on_gender_detect spam_detect'
    [greek]='offensive_detect offensive_detect_g'
    [english]='hate_detect_2 hate_offens_detect hostility_directness_detect offensive_detect_easy threat_detect toxicity_detect'
    [german]='hate_detect hate_off_detect hate_detect_iwg_1 hate_detect_check offensive_detect_eval'
    [korean]='abusive_detect abusive_detect_2 abusive_detect_4 hate_detect_3 hate_detect_6 hate_detect_7'
    [portuguese]='homophobia_detect insult_detect misogyny_detect offensive_detect_2 offensive_detect_3'
    [spanish]='offensive_detect_ami offensive_detect_mex_a3t offensive_detect_mex_offend hate_detect_eval hate_detect_haterNet stereotype_detect mockery_detect insult_detect improper_detect aggressiveness_detect negative_stance_detect'
    [turkish]='offensive_detect offensive_detect_corpus offensive_detect_finegrained offensive_detect_kaggle offensive_detect_kaggle2 abusive_detect spam_detect'
)

# Define evaluation configurations
adapter_languages=("arabic" "bengali" "chinese" "english" "german" "greek" "korean" "portuguese" "spanish" "turkish" "arabic,bengali,chinese,english,german,greek,korean,portuguese,spanish,turkish")
test_languages=("arabic" "bengali" "chinese" "english" "german" "greek" "korean" "portuguese" "spanish" "turkish")
# models=("meta-llama/Meta-Llama-3.1-8B")
models=("meta-llama/Meta-Llama-3.1-8B-Instruct")
eval_types=("zero-shot" "baseline" "cultural_context")

# Base directories
BASE_MODEL_DIR="/tud/models"
BASE_OUTPUT_DIR="/tud/value-adapters/culturellm/eval_outputs/inv_complete"

PLACEHOLDER_ADAPTER_LANG="none"
# Main evaluation loop
for eval_type in "${eval_types[@]}"; do
    echo "Starting evaluations for eval_type: $eval_type"
    
    for model in "${models[@]}"; do
        MODEL_NAME=$model
        MODEL_TYPE="${MODEL_NAME#*/}"

        OUTPUT_DIR="${BASE_OUTPUT_DIR}/${eval_type}/"
        mkdir -p "$OUTPUT_DIR"

        if [[ "$eval_type" == "zero-shot" ]]; then
            # Zero-shot evaluation (no adapters)
            for test_lang in "${test_languages[@]}"; do
                for task in ${combined_tasks[$test_lang]}; do
                    echo "Zero-shot: Model=$MODEL_NAME | Test=$test_lang | Task=$task"

                    python -m src.test_cross_inv \
                        --adapter_lang "$PLACEHOLDER_ADAPTER_LANG" \
                        --test_lang "$test_lang" \
                        --task "$task" \
                        --output_dir "$OUTPUT_DIR" \
                        --context False \
                        --model_name "$MODEL_NAME" \
                        --model_type "$MODEL_TYPE" \
                        --eval_type "$eval_type"
                done
            done
        else
            # Finetuning scenarios (baseline and cultural_context)
            for adapter_lang in "${adapter_languages[@]}"; do
                for test_lang in "${test_languages[@]}"; do
                    for task in ${combined_tasks[$test_lang]}; do
                        ADAPTER_PATH="${BASE_MODEL_DIR}/${eval_type}/${MODEL_TYPE}/${adapter_lang}"

                        # Check if adapter exists
                        if [ ! -d "$ADAPTER_PATH" ]; then
                            echo "WARNING: Adapter path $ADAPTER_PATH does not exist. Skipping."
                            continue
                        fi

                        echo "$eval_type: Model=$MODEL_NAME | Adapter=$adapter_lang | Test=$test_lang | Task=$task"

                        python -m src.test_cross_inv \
                            --adapter_lang "$adapter_lang" \
                            --test_lang "$test_lang" \
                            --task "$task" \
                            --output_dir "$OUTPUT_DIR" \
                            --context False \
                            --model_name "$MODEL_NAME" \
                            --model_type "$MODEL_TYPE" \
                            --lora_path "$ADAPTER_PATH" \
                            --eval_type "$eval_type"
                    done
                done
            done
        fi
    done
done

echo "All evaluations completed."



