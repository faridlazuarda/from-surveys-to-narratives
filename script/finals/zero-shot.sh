#!/bin/bash

#SBATCH --job-name="cross1"
#SBATCH --output="/tud/outputs/run_test_cross1.out"
#SBATCH --gres=gpu:1
#SBATCH -N 1
#SBATCH -A nlp-dept
#SBATCH -p nlp-dept
#SBATCH -q nlp-pool

# ------------------------- Environment Setup -------------------------
echo "Starting job script..."
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
nvidia-smi

# ------------------------- Directory and Array Definitions -----------

# If you have a separate directory for adapters (e.g., /home/user/adapters)
# be sure to define it:
ADAPTERS_DIR="tud/models/normal"

# Base output directory
BASE_OUTPUT_DIR="tud/value-adapters/culturellm/eval_outputs/zero-shot"

# Example tasks per language
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

# Define (adapter) languages and test languages
adapter_languages=("arabic" "bengali" "chinese" "english" "german" "greek" "korean" "portuguese" "spanish" "turkish")
test_languages=("arabic" "bengali" "chinese" "english" "german" "greek" "korean" "portuguese" "spanish" "turkish")

# Models to evaluate
models=("google/gemma-2-9b-it" "Qwen/Qwen2.5-7B-Instruct")
EVAL_TYPE="zero-shot"

# ------------------------- Main Loop --------------------------------

for model in "${models[@]}"; do
    MODEL_NAME=$model
    # MODEL_TYPE will be the part after the slash (e.g., "gemma-2-9b-it")
    MODEL_TYPE="${MODEL_NAME#*/}"

    # Loop over all adapters you want to test
    for adapter_lang in "${adapter_languages[@]}"; do
        
        echo "============================"
        echo "Model: $MODEL_NAME"
        echo "Adapter Language: $adapter_lang"
        echo "============================"

        # Now evaluate on each test language (and tasks)
        for test_lang in "${test_languages[@]}"; do

            echo "Test Language: $test_lang"
            
            for task in ${combined_tasks[$test_lang]}; do

                # Path to the LoRA adapter (adapter_lang)
                ADAPTER_PATH="${ADAPTERS_DIR}/${MODEL_TYPE}/${adapter_lang}"

                # Define the directory for storing evaluation results
                # You can nest them by model_type/adapter_lang/test_lang if you prefer
                OUTPUT_DIR="${BASE_OUTPUT_DIR}/${MODEL_TYPE}/${test_lang}"

                # Ensure the output directory exists
                mkdir -p "$OUTPUT_DIR"

                # Construct the filenames for your results/predictions
                # For example, _res.jsonl and _predictions.jsonl
                result_file="${OUTPUT_DIR}/${task}_res.jsonl"
                prediction_file="${OUTPUT_DIR}/${task}_predictions.jsonl"

                # --------------- Check if results exist ---------------
                if [[ -f "$result_file" && -f "$prediction_file" ]]; then
                    echo "Results already exist for task [$task] (Adapter: $adapter_lang, Test: $test_lang). Skipping..."
                    continue
                fi

                # --------------- Run evaluation script ---------------
                echo "Running evaluation for Model: $MODEL_NAME, Adapter: $adapter_lang, Test Culture: $test_lang, Task: $task."

                CUDA_VISIBLE_DEVICES=0 python -m src.test_cross \
                    --adapter_lang "$adapter_lang" \
                    --test_lang "$test_lang" \
                    --task "$task" \
                    --output_dir "$OUTPUT_DIR" \
                    --context False \
                    --model_name "$MODEL_NAME" \
                    --model_type "$MODEL_TYPE" \
                    --lora_path "$ADAPTER_PATH" \
                    --eval_type "$EVAL_TYPE"

                echo "Completed evaluation for $task on $test_lang with adapter $adapter_lang."
                echo "--------------------------------------------------"

            done  # end of task loop
        done      # end of test_lang loop
    done          # end of adapter_lang loop
done              # end of models loop

echo "All evaluations completed."
