#!/bin/bash

#SBATCH --job-name="cross1"
#SBATCH --output="/tud/outputs/run_test_cross1.out"
#SBATCH --gres=gpu:1
#SBATCH -N 1 
#SBATCH -A nlp-dept 
#SBATCH -p nlp-dept 
#SBATCH -q nlp-pool


# Set the visible device
# VISIBLE_DEVICE=1

nvidia-smi
echo $CUDA_VISIBLE_DEVICES


# Define base output directory
BASE_OUTPUT_DIR="/tud/value-adapters/culturellm/eval_outputs/univar"

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

# Define source and target languages
# adapter_languages=( "arabic,chinese,english,german,spanish,turkish" "arabic" "chinese" "english" "german" "spanish" "turkish")
# adapter_languages=( "arabic,chinese,english,german,spanish,turkish")
adapter_languages=("arabic" "chinese" "english" "german" "spanish" "turkish")
test_languages=("arabic" "chinese" "english" "german" "spanish" "turkish")

# adapter_languages=("bengali" "greek")
# test_languages=("greek" "korean" "turkish")

# Define model name and extract model type
# MODEL_NAME='google/gemma-2-2b-it'
# MODEL_NAME='meta-llama/Meta-Llama-3.1-8B-Instruct'
# MODEL_NAME='meta-llama/Llama-2-7b-hf'
# MODEL_NAME="meta-llama/Meta-Llama-3.1-8B"
# MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"

# MODEL_TYPE="llama-2-7b-chat-hf"
# MODEL_TYPE="meta-llama-3.1-8b"
# MODEL_TYPE="gemma-2-2b-it"
# MODEL_TYPE="llama-3.1-8B-it"
# MODEL_TYPE="llama-2-7b-hf"

models=("google/gemma-2-2b-it")


# Evaluate each language adapter on all other languages' tasks
for model in "${models[@]}"; do
    MODEL_NAME=$model
    MODEL_TYPE="${MODEL_NAME#*/}"
    for adapter_lang in "${adapter_languages[@]}"; do
        for test_lang in "${test_languages[@]}"; do
            if [ "$adapter_lang" == "$test_lang" ]; then
                for task in ${combined_tasks[$test_lang]}; do
                    # Define output directory for storing evaluation results
                    OUTPUT_DIR="$BASE_OUTPUT_DIR"
                    mkdir -p "$OUTPUT_DIR"
                    # Run the evaluation and save the results
                    echo $adapter_lang
                    python -m src.test_cross \
                    --adapter_lang "$adapter_lang" \
                    --test_lang $test_lang \
                    --task $task \
                    --experiment_type "univar" \
                    --output_dir $OUTPUT_DIR \
                    --context False \
                    --model_name $MODEL_NAME \
                    --model_type $MODEL_TYPE \
                    --eval_type single \
                    # --debug
                done
            fi
        done
    done
done 