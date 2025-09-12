#!/bin/bash
#SBATCH --job-name="l31-mu"
#SBATCH --output="/tud/outputs/run_train_llama31_madeup.out"
#SBATCH --gres=gpu:2
#SBATCH -N 1 
#SBATCH -A nlp-dept 
#SBATCH -p nlp-dept 
#SBATCH -q nlp-pool
#SBATCH --mem=80G            # memory per node in GB 

# Set the visible device
# VISIBLE_DEVICE=2,3
WANDB_API_KEY=""
OUT_DIR="models"

# List of models
# models=("meta-llama/Meta-Llama-3.1-8B-Instruct" "meta-llama/Meta-Llama-3.1-8B")
models=("meta-llama/Meta-Llama-3.1-8B")


DATA_PATH="data"
# List of cultures
# cultures=("arabic,chinese,english,german,spanish,turkish" "arabic" "chinese" "english" "german" "spanish" "turkish")
cultures=("bengali" "chinese")

# cultures=("madeup")
# cultures=("arabic,bengali,chinese,english,german,greek,korean,portuguese,spanish,turkish" "arabic" "bengali" "chinese" "english" "german" "greek" "korean" "portuguese" "spanish" "turkish")


nvidia-smi

# Loop through each model
for MODEL_NAME in "${models[@]}"; do
    # Extract the model type from the model name
    MODEL_TYPE="${MODEL_NAME#*/}"

    # Loop through each culture
    for culture in "${cultures[@]}"; do
        # if [[ "$culture" == *","* ]]; then
        #     # Handle combined culture case
        #     IFS=',' read -ra CULTURE_ARRAY <<< "$culture"
        #     llama_files=""
        #     for sub_culture in "${CULTURE_ARRAY[@]}"; do
        #         llama_file=$(find ${DATA_PATH}/${sub_culture}/Finetune/ -name "*llama*" | head -n 1)
        #         if [ -n "$llama_file" ]; then
        #             llama_files="${llama_files}${llama_file},"
        #         else
        #             echo "No files containing 'llama' found for culture: ${sub_culture}"
        #         fi
        #     done
        #     # Remove trailing comma
        #     llama_files=${llama_files%,}
        # else
        #     # Handle single culture case
        #     llama_file=$(find ${DATA_PATH}/${culture}/Finetune/ -name "*llama*" | head -n 1)
        #     if [ -n "$llama_file" ]; then
        #         echo $llama_file
        #         llama_files=$llama_file
        #     else
        #         echo "No files containing 'llama' found for culture: ${culture}"
        #         continue
        #     fi
        # fi

        # Run the finetuning script for the current model and culture
        CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.llama_finetune \
        --model_name ${MODEL_NAME} \
        --culture ${culture} \
        --wandb_api_key ${WANDB_API_KEY} \
        --wandb_name valadapt_${MODEL_TYPE}_${culture} \
        --out_path ${OUT_DIR}/baseline/${MODEL_TYPE}/${culture} \
        --data_files $llama_files \
        --override_results

    done
done



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
adapter_languages=("bengali" "chinese")
test_languages=("arabic" "bengali" "chinese" "english" "german" "greek" "korean" "portuguese" "spanish")


# adapter_languages=("turkish")
# test_languages=("turkish")

models=("meta-llama/Meta-Llama-3.1-8B")
# models=("meta-llama/Meta-Llama-3.1-8B-Instruct")

scenarios=("baseline" "cultural_context")

for scenario in "${scenarios[@]}"; do
    # Define base output directory
    BASE_OUTPUT_DIR="/tud/value-adapters/culturellm/eval_outputs/proba/${scenario}"

    ADAPTERS_DIR="/tud/models/${scenario}"
    # Evaluate each language adapter on all other languages' tasks
    for model in "${models[@]}"; do
        MODEL_NAME=$model
        MODEL_TYPE="${MODEL_NAME#*/}"
        for adapter_lang in "${adapter_languages[@]}"; do
            for test_lang in "${test_languages[@]}"; do
                for task in ${combined_tasks[$test_lang]}; do
                    ADAPTER_PATH="${ADAPTERS_DIR}/${MODEL_TYPE}/${adapter_lang}"

                    OUTPUT_DIR="${BASE_OUTPUT_DIR}"
                    
                    mkdir -p "$OUTPUT_DIR"

                    echo "Running evaluation for Model: $MODEL_NAME, Adapter: $adapter_lang, Test Culture: $test_lang."

                    # Define output directory for storing evaluation results
                    OUTPUT_DIR="$BASE_OUTPUT_DIR"
                    mkdir -p "$OUTPUT_DIR"
                    # Run the evaluation and save the results
                    python -m src.test_cross_prob \
                    --adapter_lang $adapter_lang \
                    --test_lang $test_lang \
                    --task $task \
                    --output_dir $OUTPUT_DIR \
                    --context False \
                    --model_name $MODEL_NAME \
                    --model_type $MODEL_TYPE \
                    --lora_path "$ADAPTER_PATH" \
                    --eval_type "$scenario" \
                    # --debug
                done
            done
        done
    done
done
