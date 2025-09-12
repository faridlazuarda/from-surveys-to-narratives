#!/bin/bash
#SBATCH --job-name="train-l3"
#SBATCH --output="/tud/outputs/run_train_llama3_valadapt.out"
#SBATCH -p it-hpc
#SBATCH --gres=gpu:2

# Set the visible device
VISIBLE_DEVICE=3,4
WANDB_API_KEY=""
OUT_DIR="models"
DATA_PATH="train"

# List of cultures
cultures=("arabic" "chinese" "english" "german" "spanish" "turkish" "arabic,chinese,english,german,spanish,turkish")

# List of models
# models=("meta-llama/Meta-Llama-3.1-8B-Instruct" "meta-llama/Meta-Llama-3.1-8B" "google/gemma-2-9b" "google/gemma-2-9b-it" "google/gemma-2-2b" "google/gemma-2-2b-it")
# models=("google/gemma-2-2b" "meta-llama/Meta-Llama-3.1-8B" "google/gemma-2-9b")
models=("meta-llama/Meta-Llama-3.1-8B" "meta-llama/Meta-Llama-3.1-8B-Instruct")

nvidia-smi

# Flags to control skipping
SKIP_COMPLETED=false
# Outer loop over models
for model in "${models[@]}"; do
    MODEL_NAME=$model
    MODEL_TYPE="${MODEL_NAME#*/}"

    # Loop through each culture
    for culture in "${cultures[@]}"; do
        # Check if we have reached the last incomplete experiment
        if [[ "$MODEL_TYPE" == "meta-llama/Meta-Llama-3.1-8B" && "$culture" == "arabic" ]]; then
            echo "Resuming from incomplete experiment: ${MODEL_TYPE} and culture ${culture}"
            SKIP_COMPLETED=false
        fi

        # If we are still skipping, continue
        if [[ "$SKIP_COMPLETED" == true ]]; then
            echo "Skipping ${MODEL_TYPE} with culture ${culture}"
            continue
        fi


        if [[ "$culture" == *","* ]]; then
            # Handle combined culture case
            IFS=',' read -ra CULTURE_ARRAY <<< "$culture"
            llama_files=""
            for sub_culture in "${CULTURE_ARRAY[@]}"; do
                llama_file="${DATA_PATH}/univar_${sub_culture}.jsonl"
                if [ -f "$llama_file" ]; then
                    llama_files="${llama_files}${llama_file},"
                else
                    echo "No file found for culture: ${sub_culture}"
                fi
            done
            # Remove trailing comma
            llama_files=${llama_files%,}
        else
            # Handle single culture case
            llama_file="${DATA_PATH}/univar_${culture}.jsonl"
            if [ -f "$llama_file" ]; then
                llama_files=$llama_file
            else
                echo "No file found for culture: ${culture}"
                continue
            fi
        fi

        CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.llama_finetune \
        --model_name ${MODEL_NAME} \
        --culture ${culture} \
        --wandb_api_key ${WANDB_API_KEY} \
        --wandb_name valadapt_${MODEL_TYPE}_${culture} \
        --out_path ${OUT_DIR}/univar/${MODEL_TYPE}/${culture} \
        --data_files $llama_files \
        --override_results \
        # --debug \
        # --wandb_offline \

    done
done
