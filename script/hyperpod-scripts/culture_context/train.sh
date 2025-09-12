#!/bin/bash
#SBATCH --job-name="train-l31"
#SBATCH --output="/tud/outputs/run_train_llama31.out"
#SBATCH -p it-hpc
#SBATCH --gres=gpu:2
echo $CUDA_VISIBLE_DEVICES

# Set the visible device
# VISIBLE_DEVICE=0,2
WANDB_API_KEY=""
OUT_DIR=""

MODEL_NAME="meta-llama/Meta-Llama-3.1-8B"
# MODEL_NAME="meta-llama/Meta-Llama-3.1-8B-Instruct"
# MODEL_NAME="google/gemma-2-9b-it"
MODEL_TYPE="${MODEL_NAME#*/}"
DATA_PATH=""

TRAINING_SCENARIO="normad_context"

# List of cultures
# cultures=("arabic" "bengali" "chinese" "english" "german" "greek" "korean" "portuguese" "spanish" "turkish" "arabic,bengali,chinese,english,german,greek,korean,portuguese,spanish,turkish" )
cultures=("arabic,bengali,chinese,english,german,greek,korean,portuguese,spanish,turkish")

nvidia-smi

# Loop through each culture
for culture in "${cultures[@]}"; do
    if [[ "$culture" == *","* ]]; then
        # Handle combined culture case
        IFS=',' read -ra CULTURE_ARRAY <<< "$culture"
        llama_files=""
        for sub_culture in "${CULTURE_ARRAY[@]}"; do
            llama_file=$(find ${DATA_PATH}/${sub_culture}/Finetune/ -name "*llama*" | head -n 1)
            if [ -n "$llama_file" ]; then
                llama_files="${llama_files}${llama_file},"
            else
                echo "No files containing 'llama' found for culture: ${sub_culture}"
            fi
        done
        # Remove trailing comma
        llama_files=${llama_files%,}
    else
        # Handle single culture case
        llama_file=$(find ${DATA_PATH}/${culture}/Finetune/ -name "*llama*" | head -n 1)
        if [ -n "$llama_file" ]; then
            llama_files=$llama_file
        else
            echo "No files containing 'llama' found for culture: ${culture}"
            continue
        fi
    fi

    # CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE 
    python -m src.llama_finetune \
    --model_name ${MODEL_NAME} \
    --culture ${culture} \
    --wandb_api_key ${WANDB_API_KEY} \
    --wandb_name valadapt_${TRAINING_SCENARIO}_${MODEL_TYPE}_${culture} \
    --out_path ${OUT_DIR}/${TRAINING_SCENARIO}/${MODEL_TYPE}/${culture} \
    --data_files $llama_files \
    --override_results \
    --training_scenario "${TRAINING_SCENARIO}"

done
