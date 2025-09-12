#!/bin/bash
#SBATCH --job-name="train-l2"
#SBATCH --output="/tud/outputs/run_train_llama2.out"
#SBATCH -p it-hpc
#SBATCH --gres=gpu:2
echo $CUDA_VISIBLE_DEVICES

# Set the visible device
# VISIBLE_DEVICE=0,2
WANDB_API_KEY="f8885d645b26e9b2b2fac972e456d72947c5e436"
OUT_DIR="/tud/models"

MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"
MODEL_TYPE="${MODEL_NAME#*/}"
DATA_PATH="/tud/value-adapters/culturellm/data"

# List of cultures
cultures=("arabic,bengali,chinese,english,german,greek,korean,portuguese,spanish,turkish" "arabic" "bengali" "chinese" "english" "german" "greek" "korean" "portuguese" "spanish" "turkish")


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
    --wandb_name valadapt_${MODEL_TYPE}_${culture} \
    --out_path ${OUT_DIR}/baseline/${MODEL_TYPE}/${culture} \
    --data_files $llama_files \
    --override_results 

done
