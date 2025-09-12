
WANDB_API=""
OUT_DIR="valadapt/outputs"  # set customised out_path
MODEL_NAME="meta-llama/Llama-2-7b-hf" # if OOM, use --load_in_8bit
MODEL_TYPE="${MODEL_NAME#*/}"
DATA_PATH="/tud/culturellm/data"


# List of cultures
cultures=("arabic,bengali,china,english,germany,greece,korean,portuguese,spanish,turkey" "arabic" "bengali" "china" "english" "germany" "greece" "korean" "portuguese" "spanish" "turkey")

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

    CUDA_VISIBLE_DEVICES=2,3 python -m src.llama_finetune \
    --model_name ${MODEL_NAME} \
    --culture ${culture} \
    --wandb_api_key ${WANDB_API} \
    --wandb_name valadapt_${MODEL_TYPE}_${culture} \
    --out_path ${OUT_DIR}/baseline/${MODEL_TYPE}/${culture} \
    --data_files $llama_files \
    --override_results 




done