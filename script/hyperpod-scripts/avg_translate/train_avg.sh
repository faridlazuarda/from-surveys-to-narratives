#!/bin/bash
#SBATCH --job-name="train-l3"
#SBATCH --output="/tud/outputs/run_train_gemma9B.out"
#SBATCH --gres=gpu:1
#SBATCH -N 1 
#SBATCH -A nlp-dept 
#SBATCH -p nlp-dept 
#SBATCH -q nlp-pool

echo $CUDA_VISIBLE_DEVICES

# Set the visible device
WANDB_API_KEY=""
OUT_DIR=""
DATA_PATH=""

# models=("meta-llama/Meta-Llama-3.1-8B" "meta-llama/Meta-Llama-3.1-8B-Instruct")
models=("meta-llama/Meta-Llama-3.1-8B-Instruct")

nvidia-smi

for model in "${models[@]}"; do
    MODEL_NAME=$model
    MODEL_TYPE="${MODEL_NAME#*/}"

    echo "Using model: $MODEL_NAME"
    echo "Using data file: $DATA_PATH"

    # Run the Python script directly with the specified file
    python -m src.llama_finetune \
    --model_name ${MODEL_NAME} \
    --culture "averaged" \
    --wandb_api_key ${WANDB_API_KEY} \
    --wandb_name valadapt_averaged_${MODEL_TYPE}_averaged \
    --out_path ${OUT_DIR}/averaged/${MODEL_TYPE}/averaged \
    --data_files $DATA_PATH \
    --wandb_offline 
done
