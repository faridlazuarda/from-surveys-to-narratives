#!/bin/bash
#SBATCH --job-name="try-combined"
#SBATCH --output="/tud/outputs/run_try_combined.out"
#SBATCH -p it-hpc
#SBATCH --gres=gpu:1

nvidia-smi

# Set the visible device
VISIBLE_DEVICE=0
MODEL_NAME='meta-llama/Llama-2-7b-hf'
OUT_DIR="/tud/value-adapters/culturellm/eval_outputs"

# combined
declare -A combined_tasks
combined_tasks[arabic]='offensive_detect offensive_detect_osact4 offensive_detect_mp offensive_detect_osact5 hate_detect_osact4 hate_detect_mp hate_detect_osact5 vulgar_detect_mp spam_detect hate_detect_fine-grained'
combined_tasks[bengali]='hate_detect_religion offensive_detect_1 offensive_detect_2 offensive_detect_3 racism_detect threat_detect'
combined_tasks[chinese]='bias_on_gender_detect spam_detect'
combined_tasks[greek]='offensive_detect offensive_detect_g'
combined_tasks[english]='hate_detect_2 hate_offens_detect hostility_directness_detect offensive_detect_easy threat_detect toxicity_detect'

languages_order=('arabic' 'bengali' 'chinese' 'greek' 'english' 'german' 'korean' 'portuguese' 'spanish' 'turkish')

for lang in "${languages_order[@]}"; do
    echo "Processing language: $lang"
    for task in ${combined_tasks[$lang]}; do
        CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval \
        --language $lang \
        --task $task \
        --output_dir $OUT_DIR \
        --context False \
        --model_name $MODEL_NAME \
        --eval_type combined\
        --debug
    done
done


