#!/bin/bash
#SBATCH --job-name="inf-remain"
#SBATCH --output="/tud/outputs/run_baseline_single.out"
#SBATCH -p it-hpc
#SBATCH --gres=gpu:1

nvidia-smi

# Set the visible device
VISIBLE_DEVICE=0

# chinese
for task in 'bias_on_gender_detect' 'spam_detect'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval --language chinese --model_text llama_chinese --task $task --context False
done

# greek
for task in 'offensive_detect' 'offensive_detect_g'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval --language greek --model_text llama_greek --task $task --context False
done

# turkish
for task in 'offensive_detect' 'offensive_detect_corpus' 'offensive_detect_finegrained' 'offensive_detect_kaggle' 'offensive_detect_kaggle2' 'abusive_detect' 'spam_detect'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval --language turkish --model_text llama_turkish --task $task --context False
done
