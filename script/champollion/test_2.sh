#!/bin/bash

# Set the visible device
VISIBLE_DEVICE=2

# germany
for task in 'hate_detect' 'hate_off_detect' 'hate_detect_iwg_1' 'hate_detect_check' 'offensive_detect_eval'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval --language germany --model_text llama_germany --task $task --context False
done

# korean
for task in 'abusive_detect' 'abusive_detect_2' 'abusive_detect_4' 'hate_detect_3' 'hate_detect_6' 'hate_detect_7'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval --language korean --model_text llama_korean --task $task --context False
done

# portuguese
for task in 'homophobia_detect' 'insult_detect' 'misogyny_detect' 'offensive_detect_2' 'offensive_detect_3'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval --language portuguese --model_text llama_portuguese --task $task --context False
done

# spanish
for task in 'offensive_detect_ami' 'offensive_detect_mex_a3t' 'offensive_detect_mex_offend' 'hate_detect_eval' 'hate_detect_haterNet' 'stereotype_detect' 'mockery_detect' 'insult_detect' 'improper_detect' 'aggressiveness_detect' 'negative_stance_detect'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval --language spanish --model_text llama_spanish --task $task --context False
done

# turkish
for task in 'offensive_detect' 'offensive_detect_corpus' 'offensive_detect_finegrained' 'offensive_detect_kaggle' 'offensive_detect_kaggle2' 'abusive_detect' 'spam_detect'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval --language turkish --model_text llama_turkish --task $task --context False
done
