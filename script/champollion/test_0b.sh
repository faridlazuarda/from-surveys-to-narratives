#!/bin/bash

# combined
declare -A combined_tasks
combined_tasks[germany]='hate_detect hate_off_detect hate_detect_iwg_1 hate_detect_check offensive_detect_eval'
combined_tasks[korean]='abusive_detect abusive_detect_2 abusive_detect_4 hate_detect_3 hate_detect_6 hate_detect_7'
combined_tasks[portuguese]='homophobia_detect insult_detect misogyny_detect offensive_detect_2 offensive_detect_3'
combined_tasks[spanish]='offensive_detect_ami offensive_detect_mex_a3t offensive_detect_mex_offend hate_detect_eval hate_detect_haterNet stereotype_detect mockery_detect insult_detect improper_detect aggressiveness_detect negative_stance_detect'
combined_tasks[turkish]='offensive_detect offensive_detect_corpus offensive_detect_finegrained offensive_detect_kaggle offensive_detect_kaggle2 abusive_detect spam_detect'

languages_order=('germany' 'korean' 'portuguese' 'spanish' 'turkish')

for lang in "${languages_order[@]}"; do
    echo "Processing language: $lang"
    for task in ${combined_tasks[$lang]}; do
        CUDA_VISIBLE_DEVICES=2 python -m src.test_offensEval --language $lang --model_text llama_combined --task $task --context False
    done
done
