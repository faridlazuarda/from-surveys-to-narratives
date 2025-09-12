#!/bin/bash

# combined
declare -A combined_tasks
# combined_tasks[arabic]='offensive_detect offensive_detect_osact4 offensive_detect_mp offensive_detect_osact5 hate_detect_osact4 hate_detect_mp hate_detect_osact5 vulgar_detect_mp spam_detect hate_detect_fine-grained'
combined_tasks[arabic]='vulgar_detect_mp spam_detect hate_detect_fine-grained'
combined_tasks[bengali]='hate_detect_religion offensive_detect_1 offensive_detect_2 offensive_detect_3 racism_detect threat_detect'
combined_tasks[chinese]='bias_on_gender_detect spam_detect'
combined_tasks[greek]='offensive_detect offensive_detect_g'
combined_tasks[english]='hate_detect_2 hate_offens_detect hostility_directness_detect offensive_detect_easy threat_detect toxicity_detect'

languages_order=('arabic' 'bengali' 'chinese' 'greek' 'english')

for lang in "${languages_order[@]}"; do
    echo "Processing language: $lang"
    for task in ${combined_tasks[$lang]}; do
        CUDA_VISIBLE_DEVICES=2 python -m src.test_offensEval --language $lang --model_text llama_combined --task $task --context False
    done
done
