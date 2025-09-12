#!/bin/bash

# combined
declare -A combined_tasks
combined_tasks[arabic]='offensive_detect offensive_detect_osact4 offensive_detect_mp offensive_detect_osact5 hate_detect_osact4 hate_detect_mp hate_detect_osact5 vulgar_detect_mp spam_detect hate_detect_fine-grained'
combined_tasks[bengali]='hate_detect_religion offensive_detect_1 offensive_detect_2 offensive_detect_3 racism_detect threat_detect'
combined_tasks[chinese]='bias_on_gender_detect spam_detect'
combined_tasks[greek]='offensive_detect offensive_detect_g'
combined_tasks[english]='hate_detect_2 hate_offens_detect hostility_directness_detect offensive_detect_easy threat_detect toxicity_detect'
combined_tasks[germany]='hate_detect hate_off_detect hate_detect_iwg_1 hate_detect_check offensive_detect_eval'
combined_tasks[korean]='abusive_detect abusive_detect_2 abusive_detect_4 hate_detect_3 hate_detect_6 hate_detect_7'
combined_tasks[portuguese]='homophobia_detect insult_detect misogyny_detect offensive_detect_2 offensive_detect_3'
combined_tasks[spanish]='offensive_detect_ami offensive_detect_mex_a3t offensive_detect_mex_offend hate_detect_eval hate_detect_haterNet stereotype_detect mockery_detect insult_detect improper_detect aggressiveness_detect negative_stance_detect'
combined_tasks[turkish]='offensive_detect offensive_detect_corpus offensive_detect_finegrained offensive_detect_kaggle offensive_detect_kaggle2 abusive_detect spam_detect'

languages_order=('arabic' 'bengali' 'chinese' 'greek' 'english' 'germany' 'korean' 'portuguese' 'spanish' 'turkish')

for lang in "${languages_order[@]}"; do
    echo "Processing language: $lang"
    for task in ${combined_tasks[$lang]}; do
        CUDA_VISIBLE_DEVICES=2,3 python test_offensEval.py --language $lang --model_text llama_combined --task $task --context False
    done
done


# arabic
for task in 'offensive_detect' 'offensive_detect_osact4' 'offensive_detect_mp' 'offensive_detect_osact5' 'hate_detect_osact4' 'hate_detect_mp' 'hate_detect_osact5' 'vulgar_detect_mp' 'spam_detect' 'hate_detect_fine-grained'
do 
    CUDA_VISIBLE_DEVICES=2,3 python test_offensEval.py --language arabic --model_text llama_Jordan_Iraq --task $task --context False
done

# bengali
for task in 'hate_detect_religion' 'offensive_detect_1' 'offensive_detect_2' 'offensive_detect_3' 'racism_detect' 'threat_detect'
do 
    CUDA_VISIBLE_DEVICES=2,3 python test_offensEval.py --language bengali --model_text llama_bengali --task $task --context False
done

# chinese
for task in 'bias_on_gender_detect' 'spam_detect'
do 
    CUDA_VISIBLE_DEVICES=2,3 python test_offensEval.py --language chinese --model_text llama_china --task $task --context False
done

# greek
for task in 'offensive_detect' 'offensive_detect_g'
do 
    CUDA_VISIBLE_DEVICES=2,3 python test_offensEval.py --language greek --model_text llama_china --task $task --context False
done

# english
for task in 'hate_detect_2' 'hate_offens_detect' 'hostility_directness_detect' 'offensive_detect_easy' 'threat_detect' 'toxicity_detect'
do 
    CUDA_VISIBLE_DEVICES=2,3 python test_offensEval.py --language english --model_text llama_english --task $task --context False
done

# germany
for task in 'hate_detect' 'hate_off_detect' 'hate_detect_iwg_1' 'hate_detect_check' 'offensive_detect_eval'
do 
    CUDA_VISIBLE_DEVICES=2,3 python test_offensEval.py --language germany --model_text llama_germanyy --task $task --context False
done

# korean
for task in 'abusive_detect' 'abusive_detect_2' 'abusive_detect_4' 'hate_detect_3' 'hate_detect_6' 'hate_detect_7'
do 
    CUDA_VISIBLE_DEVICES=2,3 python test_offensEval.py --language korean --model_text llama_korean --task $task --context False
done

# portuguese
for task in 'homophobia_detect' 'insult_detect' 'misogyny_detect' 'offensive_detect_2' 'offensive_detect_3'
do 
    CUDA_VISIBLE_DEVICES=2,3 python test_offensEval.py --language portuguese --model_text llama_portuguese --task $task --context False
done

# spanish
for task in 'offensive_detect_ami' 'offensive_detect_mex_a3t' 'offensive_detect_mex_offend' 'hate_detect_eval' 'hate_detect_haterNet' 'stereotype_detect' 'mockery_detect' 'insult_detect' 'improper_detect' 'aggressiveness_detect' 'negative_stance_detect'
do 
    CUDA_VISIBLE_DEVICES=2,3 python test_offensEval.py --language spanish --model_text llama_spanish --task $task --context False
done

# turkish
for task in 'offensive_detect' 'offensive_detect_corpus' 'offensive_detect_finegrained' 'offensive_detect_kaggle' 'offensive_detect_kaggle2' 'abusive_detect' 'spam_detect'
do 
    CUDA_VISIBLE_DEVICES=2,3 python test_offensEval.py --language turkish --model_text llama_turkish --task $task --context False
done

