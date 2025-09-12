#!/bin/bash

# Set the visible device
VISIBLE_DEVICE=2

# arabic
for task in 'offensive_detect' 'offensive_detect_osact4' 'offensive_detect_mp' 'offensive_detect_osact5' 'hate_detect_osact4' 'hate_detect_mp' 'hate_detect_osact5' 'vulgar_detect_mp' 'spam_detect' 'hate_detect_fine-grained'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval --language arabic --model_text llama_Jordan_Iraq --task $task --context False
done

# bengali
for task in 'hate_detect_religion' 'offensive_detect_1' 'offensive_detect_2' 'offensive_detect_3' 'racism_detect' 'threat_detect'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval --language bengali --model_text llama_bengali --task $task --context False
done

# chinese
for task in 'bias_on_gender_detect' 'spam_detect'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval --language chinese --model_text llama_china --task $task --context False
done

# greek
for task in 'offensive_detect' 'offensive_detect_g'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval --language greek --model_text llama_china --task $task --context False
done

# english
for task in 'hate_detect_2' 'hate_offens_detect' 'hostility_directness_detect' 'offensive_detect_easy' 'threat_detect' 'toxicity_detect'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval --language english --model_text llama_english --task $task --context False
done
