#!/bin/bash
#SBATCH --job-name="try-single-cross"
#SBATCH --output="/tud/outputs/run_try_single_cross.out"
#SBATCH -p it-hpc
#SBATCH --gres=gpu:1

nvidia-smi

# Set the visible device
VISIBLE_DEVICE=0
MODEL_NAME='meta-llama/Llama-2-7b-hf'
OUT_DIR="/tud/value-adapters/culturellm/eval_outputs"

# english
for task in 'hate_detect_2' 'hate_offens_detect' 'hostility_directness_detect' 'offensive_detect_easy' 'threat_detect' 'toxicity_detect'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval --language english --task $task --output_dir $OUT_DIR --context False --model_name $MODEL_NAME --eval_type single --debug
done

# arabic
for task in 'offensive_detect' 'offensive_detect_osact4' 'offensive_detect_mp' 'offensive_detect_osact5' 'hate_detect_osact4' 'hate_detect_mp' 'hate_detect_osact5' 'vulgar_detect_mp' 'spam_detect' 'hate_detect_fine-grained'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval --language arabic --task $task --output_dir $OUT_DIR --context False --model_name $MODEL_NAME --eval_type single --debug
done

# bengali
for task in 'hate_detect_religion' 'offensive_detect_1' 'offensive_detect_2' 'offensive_detect_3' 'racism_detect' 'threat_detect'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval --language bengali --task $task --output_dir $OUT_DIR --context False --model_name $MODEL_NAME --eval_type single --debug
done

# chinese
for task in 'bias_on_gender_detect' 'spam_detect'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval --language chinese --task $task --output_dir $OUT_DIR --context False --model_name $MODEL_NAME --eval_type single --debug
done

# greek
for task in 'offensive_detect' 'offensive_detect_g'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval --language greek --task $task --output_dir $OUT_DIR --context False --model_name $MODEL_NAME --eval_type single --debug
done

# german
for task in 'hate_detect' 'hate_off_detect' 'hate_detect_iwg_1' 'hate_detect_check' 'offensive_detect_eval'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval --language german --task $task --output_dir $OUT_DIR --context False --model_name $MODEL_NAME --eval_type single --debug
done

# korean
for task in 'abusive_detect' 'abusive_detect_2' 'abusive_detect_4' 'hate_detect_3' 'hate_detect_6' 'hate_detect_7'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval --language korean --task $task --output_dir $OUT_DIR --context False --model_name $MODEL_NAME --eval_type single --debug
done

# portuguese
for task in 'homophobia_detect' 'insult_detect' 'misogyny_detect' 'offensive_detect_2' 'offensive_detect_3'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval --language portuguese --task $task --output_dir $OUT_DIR --context False --model_name $MODEL_NAME --eval_type single --debug
done

# spanish
for task in 'offensive_detect_ami' 'offensive_detect_mex_a3t' 'offensive_detect_mex_offend' 'hate_detect_eval' 'hate_detect_haterNet' 'stereotype_detect' 'mockery_detect' 'insult_detect' 'improper_detect' 'aggressiveness_detect' 'negative_stance_detect'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval --language spanish --task $task --output_dir $OUT_DIR --context False --model_name $MODEL_NAME --eval_type single --debug
done

# turkish
for task in 'offensive_detect' 'offensive_detect_corpus' 'offensive_detect_finegrained' 'offensive_detect_kaggle' 'offensive_detect_kaggle2' 'abusive_detect' 'spam_detect'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval --language turkish --task $task --output_dir $OUT_DIR --context False --model_name $MODEL_NAME --eval_type single --debug
done



declare -A combined_tasks
combined_tasks[arabic]='vulgar_detect_mp spam_detect hate_detect_fine-grained'
combined_tasks[bengali]='hate_detect_religion offensive_detect_1 offensive_detect_2 offensive_detect_3 racism_detect threat_detect'
combined_tasks[chinese]='bias_on_gender_detect spam_detect'
combined_tasks[greek]='offensive_detect offensive_detect_g'
combined_tasks[english]='hate_detect_2 hate_offens_detect hostility_directness_detect offensive_detect_easy threat_detect toxicity_detect'
combined_tasks[german]='hate_detect hate_off_detect hate_detect_iwg_1 hate_detect_check offensive_detect_eval'
combined_tasks[korean]='abusive_detect abusive_detect_2 abusive_detect_4 hate_detect_3 hate_detect_6 hate_detect_7'
combined_tasks[portuguese]='homophobia_detect insult_detect misogyny_detect offensive_detect_2 offensive_detect_3'
combined_tasks[spanish]='offensive_detect_ami offensive_detect_mex_a3t offensive_detect_mex_offend hate_detect_eval hate_detect_haterNet stereotype_detect mockery_detect insult_detect improper_detect aggressiveness_detect negative_stance_detect'
combined_tasks[turkish]='offensive_detect offensive_detect_corpus offensive_detect_finegrained offensive_detect_kaggle offensive_detect_kaggle2 abusive_detect spam_detect'

# Define source and target languages
src_languages=("arabic" "bengali" "chinese" "english" "german" "greek" "korean" "portuguese" "spanish" "turkish")
tgt_languages=("arabic" "bengali" "chinese" "english" "german" "greek" "korean" "portuguese" "spanish" "turkish")

# Define model name and extract model type
MODEL_NAME="meta-llama/Llama-2-7b-hf"
MODEL_TYPE="${MODEL_NAME#*/}"

# Evaluate each language adapter on all other languages' tasks
for src_lang in "${src_languages[@]}"; do
    for tgt_lang in "${tgt_languages[@]}"; do
        if [ "$src_lang" != "$tgt_lang" ]; then
            # Create directory for storing evaluation results
            eval_dir="$BASE_OUTPUT_DIR/$MODEL_TYPE/cross_language/$src_lang/$tgt_lang"
            mkdir -p "$eval_dir"
            for task in ${combined_tasks[$tgt_lang]}; do
                # Run the evaluation and save the results
                CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval \
                --language $tgt_lang \
                --model_text llama_$src_lang \
                --task $task \
                --context False \
                --output_dir $eval_dir \
                --model_name $MODEL_NAME \
                --eval_type cross_culture \
                --debug\
            done
        fi
    done
done