#!/bin/bash
#SBATCH --job-name="single-1"
#SBATCH --output="/tud/outputs/run_zeroshot.out"
#SBATCH --gres=gpu:1
#SBATCH -N 1 
#SBATCH -A nlp-dept 
#SBATCH -p nlp-dept 
#SBATCH -q nlp-pool

nvidia-smi

echo $CUDA_VISIBLE_DEVICES

# Set the visible device
VISIBLE_DEVICE=6
MODEL_NAME='google/gemma-2-2b-it'
# MODEL_NAME='meta-llama/Meta-Llama-3.1-8B-Instruct'
# MODEL_NAME='meta-llama/Llama-2-7b-hf'
# MODEL_NAME="meta-llama/Meta-Llama-3.1-8B"
# MODEL_NAME="meta-llama/Meta-Llama-3.1-8B"
# MODEL_NAME="meta-llama/Llama-2-7b-chat-hf"


# MODEL_TYPE="llama-2-7b-chat-hf"
# MODEL_TYPE="meta-llama-3.1-8b"
# MODEL_TYPE="meta-llama-3.1-8b"
# MODEL_TYPE="gemma-2-9b-it"
MODEL_TYPE="llama-3.1-8B-it"
# MODEL_TYPE="llama-2-7b-hf"

EVAL_TYPE="zero-shot"
MODEL_TYPE="${MODEL_NAME#*/}"

OUT_DIR="/tud/value-adapters/culturellm/eval_outputs"


# english
for task in 'hate_detect_2' 'hate_offens_detect' 'hostility_directness_detect' 'offensive_detect_easy' 'threat_detect' 'toxicity_detect'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval \
    --language english \
    --task $task \
    --output_dir $OUT_DIR \
    --context False \
    --model_name $MODEL_NAME \
    --base_model_name $MODEL_TYPE \
    --eval_type $EVAL_TYPE
done

# arabic
for task in 'offensive_detect' 'offensive_detect_osact4' 'offensive_detect_mp' 'offensive_detect_osact5' 'hate_detect_osact4' 'hate_detect_mp' 'hate_detect_osact5' 'vulgar_detect_mp' 'spam_detect' 'hate_detect_fine-grained'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval \
    --language arabic \
    --task $task \
    --output_dir $OUT_DIR \
    --context False \
    --model_name $MODEL_NAME \
    --base_model_name $MODEL_TYPE \
    --eval_type $EVAL_TYPE 
done

# bengali
for task in 'hate_detect_religion' 'offensive_detect_1' 'offensive_detect_2' 'offensive_detect_3' 'racism_detect' 'threat_detect'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval \
    --language bengali \
    --task $task \
    --output_dir $OUT_DIR \
    --context False \
    --model_name $MODEL_NAME \
    --base_model_name $MODEL_TYPE \
    --eval_type $EVAL_TYPE
done

# chinese
for task in 'bias_on_gender_detect' 'spam_detect'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval \
    --language chinese \
    --task $task \
    --output_dir $OUT_DIR \
    --context False \
    --model_name $MODEL_NAME \
    --base_model_name $MODEL_TYPE \
    --eval_type $EVAL_TYPE
done

# greek
for task in 'offensive_detect' 'offensive_detect_g'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval \
    --language greek \
    --task $task \
    --output_dir $OUT_DIR \
    --context False \
    --model_name $MODEL_NAME \
    --base_model_name $MODEL_TYPE \
    --eval_type $EVAL_TYPE
done

# german
for task in 'hate_detect' 'hate_off_detect' 'hate_detect_iwg_1' 'hate_detect_check' 'offensive_detect_eval'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval \
    --language german \
    --task $task \
    --output_dir $OUT_DIR \
    --context False \
    --model_name $MODEL_NAME \
    --base_model_name $MODEL_TYPE \
    --eval_type $EVAL_TYPE
done

# korean
for task in 'abusive_detect' 'abusive_detect_2' 'abusive_detect_4' 'hate_detect_3' 'hate_detect_6' 'hate_detect_7'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval \
    --language korean \
    --task $task \
    --output_dir $OUT_DIR \
    --context False \
    --model_name $MODEL_NAME \
    --base_model_name $MODEL_TYPE \
    --eval_type $EVAL_TYPE
done

# portuguese
for task in 'homophobia_detect' 'insult_detect' 'misogyny_detect' 'offensive_detect_2' 'offensive_detect_3'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval \
    --language portuguese \
    --task $task \
    --output_dir $OUT_DIR \
    --context False \
    --model_name $MODEL_NAME \
    --base_model_name $MODEL_TYPE \
    --eval_type $EVAL_TYPE
done

# spanish
for task in 'offensive_detect_ami' 'offensive_detect_mex_a3t' 'offensive_detect_mex_offend' 'hate_detect_eval' 'hate_detect_haterNet' 'stereotype_detect' 'mockery_detect' 'insult_detect' 'improper_detect' 'aggressiveness_detect' 'negative_stance_detect'
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval \
    --language spanish \
    --task $task \
    --output_dir $OUT_DIR \
    --context False \
    --model_name $MODEL_NAME \
    --base_model_name $MODEL_TYPE \
    --eval_type $EVAL_TYPE
done

# turkish
for task in 'offensive_detect' 'offensive_detect_corpus' 'offensive_detect_finegrained' 'offensive_detect_kaggle' 'offensive_detect_kaggle2' 'abusive_detect' 
do 
    CUDA_VISIBLE_DEVICES=$VISIBLE_DEVICE python -m src.test_offensEval \
    --language turkish \
    --task $task \
    --output_dir $OUT_DIR \
    --context False \
    --model_name $MODEL_NAME \
    --base_model_name $MODEL_TYPE \
    --eval_type $EVAL_TYPE
done

