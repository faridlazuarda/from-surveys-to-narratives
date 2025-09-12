import jsonlines
import os
import re
from sklearn.metrics import f1_score, accuracy_score
import fire
from peft import PeftModel
from tqdm import tqdm
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteriaList,
    StoppingCriteria
)
import torch
from .get_translated_prompt import getTranslatedPrompt  # Absolute import
import numpy as np
import random
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
access_token = os.getenv("HUGGINGFACE_TOKEN")
if access_token:
    login(token=access_token)
else:
    print("Warning: HUGGINGFACE_TOKEN not found in environment variables.")

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

context_dict = dict()

# Load context data
context_path = "/tud/value-adapters/culturellm/data/culture_context.jsonl"
with open(context_path, "r", encoding="utf8") as f:
    for item in jsonlines.Reader(f):
        for key in item.keys():
            context_dict[key] = item[key]

# Precompile regex patterns
ANSWER_REGEX = re.compile(r"### Answer:\s*(.*)", re.DOTALL)
SPLIT_REGEX = re.compile(r'[\n\.]')

# Custom stopping criteria (no changes needed here)
class StopOnAnswer(StoppingCriteria):
    def __init__(self, stop_string, tokenizer):
        self.stop_string = stop_string
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return self.stop_string in decoded_text

def get_valid_responses(task):
    if ('offensive_detect' in task.lower() and task != 'offensive_detect_finegrained') or 'abusive_detect' in task.lower():
        return ['1', '2', 'Offensive', 'Not offensive']
    elif 'hate_detect' in task.lower() and task != 'hate_detect_fine-grained':
        return ['1', '2', 'Hatespeech', 'Not Hatespeech']
    elif task == 'vulgar_detect_mp':
        return ['1', '2', 'Vulgar', 'Not Vulgar']
    elif task == 'spam_detect':
        return ['1', '2', 'Spam', 'Not Spam']
    elif task == 'hate_detect_fine-grained':
        return ['1', '2', '3', '4', '5', '6', '7', 'Not Hatespeech', 'Race', 'Religion', 'Ideology', 'Disability', 'Social Class', 'Gender']
    elif task == 'offensive_detect_finegrained':
        return ['1', '2', '3', '4', '5', 'Not hatespeech', 'Profanity', 'Offense towards a group', 'Offense towards an individual', 'Offense towards another entity']
    elif task == 'hate_off_detect':
        return ['1', '2', 'Hate or Offensive', 'Not Hate or Offensive']
    elif task in ['stereotype_detect', 'mockery_detect', 'insult_detect', 'improper_detect', 'aggressiveness_detect', 'toxicity_detect', 'negative_stance_detect', 'bias_on_gender_detect', 'homophobia_detect', 'racism_detect', 'misogyny_detect', 'threat_detect', 'hostility_directness_detect']:
        return ['0', '1', 'No', 'Yes']
    elif task == 'hate_offens_detect':
        return ['0', '1', 'No', 'Yes']
    else:
        return []  # For tasks where valid responses are not predefined

def getResponseBatch(prompts, country, context, lora_path, base_model=None, base_tokenizer=None, eval_type=None, adapter_lang=None, task=None):
    def base_prompt(prompt):
        return f"You are a {country} chatbot that knows {country} very well.\n{prompt}"

    # Modify prompts based on adapter_lang
    modified_prompts = []
    for prompt in prompts:
        if not isinstance(prompt, str):
            if isinstance(prompt, list):
                # If it's a list, join the elements into a single string
                prompt = ' '.join(map(str, prompt))
                print(f"Converted prompt list to string: {prompt}")
            else:
                # For other types, convert to string directly
                prompt = str(prompt)
                print(f"Converted non-string prompt to string: {prompt}")
        modified_prompts.append(base_prompt(prompt) if adapter_lang == country else prompt)

    # Tokenize all prompts in the batch with left padding
    input_ids = base_tokenizer(
        modified_prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    ).input_ids.to(base_model.device)

    # Generate outputs for the batch
    outputs = base_model.generate(
        input_ids,
        max_new_tokens=10,  # Adjust max_new_tokens as needed
        temperature=0.7,
        num_beams=1,
        do_sample=False,
        pad_token_id=base_tokenizer.pad_token_id,
        eos_token_id=base_tokenizer.eos_token_id,
    )

    # Decode all outputs
    output_texts = base_tokenizer.batch_decode(outputs, skip_special_tokens=True)

    responses = []
    invalid_responses = []

    for output_txt in output_texts:
        # Extract only the first line after "### Answer:"
        match = ANSWER_REGEX.search(output_txt)
        if match:
            response = match.group(1).strip()
            # Get the first line or sentence
            response = SPLIT_REGEX.split(response)[0].strip()
            # Define valid responses based on the task
            valid_responses = get_valid_responses(task)
            if any(vr.lower() in response.lower() for vr in valid_responses):
                invalid_response = False
            else:
                invalid_response = True
        else:
            # Attempt to extract response without regex if pattern not found
            try:
                response = output_txt.split("### Answer: ")[1].split()[0].strip()
                invalid_response = True
            except IndexError:
                response = ""
                invalid_response = True

        responses.append(response)
        invalid_responses.append(invalid_response)

    return output_texts, responses, invalid_responses

def getPrompt(input_txt, task, adapter_lang=None, eval_type=None):
    if eval_type == "translate" and adapter_lang:
        # Use the translated prompt if needed
        return getTranslatedPrompt(input_txt, task, adapter_lang.lower())
    else:
        if ('offensive_detect' in task.lower() and task != 'offensive_detect_finegrained') or 'abusive_detect' in task.lower():
            prompt = f"### Question: Is the following sentence offensive: {input_txt} \"1. Offensive\", \"2. Not offensive\" without explanation. ### Answer:"
        elif 'hate_detect' in task.lower() and task != 'hate_detect_fine-grained':
            prompt = f"### Question: Does the following sentence contain hate speech: {input_txt} \"1. Hatespeech\", \"2. Not Hatespeech\" without explanation. ### Answer:"
        elif task == 'vulgar_detect_mp':
            prompt = f"### Question: Does the following sentence contain vulgar speech: {input_txt} \"1. Vulgar\", \"2. Not Vulgar\" without explanation. ### Answer:"
        elif task == 'spam_detect':
            prompt = f"### Question: Is the following sentence a spam tweet: {input_txt} \"1. Spam\", \"2. Not Spam\" without explanation. ### Answer:"
        elif task == 'hate_detect_fine-grained':
            prompt = f"""### Question:
Does the following sentence contain hate speech?

{input_txt}

Please choose one of the following options without explanation:
1. Not Hatespeech
2. Race
3. Religion
4. Ideology
5. Disability
6. Social Class
7. Gender

### Answer:"""
        elif task == 'offensive_detect_finegrained':
            prompt = f"""### Question:
Does the following sentence contain offensive speech?

{input_txt}

Please choose one of the following options without explanation:
1. Not Hatespeech
2. Profanity, or non-targeted offense
3. Offense towards a group
4. Offense towards an individual
5. Offense towards another entity (non-human), often an event or organization

### Answer:"""
        elif task == 'hate_off_detect':
            prompt = f"### Question: Does the following sentence contain hate speech or offensive content: {input_txt} \"1. Hate or Offensive\", \"2. Not Hate or Offensive\" without explanation. ### Answer:"
        elif task in ['stereotype_detect', 'mockery_detect', 'insult_detect', 'improper_detect', 'aggressiveness_detect', 'toxicity_detect', 'negative_stance_detect', 'bias_on_gender_detect']:
            entity = task[:-7]
            if task in ['bias_on_gender_detect', 'hostility_directness_detect']:
                entity = entity.replace('_', ' ')
            prompt = f"### Question: Does the following sentence contain {entity}: {input_txt} \"0. No\", \"1. Yes\" without explanation. ### Answer:"
        elif task == 'hate_offens_detect':
            prompt = f"### Question: Does the following sentence contain hate speech: {input_txt} \"0. No\", \"1. Yes\" without explanation. ### Answer:"
        else:
            prompt = input_txt

        return prompt

def getDataPath(language, task):
    # Define paths based on language and task
    if task == 'GSM8K':
        path = 'data/forget/GSM8K_test.jsonl'
    elif language.lower() == 'arabic':
        if task == 'offensive_detect':
            path = "data/arabic/OffensEval2020/OffensEval.jsonl"
        elif task == 'hate_detect_osact4':
            path = "data/arabic/OSACT4/dev_data.jsonl"
        elif task == 'offensive_detect_osact4':
            path = "data/arabic/OSACT4/dev_data_offens.jsonl"
        elif task == 'offensive_detect_mp':
            path = "data/arabic/MP/offens.jsonl"
        elif task == 'hate_detect_mp':
            path = "data/arabic/MP/hateSpeech.jsonl"
        elif task == 'vulgar_detect_mp':
            path = "data/arabic/MP/VulgarSpeech.jsonl"
        elif task == 'spam_detect':
            path = "data/arabic/SpamDetect/span_detect_2.jsonl"
        elif task == 'offensive_detect_osact5':
            path = "data/arabic/OSACT5/offens.jsonl"
        elif task == 'hate_detect_osact5':
            path = "data/arabic/OSACT5/hateSpeech.jsonl"
        elif task == 'hate_detect_fine-grained':
            path = "data/arabic/OSACT5/hate_Finegrained.jsonl"
    elif language.lower() == 'bengali':
        if task == 'hate_detect_religion':
            path = "data/bengali/Bengali hate speech dataset/religion_data-2.jsonl"
        elif task == 'offensive_detect_1':
            path = "data/bengali/Trac2-Task1-Aggresion/aggression-data-2.jsonl"
        elif task == 'offensive_detect_2':
            path = "data/bengali/Trac2-Task2-Misogynistic/Misogynistic-data-2.jsonl"
        elif task == 'offensive_detect_3':
            path = "data/bengali/BAD-Bangla-Aggressive-Text-Dataset/data-2.jsonl"
        elif task == 'racism_detect':
            path = "data/bengali/Bangla-Abusive-Comment-Dataset/racism.jsonl"
        elif task == 'threat_detect':
            path = "data/bengali/Bangla-Abusive-Comment-Dataset/threat.jsonl"
    elif language.lower() == 'chinese':
        if task == 'bias_on_gender_detect':
            path = 'data/chinese/CDial-Bias/gender-2.jsonl'
        elif task == 'spam_detect':
            path = 'data/chinese/Chinese-Camouflage-Spam-dataset/data-2.jsonl'
    elif language.lower() == 'greek':
        if task == 'offensive_detect':
            path = "data/greek/OffensEval2020/OffensEval.jsonl"
        elif task == 'offensive_detect_g':
            path = 'data/greek/gazzetta/G-TEST-S-preprocessed.jsonl'
    elif language.lower() == 'english':
        if task == 'hate_detect_2':
            path = 'data/english/MLMA hate speech/data-2.jsonl'
        elif task == 'hostility_directness_detect':
            path = 'data/english/MLMA hate speech/directness.jsonl'
        elif task == 'hate_offens_detect':
            path = 'data/english/hate-speech-and-offensive-language/data.jsonl'
        elif task == 'offensive_detect_easy':
            path = 'data/english/SOLID/test_a_tweets_easy.jsonl'
        elif task == 'toxicity_detect':
            path = 'data/english/Toxic Comment Classification Challenge/toxic.jsonl'
        elif task == 'threat_detect':
            path = 'data/english/Toxic Comment Classification Challenge/threat.jsonl'
    elif language.lower() == 'german':
        if task == 'hate_detect':
            path = 'data/german/IWG_hatespeech_public/german_hatespeech_refugees_2.jsonl'
        elif task == 'hate_off_detect':
            path = 'data/german/HASOC/hate_off_detect.jsonl'
        elif task == 'offensive_detect_eval':
            path = 'data/german/GermEval/germeval2018.jsonl'
        elif task == 'hate_detect_check':
            path = 'data/german/MHC/hatecheck_cases_final_german.jsonl'
        elif task == 'hate_detect_iwg_1':
            path = 'data/german/IWG_hatespeech_public/german_hatespeech_refugees_1.jsonl'
    elif language.lower() == 'korean':
        if task == 'abusive_detect':
            path = 'data/korean/AbuseEval/data-2.jsonl'
        elif task == 'abusive_detect_2':
            path = 'data/korean/CADD/data-2.jsonl'
        elif task == 'abusive_detect_4':
            path = 'data/korean/Waseem/data-2.jsonl'
        elif task == 'hate_detect_3':
            path = 'data/korean/K-MHaS/data-2.jsonl'
        elif task == 'hate_detect_6':
            path = 'data/korean/Korean-Hate-Speech-Detection/data-2.jsonl'
        elif task == 'hate_detect_7':
            path = 'data/korean/KoreanHateSpeechdataset/data-2.jsonl'
    elif language.lower() == 'portuguese':
        if task == 'homophobia_detect':
            path = 'data/portuguese/ToLD-Br/homophobia.jsonl'
        elif task == 'insult_detect':
            path = 'data/portuguese/ToLD-Br/insult.jsonl'
        elif task == 'misogyny_detect':
            path = 'data/portuguese/ToLD-Br/misogyny.jsonl'
        elif task == 'offensive_detect_2':
            path = "data/portuguese/OffComBR/data.jsonl"
        elif task == 'offensive_detect_3':
            path = "data/portuguese/HateBR/data-2.jsonl"
    elif language.lower() == 'spanish':
        if task == 'offensive_detect_ami':
            path = 'data/spanish/AMI IberEval 2018_offens/data-2.jsonl'
        elif task == 'offensive_detect_mex_a3t':
            path = 'data/spanish/MEX-A3T_offens/data-2.jsonl'
        elif task == 'offensive_detect_mex_offend':
            path = 'data/spanish/OffendES_offens/data-2.jsonl'
        elif task == 'hate_detect_eval':
            path = 'data/spanish/HateEval 2019_HS/data-2.jsonl'
        elif task == 'hate_detect_haterNet':
            path = 'data/spanish/HaterNet_HS/data-2.jsonl'
        elif task == 'stereotype_detect':
            path = 'data/spanish/DETOXIS 2021/stereotype.jsonl'
        elif task == 'mockery_detect':
            path = 'data/spanish/DETOXIS 2021/mockery.jsonl'
        elif task == 'insult_detect':
            path = 'data/spanish/DETOXIS 2021/insult.jsonl'
        elif task == 'improper_detect':
            path = 'data/spanish/DETOXIS 2021/improper_language.jsonl'
        elif task == 'aggressiveness_detect':
            path = 'data/spanish/DETOXIS 2021/aggressiveness.jsonl'
        elif task == 'negative_stance_detect':
            path = 'data/spanish/DETOXIS 2021/negative_stance.jsonl'
    elif language.lower() == 'turkish':
        if task == 'offensive_detect':
            path = "data/turkish/OffensEval2020/OffensEval.jsonl"
        elif task == 'offensive_detect_corpus':
            path = "data/turkish/offenseCorpus/offens.jsonl"
        elif task == 'offensive_detect_finegrained':
            path = "data/turkish/offenseCorpus/offens_fine-grained.jsonl"
        elif task == 'offensive_detect_kaggle':
            path = "data/turkish/offenssDetect-kaggle/turkish_tweets_2020.jsonl"
        elif task == 'offensive_detect_kaggle2':
            path = "data/turkish/offensDetect-kaggle2/test.jsonl"
        elif task == 'abusive_detect':
            path = "data/turkish/ATC/fold_0_test.jsonl"
        elif task == 'spam_detect':
            path = "data/turkish/TurkishSpam/trspam.jsonl"
    else:
        raise ValueError(f"Unsupported language: {language}")

    absolute = "/tud/value-adapters/culturellm/"
    return os.path.join(absolute, path)

def postprocess(task, output):
    if output == '':
        return output

    output = output.lower()

    if ('offensive_detect' in task.lower() and task != 'offensive_detect_finegrained') or 'abusive_detect' in task.lower():
        if '1' in output or 'offensive' in output:
            prediction = 'OFF'
        elif '2' in output or 'not offensive' in output:
            prediction = 'NOT'
        else:
            prediction = output
    elif 'hate_detect' in task.lower() and task != 'hate_detect_fine-grained':
        if '1' in output or 'hatespeech' in output:
            prediction = 'HS'
        elif '2' in output or 'not hatespeech' in output:
            prediction = 'NOT_HS'
        else:
            prediction = output
    elif task == 'vulgar_detect_mp':
        if '1' in output or 'vulgar' in output:
            prediction = 'V'
        elif '2' in output or 'not vulgar' in output:
            prediction = '-'
        else:
            prediction = output
    elif task == 'spam_detect':
        if '1' in output or 'spam' in output:
            prediction = 'Spam'
        elif '2' in output or 'not spam' in output:
            prediction = 'Ham'
        else:
            prediction = output
    elif task == 'hate_detect_fine-grained':
        if '1' in output or 'not hatespeech' in output:
            prediction = 'NOT_HS'
        elif '2' in output or 'race' in output:
            prediction = 'HS1'
        elif '3' in output or 'religion' in output:
            prediction = 'HS2'
        elif '4' in output or 'ideology' in output:
            prediction = 'HS3'
        elif '5' in output or 'disability' in output:
            prediction = 'HS4'
        elif '6' in output or 'social class' in output:
            prediction = 'HS5'
        elif '7' in output or 'gender' in output:
            prediction = 'HS6'
        else:
            prediction = output
    elif task == 'offensive_detect_finegrained':
        if '1' in output or 'not hatespeech' in output:
            prediction = 'non'
        elif '2' in output or 'profanity' in output:
            prediction = 'prof'
        elif '3' in output or 'offense towards a group' in output:
            prediction = 'grp'
        elif '4' in output or 'offense towards an individual' in output:
            prediction = 'indv'
        elif '5' in output or 'offense towards another entity' in output:
            prediction = 'oth'
        else:
            prediction = output
    elif task == 'hate_off_detect':
        if '2' in output or 'not hate or offensive' in output:
            prediction = 'NOT'
        elif '1' in output or 'hate or offensive' in output:
            prediction = 'HOF'
        else:
            prediction = output
    elif task in ['stereotype_detect', 'mockery_detect', 'insult_detect', 'improper_detect', 'aggressiveness_detect', 'toxicity_detect', 'negative_stance_detect', 'bias_on_gender_detect', 'homophobia_detect', 'racism_detect', 'misogyny_detect', 'threat_detect', 'hostility_directness_detect']:
        if '1' in output or 'yes' in output:
            prediction = '1'
        elif '0' in output or 'no' in output:
            prediction = '0'
        else:
            prediction = output
    elif task == 'hate_offens_detect':
        if '0' in output or 'no' in output:
            prediction = '0'
        elif '1' in output or 'yes' in output:
            prediction = '1'
        else:
            prediction = '2'
    elif task == 'GSM8K':
        pattern = r'\d+,\d+'
        matches = re.findall(pattern, output)
        formatted_numbers = [match.replace(',', '') for match in matches if ',' in match]

        if not formatted_numbers:
            pattern = r'\d+'
            matches = re.findall(pattern, output)
            formatted_numbers = matches

        prediction = formatted_numbers[-1] if formatted_numbers else output

    return prediction

def computeMetrics(task, gt_list, pred_list):
    if ('offensive_detect' in task.lower() and task != 'offensive_detect_finegrained') or 'abusive_detect' in task.lower():
        final_f1_score = f1_score(gt_list, pred_list, labels=['OFF', 'NOT'], average='macro')
    elif 'hate_detect' in task.lower() and task != 'hate_detect_fine-grained':
        final_f1_score = f1_score(gt_list, pred_list, labels=['HS', 'NOT_HS'], average='macro')
    elif task == 'vulgar_detect_mp':
        final_f1_score = f1_score(gt_list, pred_list, labels=['V', '-'], average='macro')
    elif task == 'spam_detect':
        final_f1_score = f1_score(gt_list, pred_list, labels=['Spam', 'Ham'], average='macro')
    elif task == 'hate_detect_fine-grained':
        final_f1_score = f1_score(gt_list, pred_list, labels=['NOT_HS', 'HS1', 'HS2', 'HS3', 'HS4', 'HS5', 'HS6'], average='macro')
    elif task == 'offensive_detect_finegrained':
        final_f1_score = f1_score(gt_list, pred_list, labels=['non', 'prof', 'grp', 'indv', 'oth'], average='macro')
    elif task == 'hate_off_detect':
        final_f1_score = f1_score(gt_list, pred_list, labels=['HOF', 'NOT'], average='macro')
    elif task in ['stereotype_detect', 'mockery_detect', 'insult_detect', 'improper_detect', 'aggressiveness_detect', 'toxicity_detect', 'negative_stance_detect', 'bias_on_gender_detect', 'homophobia_detect', 'racism_detect', 'misogyny_detect', 'threat_detect', 'hostility_directness_detect']:
        final_f1_score = f1_score(gt_list, pred_list, labels=['0', '1'], average='macro')
    elif task == 'hate_offens_detect':
        final_f1_score = f1_score(gt_list, pred_list, labels=['0', '1'], average='macro')
    elif task == 'GSM8K':
        final_f1_score = accuracy_score(gt_list, pred_list)
    else:
        final_f1_score = 0.0  # Default if task not recognized

    return final_f1_score

def run(
    test_lang, 
    task, 
    output_dir, 
    context, 
    model_name, 
    model_type, 
    adapter_lang=None,
    eval_type='single', 
    debug=False,
    lora_path=None,
    batch_size=1,  # Added batch_size parameter
    override_result=False  # Added override_result parameter
    ):
    if isinstance(adapter_lang, tuple):
        adapter_lang = ','.join(adapter_lang)
    
    print(f"Adapter Language: {adapter_lang}")
    print(f"Test Language: {test_lang}")
    print(f"Task: {task}")
    print(f"Output Directory: {output_dir}")
    print(f"Context: {context}")
    print(f"Model Name: {model_name}")
    print(f"Model Type: {model_type}")
    print(f"Evaluation Type: {eval_type}")
    print(f"Debug Mode: {debug}")
    print(f"LoRA Path: {lora_path}")
    print(f"Batch Size: {batch_size}")
    print(f"Override Result: {override_result}")
    
    set_seed()

    # Determine directory path based on conditions
    if debug:
        dir_path = os.path.join(
            output_dir,
            model_name.split("/")[-1],
            "debug",
            adapter_lang.lower(),
            test_lang.lower()
        )
    elif eval_type == "zero-shot":
        dir_path = os.path.join(
            output_dir,
            model_name.split("/")[-1],
            test_lang.lower()
        )
    else:
        dir_path = os.path.join(
            output_dir,
            model_name.split("/")[-1],
            adapter_lang.lower(),
            test_lang.lower()
        )

    # Append suffix if debugging
    suffix = "_debug" if debug else ""

    # Construct the result file paths
    res_file = os.path.join(dir_path, f"{task}_res{suffix}.jsonl")
    predictions_file = os.path.join(dir_path, f"{task}_predictions{suffix}.jsonl")

    # Check if result files exist
    if not override_result and os.path.exists(res_file) and os.path.exists(predictions_file):
        print(f"Results for task '{task}' already exist in '{dir_path}'. Skipping experiment.")
        return

    # Ensure the directory exists
    os.makedirs(dir_path, exist_ok=True)

    # Print the final directory path
    print("Output Directory:", dir_path)

    def load_data(languages, task):
        gt_list = []
        for lang in languages:
            path = getDataPath(lang.lower(), task)
            if not os.path.exists(path):
                print(f"Warning: Data path does not exist: {path}")
                continue
            with open(path, "r", encoding="utf8") as f:
                for i, item in enumerate(jsonlines.Reader(f)):
                    label = item['answer'] if task == 'GSM8K' else item['label']
                    gt_list.append(label)
                    if debug and i >= 19:
                        break
        return gt_list

    def get_model_and_tokenizer(language, model_name):
        compute_dtype = getattr(torch, "float16")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )

        base_model_name = model_name.split('/')[-1].lower()

        if eval_type != "zero-shot":
            if lora_path is None:
                raise ValueError(f"lora_path is required for language: {language}")

        # Load the model
        pretrained_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            use_auth_token=access_token
        )

        if eval_type != "zero-shot":
            print("Loading LoRA adapter...")
            final_model = PeftModel.from_pretrained(
                pretrained_model, 
                lora_path, 
                device_map="cuda:0"
            )
        else:
            final_model = pretrained_model

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=access_token)

        # Fix the pad_token issue
        if tokenizer.pad_token is None:
            print("Tokenizer does not have a pad_token. Setting pad_token to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
            # Resize the model embeddings to include the new pad_token
            final_model.resize_token_embeddings(len(tokenizer))
        else:
            print(f"Tokenizer pad_token: {tokenizer.pad_token}")

        # Set padding_side to 'left' for decoder-only models
        tokenizer.padding_side = 'left'
        # print(f"Tokenizer padding_side set to: {tokenizer.padding_side}")

        final_model.eval()  # Ensure model is in eval mode

        return final_model, tokenizer

    from tqdm import tqdm  # Ensure tqdm is imported

    def process_data(languages, task, context, lora_path, pretrained_model, base_tokenizer, gt_list, batch_size):
        pred_list = []
        results_list = []
        gt_index = 0
        invalid_count = 0  # Initialize invalid response count

        with torch.no_grad():  # Disable gradient calculations
            for lang in languages:
                path = getDataPath(lang.lower(), task)
                if not os.path.exists(path):
                    print(f"Warning: Data path does not exist: {path}")
                    continue
                with open(path, "r", encoding="utf8") as f:
                    reader = jsonlines.Reader(f)
                    iterator = iter(reader)
                    
                    # **Initialize tqdm for batch processing**
                    # To prevent division by zero
                    total_batches = len(gt_list) // batch_size if batch_size > 0 else 0
                    with tqdm(desc=f"Processing {lang} - {task}", unit='batch', total=total_batches) as pbar:
                        while True:
                            batch_items = []
                            try:
                                for _ in range(batch_size):
                                    item = next(iterator)
                                    input_txt = item.get('comment') or item.get('data') or item.get('tweet') or item.get('question')
                                    
                                    # **Check if input_txt is a string**
                                    if not isinstance(input_txt, str):
                                        if isinstance(input_txt, list):
                                            # If it's a list, join the elements into a single string
                                            input_txt = ' '.join(map(str, input_txt))
                                            print(f"Converted list to string: {input_txt}")
                                        else:
                                            # For other types, convert to string directly
                                            input_txt = str(input_txt)
                                            print(f"Converted non-string to string: {input_txt}")

                                    # **Handle empty strings**
                                    if not input_txt.strip():
                                        print(f"Warning: Empty input text in item: {item}")
                                        continue
                                    
                                    prompt = getPrompt(input_txt, task, adapter_lang=adapter_lang, eval_type=eval_type)
                                    batch_items.append((input_txt, prompt, item))
                                    if debug and len(batch_items) >= 20:
                                        break
                            except StopIteration:
                                pass

                            if not batch_items:
                                break  # No more items to process

                            # Extract prompts
                            prompts = [item[1] for item in batch_items]

                            # **Ensure all prompts are strings**
                            if not all(isinstance(p, str) for p in prompts):
                                print(f"Warning: Not all prompts are strings: {prompts}")
                                continue

                            # Get responses in batch
                            output_texts, responses, invalid_responses = getResponseBatch(
                                prompts=prompts, 
                                country=lang, 
                                context=context, 
                                lora_path=lora_path, 
                                base_model=pretrained_model, 
                                base_tokenizer=base_tokenizer, 
                                eval_type=eval_type,
                                adapter_lang=adapter_lang,
                                task=task
                            )

                            # Postprocess each response
                            for i in range(len(batch_items)):
                                input_txt, _, original_item = batch_items[i]
                                output_txt = output_texts[i]
                                response = responses[i]
                                invalid_response = invalid_responses[i]

                                if invalid_response:
                                    invalid_count += 1

                                prediction = postprocess(task, response)
                                pred_list.append(prediction)
                                results_list.append({
                                    "input": input_txt, 
                                    "output": output_txt,  # Store full output
                                    "extracted_output": response, 
                                    "prediction": prediction,
                                    "label": gt_list[gt_index] if gt_index < len(gt_list) else None,
                                    "invalid_response": invalid_response
                                })
                                gt_index += 1
                                if debug and gt_index >= 20:
                                    break

                            # **Update the progress bar after processing the batch**
                            pbar.update(1)

                            if debug and gt_index >= 20:
                                break

        return pred_list, results_list, invalid_count

    # Define combined languages
    combined_languages = [
        'arabic', 'bengali', 'chinese', 'english', 'german',
        'greek', 'korean', 'portuguese', 'spanish', 'turkish'
    ]
    languages = combined_languages if test_lang.lower() == "combined" else [test_lang]

    # Load ground truth labels
    gt_list = load_data(languages, task)

    # Load model and tokenizer
    if eval_type != "zero-shot":
        pretrained_model, base_tokenizer = get_model_and_tokenizer(
            "combined" if adapter_lang.lower() == "combined" else adapter_lang.lower(), 
            model_name
        )
    else:
        pretrained_model, base_tokenizer = get_model_and_tokenizer(
            adapter_lang.lower(), 
            model_name
        )

    print("Test Languages: ", languages)
    print("Adapter Path: ", lora_path)

    # Process data and get predictions
    pred_list, results_list, invalid_count = process_data(
        languages=languages, 
        task=task, 
        context=context, 
        lora_path=lora_path, 
        pretrained_model=pretrained_model, 
        base_tokenizer=base_tokenizer,
        gt_list=gt_list,
        batch_size=batch_size  # Pass the batch_size
    )

    # Compute metrics
    final_f1_score = computeMetrics(task, gt_list, pred_list)

    print('F1 Score:', final_f1_score)
    print(f'Total invalid responses: {invalid_count} out of {len(gt_list)}')

    # Save results
    with jsonlines.open(os.path.join(dir_path, f"{task}_res{suffix}.jsonl"), mode='a') as writer:
        writer.write({
            'Model': model_name, 
            "f1_score": final_f1_score, 
            'Context': context,
            'total_invalid_responses': invalid_count,
            'total_samples': len(gt_list)
        })

    with jsonlines.open(os.path.join(dir_path, f"{task}_predictions{suffix}.jsonl"), mode='a') as writer:
        for result in results_list:
            writer.write(result)

if __name__ == '__main__':
    fire.Fire(run)
