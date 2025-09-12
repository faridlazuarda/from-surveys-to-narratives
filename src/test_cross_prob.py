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
import gc  # Import garbage collector

from dotenv import load_dotenv

# Clear CUDA cache and collect garbage
torch.cuda.empty_cache()
gc.collect()

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

# Custom stopping criteria (no changes needed here)
class StopOnAnswer(StoppingCriteria):
    def __init__(self, stop_string, tokenizer):
        self.stop_string = stop_string
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return self.stop_string in decoded_text

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
    elif language.lower() == 'greek':
        if task == 'offensive_detect':
            path = "data/greek/OffensEval2020/OffensEval.jsonl"
        elif task == 'offensive_detect_g':
            path = 'data/greek/gazzetta/G-TEST-S-preprocessed.jsonl'
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
    elif language.lower() == 'bengali':
        if task == 'offensive_detect_1':
            path = 'data/bengali/Trac2-Task1-Aggresion/aggression-data-2.jsonl'
        elif task == 'offensive_detect_2':
            path = 'data/bengali/Trac2-Task2-Misogynistic/Misogynistic-data-2.jsonl'
        elif task == 'offensive_detect_3':
            path = 'data/bengali/BAD-Bangla-Aggressive-Text-Dataset/data-2.jsonl'
        elif task == 'hate_detect_religion':
            path = 'data/bengali/Bengali hate speech dataset/religion_data-2.jsonl'
        elif task == 'threat_detect':
            path = 'data/bengali/Bangla-Abusive-Comment-Dataset/threat.jsonl'
        elif task == 'racism_detect':
            path = 'data/bengali/Bangla-Abusive-Comment-Dataset/racism.jsonl'   
    elif language.lower() == 'chinese':
        if task == 'spam_detect':
            path = 'data/chinese/Chinese-Camouflage-Spam-dataset/data-2.jsonl'
        elif task == 'bias_on_gender_detect':
            path = 'data/chinese/CDial-Bias/gender-2.jsonl'
    elif language.lower() == 'korean':
        if task == 'hate_detect_3':
            path = 'data/korean/K-MHaS/data-2.jsonl'
        elif task == 'hate_detect_6':
            path = 'data/korean/Korean-Hate-Speech-Detection/data-2.jsonl'
        elif task == 'hate_detect_7':
            path = 'data/korean/KoreanHateSpeechdataset/data-2.jsonl'
        elif task == 'abusive_detect':
            path = 'data/korean/AbuseEval/data-2.jsonl'
        elif task == 'abusive_detect_2':
            path = 'data/korean/CADD/data-2.jsonl'
        elif task == 'abusive_detect_4':
            path = 'data/korean/Waseem/data-2.jsonl'
    elif language.lower() == 'portuguese':
        if task == 'offensive_detect_2':
            path = "data/portuguese/OffComBR/data.jsonl"
        elif task == 'offensive_detect_3':
            path = "data/portuguese/HateBR/data-2.jsonl"
        elif task == 'homophobia_detect':
            path = 'data/portuguese/ToLD-Br/homophobia.jsonl'
        elif task == 'misogyny_detect':
            path = 'data/portuguese/ToLD-Br/misogyny.jsonl'
        elif task == 'insult_detect':
            path = 'data/portuguese/ToLD-Br/insult.jsonl'
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
    else:
        raise ValueError(f"Unsupported language: {language}")
    
    absolute = "/tud/value-adapters/culturellm/"
    return os.path.join(absolute, path)

# New function to define label mappings
def getLabelMap(task):
    # Define possible labels and their mappings based on the task
    if ('offensive_detect' in task.lower() and task != 'offensive_detect_finegrained') or 'abusive_detect' in task.lower():
        label_map = {
            "1. Offensive": "OFF",
            "2. Not offensive": "NOT"
        }
    elif 'hate_detect' in task.lower() and task != 'hate_detect_fine-grained':
        label_map = {
            "1. Hatespeech": "HS",
            "2. Not Hatespeech": "NOT_HS"
        }
    elif task == 'vulgar_detect_mp':
        label_map = {
            "1. Vulgar": "V",
            "2. Not Vulgar": "-"
        }
    elif task == 'spam_detect':
        label_map = {
            "1. Spam": "Spam",
            "2. Not Spam": "Ham"
        }
    elif task == 'hate_detect_fine-grained':
        label_map = {
            "1. Not Hatespeech": "NOT_HS",
            "2. Race": "HS1",
            "3. Religion": "HS2",
            "4. Ideology": "HS3",
            "5. Disability": "HS4",
            "6. Social Class": "HS5",
            "7. Gender": "HS6"
        }
    elif task == 'offensive_detect_finegrained':
        label_map = {
            "1. Not hatespeech": "non",
            "2. Profanity, or non-targeted offense": "prof",
            "3. Offense towards a group": "grp",
            "4. Offense towards an individual": "indv",
            "5. Offense towards an other (non-human) entity, often an event or organization": "oth"
        }
    elif task == 'hate_off_detect':
        label_map = {
            "1. Hate or Offensive": "HOF",
            "2. Not Hate or Offensive": "NOT"
        }
    elif task in [
        'stereotype_detect', 'mockery_detect', 'insult_detect', 'improper_detect',
        'aggressiveness_detect', 'toxicity_detect', 'negative_stance_detect',
        'bias_on_gender_detect', 'homophobia_detect', 'racism_detect', 'misogyny_detect',
        'threat_detect', 'hostility_directness_detect'
    ]:
        label_map = {
            "0. No": "0",
            "1. Yes": "1"
        }
    elif task == 'hate_offens_detect':
        label_map = {
            "0. No": "0",
            "1. Yes": "1"
        }
    else:
        # For tasks without predefined labels, raise an error
        raise ValueError(f"No label mapping defined for task: {task}")
    return label_map

def getResponse(batch_prompts, country, context, lora_path, base_model=None, base_tokenizer=None, eval_type=None, adapter_lang=None, task=None):
    """
    Processes a batch of prompts and returns normalized probabilities for each possible answer.
    """
    def base_prompt(prompt):
        return f"You are a {country} chatbot that knows {country} very well.\n{prompt}"

    if eval_type == "single":
        batch_prompts = [base_prompt(prompt) for prompt in batch_prompts]

    # Skip empty batches
    if not batch_prompts:
        print("Warning: Empty batch received")
        return []

    # Tokenize all prompts in the batch with padding
    encoding = base_tokenizer(
        batch_prompts, 
        return_tensors="pt", 
        padding=True, 
        truncation=True,
        max_length=512  # Add explicit max length
    ).to("cuda")
    
    # Get the possible labels based on the task
    label_map = getLabelMap(task)
    possible_answers = list(label_map.keys())
    
    if not possible_answers:
        raise ValueError(f"No possible answers defined in label_map for task: {task}")
    
    # Pre-tokenize all possible answers
    answer_encodings = base_tokenizer(
        possible_answers,
        return_tensors="pt", 
        padding=True,
        truncation=True,
        max_length=128  # Add explicit max length for answers
    ).to("cuda")
    
    # Error handling for empty or invalid encodings
    if answer_encodings.input_ids.numel() == 0:
        raise ValueError("Empty answer encodings")
    
    answer_ids = answer_encodings.input_ids
    if answer_ids.dim() == 1:
        answer_ids = answer_ids.unsqueeze(0)

    # Get the logits for all prompts with error handling
    try:
        with torch.no_grad():
            outputs = base_model(input_ids=encoding.input_ids)
            logits = outputs.logits
    except Exception as e:
        print(f"Error in model forward pass: {e}")
        return [{}] * len(batch_prompts)  # Return empty probabilities

    # Get the last token's logits
    last_token_logits = logits[:, -1, :]
    batch_size = last_token_logits.size(0)
    log_probs = []

    # Calculate log probabilities for each answer
    for ans_id in answer_ids:
        if ans_id.dim() == 1:
            ans_id = ans_id.unsqueeze(0)
        
        # Repeat token_id across the batch
        token_id = ans_id[0, 0].unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1)  # Shape: [batch_size, 1]
        
        # Gather log probabilities
        token_logprob = torch.log_softmax(last_token_logits, dim=-1).gather(1, token_id).squeeze(-1)  # Shape: [batch_size]
        total_logprob = token_logprob

        # Process remaining tokens
        input_ids_ans = encoding.input_ids.clone()
        
        for j in range(1, ans_id.size(1)):
            if ans_id[0, j].item() == base_tokenizer.pad_token_id:
                break
                
            token_id = ans_id[0, j].unsqueeze(0).unsqueeze(-1).repeat(batch_size, 1)  # Shape: [batch_size, 1]
            new_input_ids = torch.cat([input_ids_ans, token_id], dim=1)  # Shape: [batch_size, current_seq_len + 1]
            
            try:
                with torch.no_grad():
                    outputs = base_model(input_ids=new_input_ids)
                    new_logits = outputs.logits[:, -1, :]
                token_logprob = torch.log_softmax(new_logits, dim=-1).gather(1, token_id).squeeze(-1)  # Shape: [batch_size]
                total_logprob += token_logprob
            except Exception as e:
                print(f"Error processing token {j}: {e}")
                continue
                
            input_ids_ans = new_input_ids

        log_probs.append(total_logprob)

    # Stack and normalize probabilities
    try:
        log_probs = torch.stack(log_probs, dim=0).transpose(0, 1)  # Shape: [batch_size, num_answers]
        max_logprob = torch.max(log_probs, dim=1, keepdim=True)[0]
        exp_probs = torch.exp(log_probs - max_logprob)
        normalized_probs = exp_probs / torch.sum(exp_probs, dim=1, keepdim=True)
    except Exception as e:
        print(f"Error in probability normalization: {e}")
        return [{}] * len(batch_prompts)

    # Convert to list of dictionaries with error handling
    probs_list = []
    for b in range(batch_size):
        try:
            probs = {label_map[ans]: normalized_probs[b, i].item() 
                    for i, ans in enumerate(possible_answers)}
            probs_list.append(probs)
        except Exception as e:
            print(f"Error creating probability dictionary for batch {b}: {e}")
            probs_list.append({})

    return probs_list

def getPrompt(input_txt, task, adapter_lang=None, eval_type=None):
    if eval_type == "translate" and adapter_lang:
        return getTranslatedPrompt(input_txt, task, adapter_lang.lower())
    else:
        if ('offensive_detect' in task.lower() and task != 'offensive_detect_finegrained') or 'abusive_detect' in task.lower():
            prompt = f"### Question: is the following sentence offensive: {input_txt}. \"1. Offensive\", \"2. Not offensive\" without explanation. ### Answer:"
        elif 'hate_detect' in task.lower() and task != 'hate_detect_fine-grained':
            prompt = f"### Question: does the following sentence contain hate speech: {input_txt}. \"1. Hatespeech\", \"2. Not Hatespeech\" without explanation. ### Answer:"
        elif task == 'vulgar_detect_mp':
            prompt = f"### Question: does the following sentence contain vulgar speech: {input_txt}. \"1. Vulgar\", \"2. Not Vulgar\" without explanation. ### Answer:"
        elif task == 'spam_detect':
            prompt = f"### Question: is the following sentence a spam tweet: {input_txt}. \"1. Spam\", \"2. Not Spam\" without explanation. ### Answer:"
        elif task == 'hate_detect_fine-grained':
            prompt = f"""
### Question:
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

### Answer:
"""
        elif task == 'offensive_detect_finegrained':
            prompt = f"""
### Question:
Does the following sentence contain offensive speech?

{input_txt}

Please choose one of the following options without explanation:
1. Not hatespeech
2. Profanity, or non-targeted offense
3. Offense towards a group
4. Offense towards an individual
5. Offense towards an other (non-human) entity, often an event or organization

### Answer:
"""
        elif task == 'hate_off_detect':
            prompt = f"### Question: does the following sentence contain hate speech or offensive content: {input_txt}. \"1. Hate or Offensive\", \"2. Not Hate or Offensive\" without explanation. ### Answer:"
        elif task in [
            'stereotype_detect', 'mockery_detect', 'insult_detect', 'improper_detect',
            'aggressiveness_detect', 'toxicity_detect', 'negative_stance_detect',
            'homophobia_detect', 'racism_detect', 'misogyny_detect', 'threat_detect',
            'hostility_directness_detect'
        ]:
            entity = task[:-7]
            if task in ['bias_on_gender_detect', 'hostility_directness_detect']:
                entity = entity.replace('_', ' ')
            prompt = f"### Question: does the following sentence contain {entity}: {input_txt}. \"0. No\", \"1. Yes\" without explanation. ### Answer:"
        elif task == 'hate_offens_detect':
            prompt = f"### Question: does the following sentence contain hate speech: {input_txt}. \"0. No\", \"1. Yes\" without explanation. ### Answer:"
        else:
            prompt = input_txt

    return prompt

def postprocess(task, probs):
    """
    Selects the label with the highest probability.

    Args:
        task (str): Task name.
        probs (Dict[str, float]): Dictionary of label probabilities.

    Returns:
        str: Predicted label.
    """
    if not probs:
        return "Unknown"

    prediction = max(probs, key=probs.get)
    return prediction

def computeMetrics(task, gt_list, pred_list):
    """
    Computes the F1 score or accuracy based on the task.

    Args:
        task (str): Task name.
        gt_list (List[str]): Ground truth labels.
        pred_list (List[str]): Predicted labels.

    Returns:
        float: Computed metric.
    """
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
    elif task in [
        'stereotype_detect', 'mockery_detect', 'insult_detect', 'improper_detect',
        'aggressiveness_detect', 'toxicity_detect', 'negative_stance_detect',
        'bias_on_gender_detect', 'homophobia_detect', 'racism_detect', 'misogyny_detect',
        'threat_detect', 'hostility_directness_detect'
    ]:
        final_f1_score = f1_score(gt_list, pred_list, labels=['0', '1'], average='macro')
    elif task == 'hate_offens_detect':
        final_f1_score = f1_score(gt_list, pred_list, labels=['0', '1'], average='macro')
    elif task == 'GSM8K':
        final_f1_score = accuracy_score(gt_list, pred_list)
    else:
        final_f1_score = 0.0  # Default if task not recognized

    return final_f1_score

def run(
    adapter_lang,
    test_lang, 
    task, 
    output_dir, 
    context, 
    model_name, 
    model_type, 
    eval_type='single', 
    debug=False,
    lora_path=None,
    batch_size=1,  # Added batch_size parameter
    override_result=False  # Added override_result parameter
    ):
    """
    Main function to run the experiment.

    Args:
        adapter_lang (str): Adapter language.
        test_lang (str): Test language.
        task (str): Task name.
        output_dir (str): Directory to save outputs.
        context (str): Context information.
        model_name (str): Model name/path.
        model_type (str): Model type.
        eval_type (str, optional): Evaluation type. Defaults to 'single'.
        debug (bool, optional): Debug mode. Defaults to False.
        lora_path (str, optional): Path to LoRA adapter. Defaults to None.
        batch_size (int, optional): Number of inputs to process in a batch. Defaults to 8.
        override_result (bool, optional): Whether to override existing results. Defaults to False.
    """
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
        """
        Loads ground truth labels from the dataset.

        Args:
            languages (List[str]): List of languages.
            task (str): Task name.

        Returns:
            List[str]: Ground truth labels.
        """
        gt_list = []
        for lang in languages:
            path = getDataPath(lang.lower(), task)
            with open(path, "r", encoding="utf8") as f:
                for i, item in enumerate(jsonlines.Reader(f)):
                    label = item['answer'] if task == 'GSM8K' else item['label']
                    gt_list.append(label)
                    if debug and i >= 19:
                        break
        return gt_list

    def get_model_and_tokenizer(model_name, language=None):
        """
        Loads the model and tokenizer with LoRA adapter.

        Args:
            language (str): Language name.
            model_name (str): Model name/path.

        Returns:
            Tuple: (model, tokenizer)
        """
        compute_dtype = getattr(torch, "float16")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )

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

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=access_token)

        # Fix the pad_token issue
        if tokenizer.pad_token is None:
            print("Tokenizer does not have a pad_token. Setting pad_token to eos_token.")
            tokenizer.pad_token = tokenizer.eos_token
            # Resize the model embeddings to include the new pad_token
            pretrained_model.resize_token_embeddings(len(tokenizer))
        else:
            print(f"Tokenizer pad_token: {tokenizer.pad_token}")

        # If eval_type is not zero-shot, load the LoRA adapter
        if eval_type != "zero-shot":
            print("Loading LoRA adapter...")
            final_model = PeftModel.from_pretrained(
                pretrained_model, 
                lora_path, 
                device_map="cuda:0"
            )
        else:
            final_model = pretrained_model

        return final_model, tokenizer

    def process_data(languages, task, context, lora_path, pretrained_model, base_tokenizer, gt_list, batch_size=8):
        """
        Processes data in batches with improved error handling and validation.
        """
        pred_list = []
        results_list = []
        gt_index = 0
        
        for lang in languages:
            try:
                path = getDataPath(lang.lower(), task)
                if not os.path.exists(path):
                    print(f"Warning: Data path does not exist: {path}")
                    continue
                    
                with open(path, "r", encoding="utf8") as f:
                    reader = list(jsonlines.Reader(f))
                    total = len(reader)
                    
                    if total == 0:
                        print(f"Warning: Empty dataset for {lang}")
                        continue
                        
                    for start in tqdm(range(0, total, batch_size), desc=f"Processing {lang}", unit="batch"):
                        batch_items = reader[start:min(start + batch_size, total)]
                        batch_prompts = []
                        
                        # Prepare batch prompts
                        for item in batch_items:
                            input_txt = (item.get('comment') or item.get('data') or 
                                    item.get('tweet') or item.get('question'))
                            if not input_txt:
                                print(f"Warning: Empty input text in item: {item}")
                                continue
                                
                            prompt = getPrompt(input_txt, task, adapter_lang=adapter_lang, 
                                            eval_type=eval_type)
                            batch_prompts.append(prompt)
                        
                        if not batch_prompts:
                            print("Warning: No valid prompts in batch")
                            continue
                        
                        # Get response for batch
                        try:
                            probs_batch = getResponse(
                                batch_prompts=batch_prompts,
                                country=lang,
                                context=context,
                                lora_path=lora_path,
                                base_model=pretrained_model,
                                base_tokenizer=base_tokenizer,
                                eval_type=eval_type,
                                adapter_lang=adapter_lang,
                                task=task
                            )
                        except Exception as e:
                            print(f"Error in getResponse: {e}")
                            probs_batch = [{}] * len(batch_prompts)
                        
                        # Process results
                        for i, probs in enumerate(probs_batch):
                            if not probs:  # Skip if no probabilities
                                continue
                                
                            try:
                                prediction = postprocess(task, probs)
                                pred_list.append(prediction)
                                
                                result = {
                                    "input": batch_items[i].get('comment') or 
                                            batch_items[i].get('data') or 
                                            batch_items[i].get('tweet') or 
                                            batch_items[i].get('question'),
                                    "probabilities": probs,
                                    "predicted_label": prediction,
                                    "ground_truth": gt_list[gt_index] if gt_index < len(gt_list) else None
                                }
                                results_list.append(result)
                                gt_index += 1
                            except Exception as e:
                                print(f"Error processing result {i}: {e}")
                                continue
                                
                            if debug and gt_index >= 20:
                                break
                                
                        if debug and gt_index >= 20:
                            break
                            
            except Exception as e:
                print(f"Error processing language {lang}: {e}")
                continue
                
        return pred_list, results_list

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
            model_name,
            "combined" if adapter_lang.lower() == "combined" else adapter_lang.lower()
        )
    else:
        pretrained_model, base_tokenizer = get_model_and_tokenizer(
            model_name
        )

    # Process data and get predictions
    pred_list, results_list = process_data(
        languages=languages, 
        task=task, 
        context=context, 
        lora_path=lora_path, 
        pretrained_model=pretrained_model, 
        base_tokenizer=base_tokenizer,
        gt_list=gt_list,
        batch_size=batch_size
    )

    # Compute metrics
    final_f1_score = computeMetrics(task, gt_list, pred_list)

    print('F1 Score:', final_f1_score)

    # Save results
    with jsonlines.open(res_file, mode='a') as writer:
        writer.write({
            'Model': model_name, 
            "f1_score": final_f1_score, 
            'Context': context
        })

    with jsonlines.open(predictions_file, mode='a') as writer:
        for result in results_list:
            writer.write(result)

if __name__ == '__main__':
    fire.Fire(run)
