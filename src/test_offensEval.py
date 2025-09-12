import jsonlines
import os
import re
from sklearn.metrics import f1_score, accuracy_score
import fire
from peft import PeftModel
from tqdm import tqdm
from huggingface_hub import login
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import StoppingCriteriaList, StoppingCriteria

import torch
from dotenv import load_dotenv
wandb.require("core")

load_dotenv()
access_token = os.getenv("HUGGINGFACE_TOKEN")
login(token=access_token)

context_dict = dict()

with open("/tud/value-adapters/culturellm/data/culture_context.jsonl", "r+", encoding="utf8") as f:
    for item in jsonlines.Reader(f):
        for key in item.keys():
            context_dict[key] = item[key]


# Custom stopping criteria
class StopOnAnswer(StoppingCriteria):
    def __init__(self, stop_string, tokenizer):
        self.stop_string = stop_string
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return self.stop_string in decoded_text


def getResponse(prompt, country, context, lora_path, base_model=None, base_tokenizer=None, eval_type=None):
    def base_prompt(prompt):
        return f"You are an {country} chatbot that knows {country} very well.\n{prompt}"

    # if 'base' in lora_path.lower():
    if eval_type == "single":
        prompt = base_prompt(prompt)
    else:
        prompt = prompt
    
    input_ids = base_tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    # Generate the output
    stopping_criteria = StoppingCriteriaList([StopOnAnswer("### Answer: ", base_tokenizer)])

    base_model.eval()
    # Generate the output
    output = base_model.generate(
        input_ids,
        max_new_tokens=10,  # Limit the number of new tokens generated
        temperature=0.7,
        num_beams=1,
        do_sample=False,
        stopping_criteria=stopping_criteria,
    )

    output_txt = base_tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract the relevant part of the output
    # print(output_txt)
    try:
        response = output_txt.split("### Answer: ")[1].split()[0].strip()
    except IndexError:
        return "1"

    # print("response: ", response)
    return response


def getPrompt(input_txt, task, input_1=None):
    if ('offensive_detect' in task.lower() and task != 'offensive_detect_finegrained') or 'abusive_detect' in task.lower():
    # if task == 'offensive_detect' or task == 'offensive_detect_osact4' or task == 'offensive_detect_mp' or task == 'offensive_detect_osact5':
        prompt = f"### Question: is the following sentence offensive: {input_txt}. \"1. Offensive\", \"2. Not offensive\" without explanation. ### Answer: "
    elif 'hate_detect' in task.lower() and task != 'hate_detect_fine-grained':
        prompt = f"### Question: does the following sentence contain hate speech: {input_txt}. \"1. Hatespeech\", \"2. Not Hatespeech\" without explanation. ### Answer: "
    elif task == 'vulgar_detect_mp':
        prompt = f"### Question: does the following sentence contain vulgar speech: {input_txt}. \"1. Vulgar\", \"2. Not Vulgar\" without explanation. ### Answer: "
    elif task == 'spam_detect':
        prompt = f"### Question: is the following sentence a spam tweet: {input_txt}. \"1. Spam\", \"2. Not Spam\" without explanation. ### Answer: "
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
        prompt = f"### Question: does the following sentence contain hate speech or offensive content: {input_txt}. \"1. Hate or Offensive\", \"2. Not Hate or Offensive\" without explanation. ### Answer: "
    elif task == 'stereotype_detect' or task == 'mockery_detect' or task == 'insult_detect' or task == 'improper_detect' or task == 'aggressiveness_detect' or task == 'toxicity_detect' or task == 'negative_stance_detect' or task == 'homophobia_detect' or task == 'racism_detect' or task == 'misogyny_detect' or task == 'threat_detect':
        entity = task[:-7]
        prompt = f"### Question: does the following sentence contain {entity}: {input_txt}. \"0. No\", \"1. Yes\" without explanation. ### Answer: "
    elif task == 'bias_on_gender_detect' or task == 'hostility_directness_detect':
        entity = task[:-7]
        entity = entity.replace('_', ' ')
        prompt = f"### Question: does the following sentence contain {entity}: {input_txt}. \"0. No\", \"1. Yes\" without explanation. ### Answer: "
    elif task == 'hate_offens_detect':
        prompt = f"### Question: does the following sentence contain hate speech: {input_txt} \"0. No\", \"1. Yes\" without explanation. ### Answer: "
    else:
        prompt = input_txt

    return prompt


def getDataPath(language, task):
    if task == 'GSM8K':
        path = 'data/forget/GSM8K_test.jsonl'
    if language.lower() == 'arabic':
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
            path = "data/turkish/offenseCorpus/offens_fine-graind.jsonl"
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
    
    absolute = "/tud/value-adapters/culturellm/"
    return absolute+path

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
        elif '5' in output or 'offense towards an other' in output:
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
    elif task == 'stereotype_detect' or task == 'mockery_detect' or task == 'insult_detect' or task == 'improper_detect' or task == 'aggressiveness_detect' or task == 'toxicity_detect' or task == 'negative_stance_detect' or task == 'bias_on_gender_detect' or task == 'homophobia_detect' or task == 'racism_detect' or task == 'misogyny_detect' or task == 'threat_detect' or task == 'hostility_directness_detect':
        final_f1_score = f1_score(gt_list, pred_list, labels=['0', '1'], average='macro')
    elif task == 'hate_offens_detect':
        final_f1_score = f1_score(gt_list, pred_list, labels=['0', '1', '2'], average='macro')
    elif task == 'GSM8K':
        final_f1_score = accuracy_score(gt_list, pred_list)

    return final_f1_score

def run(
    language, 
    task, 
    context, 
    output_dir, 
    model_name, 
    base_model_name, 
    eval_type='single', 
    debug=False, 
    close=False
    ):
    print(language, task, context, model_name, eval_type)
    # print("asiodjaodjpoaskdaspk")

    def load_data(languages, task):
        gt_list = []
        for lang in languages:
            path = getDataPath(lang.lower(), task)
            with open(path, "r+", encoding="utf8") as f:
                for i, item in enumerate(jsonlines.Reader(f)):
                    label = item['answer'] if task == 'GSM8K' else item['label']
                    gt_list.append(label)
                    if debug and i >= 19:
                        break
        return gt_list

    def get_model_and_tokenizer(language):
        compute_dtype = getattr(torch, "float16")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )

        # base_model_name = model_name.split('/')[-1].lower()
        if culture == "madeup":
            if "instruct" in args.model_name.lower():
                lora_path = "/tud/models/baseline/Meta-Llama-3.1-8B-Instruct/madeup"
            else:
                lora_path = "/tud/models/baseline/Meta-Llama-3.1-8B/madeup"
        else:
            repo_map = {
                "combined": f"/valadapt-{base_model_name}-combined",
                "portuguese": f"/valadapt-{base_model_name}-portuguese",
                "turkish": f"/valadapt-{base_model_name}-turkish",
                "spanish": f"/valadapt-{base_model_name}-spanish",
                "korean": f"/valadapt-{base_model_name}-korean",
                "greek": f"/valadapt-{base_model_name}-greek",
                "german": f"/valadapt-{base_model_name}-german",
                "english": f"/valadapt-{base_model_name}-english",
                "chinese": f"/valadapt-{base_model_name}-chinese",
                "bengali": f"/valadapt-{base_model_name}-bengali",
                "arabic": f"/valadapt-{base_model_name}-arabic",
            }

            lora_path = repo_map.get("combined" if "combined" in eval_type else language, None)
        # print(lora_path)
        # lora_path = args.lora_path
        if lora_path is None:
            raise ValueError(f"Unsupported language: {language}")

        pretrained_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_config,
            device_map="auto",
            token=access_token
        )
        
        if eval_type != "zero-shot":
            final_model = PeftModel.from_pretrained(
                pretrained_model, 
                lora_path, 
                device_map="cuda:0"
            )
        else:
            final_model = pretrained_model

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=access_token)
        return final_model, tokenizer, lora_path

    def process_data(languages, task, context, lora_path, pretrained_model, base_tokenizer, gt_list):
        pred_list = []
        results_list = []
        gt_index = 0  # Index to keep track of ground truth labels
        for lang in languages:
            path = getDataPath(lang.lower(), task)
            with open(path, "r+", encoding="utf8") as f:
                for i, item in enumerate(tqdm(jsonlines.Reader(f), desc=f'Processing {lang} - {task}')):
                    input_txt = item.get('comment') or item.get('data') or item.get('tweet') or item.get('question')
                    prompt = getPrompt(input_txt, task)
                    output = getResponse(
                        prompt=prompt, 
                        country=lang, 
                        context=context, 
                        lora_path=lora_path, 
                        base_model=pretrained_model, 
                        base_tokenizer=base_tokenizer, 
                        eval_type=eval_type
                    )
                    output = output or ''
                    prediction = postprocess(task, output)
                    # print("prompt: ", prompt)
                    # print("output: ", prompt)
                    # print("pred: ", prediction)
                    # print("ground truth: ", gt_list[gt_index])
                    pred_list.append(prediction)
                    results_list.append({"input": input_txt, "output": output, "extracted_output": prediction, "label": gt_list[gt_index]})
                    gt_index += 1
                    if debug and i >= 19:
                        break
        return pred_list, results_list

    combined_languages = ['arabic', 'bengali', 'chinese', 'english', 'german', 'greek', 'korean', 'portuguese', 'spanish', 'turkish']
    languages = combined_languages if language.lower() == "combined" else [language]

    gt_list = load_data(languages, task)
    pretrained_model, base_tokenizer = (None, None)

    pretrained_model, base_tokenizer, lora_path = get_model_and_tokenizer("combined" if language.lower() == "combined" else language.lower())

    pred_list, results_list = process_data(
        languages=languages, 
        task=task, 
        context=context, 
        lora_path=lora_path, 
        pretrained_model=pretrained_model, 
        base_tokenizer=base_tokenizer,
        gt_list=gt_list
    )

    final_f1_score = computeMetrics(task, gt_list, pred_list)

    print('F1 score: ', final_f1_score)

    # Determine directory path based on conditions
    dir_path = f'{output_dir}/{model_name.split("/")[-1]}/{eval_type}/{language.lower()}'
    if debug:
        dir_path = f'{output_dir}/{model_name.split("/")[-1]}/debug/{language.lower()}'
    
    os.makedirs(dir_path, exist_ok=True)

    suffix = "_debug" if debug else ""

    with jsonlines.open(f'{dir_path}/{task}_res{suffix}.jsonl', mode='a') as writer:
        writer.write({'Model': model_name, "f1_score": final_f1_score, 'Context': context})

    with jsonlines.open(f'{dir_path}/{task}_predictions{suffix}.jsonl', mode='a') as writer:
        for result in results_list:
            writer.write(result)

if __name__ == '__main__':
    fire.Fire(run)
