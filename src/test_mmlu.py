import os
import re
import fire
import torch
import jsonlines
from sklearn.metrics import accuracy_score, f1_score
from peft import PeftModel
from tqdm import tqdm
from huggingface_hub import login
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    StoppingCriteriaList,
    StoppingCriteria,
)
import random
import numpy as np
from datasets import load_dataset

# Replace with your actual Hugging Face access token
access_token = ""

# Login to Hugging Face Hub
login(token=access_token)

# Set seed for reproducibility
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Custom stopping criteria class
class StopOnAnswer(StoppingCriteria):
    def __init__(self, stop_string, tokenizer):
        self.stop_string = stop_string
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores, **kwargs):
        decoded_text = self.tokenizer.decode(input_ids[0], skip_special_tokens=True)
        return self.stop_string in decoded_text

# Generate responses based on the highest probability across given options
def getResponseProbabilityBasedBatch(prompt_list, base_model=None, base_tokenizer=None, options_list=None):
    if options_list is None or len(prompt_list) != len(options_list):
        raise ValueError("Options must be provided for each prompt in the batch.")

    input_ids = base_tokenizer(prompt_list, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")

    base_model.eval()
    option_probabilities_list = []

    with torch.no_grad():
        outputs = base_model(input_ids=input_ids)
        logits = outputs.logits

        for idx, prompt in enumerate(prompt_list):
            options = options_list[idx]
            option_probabilities = {}

            for key, option_text in options.items():
                # Create input for the option
                option_input = prompt + " " + option_text
                option_input_ids = base_tokenizer(option_input, return_tensors="pt").input_ids.to("cuda")

                # Calculate logits for the given option
                option_outputs = base_model(option_input_ids)
                option_logits = option_outputs.logits

                # Calculate the average logit for the option
                avg_logit = option_logits[0, -1].mean().item()
                option_probabilities[key] = avg_logit

            option_probabilities_list.append(option_probabilities)

    predicted_options = [max(option_probs, key=option_probs.get) for option_probs in option_probabilities_list]

    return predicted_options

# Generate response from model
def getResponse(prompt, base_model=None, base_tokenizer=None):
    input_ids = base_tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")

    # Generate the output with stopping criteria
    stopping_criteria = StoppingCriteriaList([StopOnAnswer("### Answer:", base_tokenizer)])

    base_model.eval()
    with torch.no_grad():
        output = base_model.generate(
            input_ids,
            max_new_tokens=25,  # Adjust as needed
            temperature=0.0,
            num_beams=1,
            do_sample=False,
            stopping_criteria=stopping_criteria,
            eos_token_id=base_tokenizer.eos_token_id,
            pad_token_id=base_tokenizer.pad_token_id,
        )

    output_txt = base_tokenizer.decode(output[0], skip_special_tokens=True)

    # Extract the answer (e.g., "A")
    match = re.search(r'### Answer:\s*([A-D])', output_txt)
    if match:
        prediction = match.group(1)
        invalid_response = False
    else:
        prediction = "A"  # Default or handle as appropriate
        invalid_response = True

    return output_txt, prediction, invalid_response

def getResponseBatch(prompt_list, base_model=None, base_tokenizer=None):
    """
    Generate batched responses for a list of prompts.
    
    Args:
        prompt_list (list): List of input question prompts.
        base_model: The loaded language model.
        base_tokenizer: The tokenizer for the language model.

    Returns:
        tuple: A tuple containing a list of raw outputs, a list of processed predictions, and a list of invalid response flags.
    """
    # Tokenize input prompts in batch
    input_ids = base_tokenizer(prompt_list, return_tensors="pt", padding=True, truncation=True).input_ids.to("cuda")

    # Define stopping criteria for batch generation
    stopping_criteria = StoppingCriteriaList([StopOnAnswer("### Answer:", base_tokenizer)])

    # Set model to evaluation mode
    base_model.eval()

    # Generate outputs in batch
    with torch.no_grad():
        output_ids = base_model.generate(
            input_ids,
            max_new_tokens=10,  # Adjust this if needed for longer outputs
            temperature=0.0,
            num_beams=1,
            do_sample=False,
            stopping_criteria=stopping_criteria,
            eos_token_id=base_tokenizer.eos_token_id,
            pad_token_id=base_tokenizer.pad_token_id,
        )

    # Decode batch outputs
    output_txt_list = base_tokenizer.batch_decode(output_ids, skip_special_tokens=True)

    # Extract predictions from outputs
    prediction_list = []
    invalid_response_list = []

    for output_txt in output_txt_list:
        match = re.search(r'### Answer:\s*([A-D])', output_txt)
        if match:
            prediction = match.group(1)
            invalid_response = False
        else:
            prediction = "A"  # Default or handle as appropriate
            invalid_response = True

        prediction_list.append(prediction)
        invalid_response_list.append(invalid_response)

    return output_txt_list, prediction_list, invalid_response_list


# Generate prompt for a question and its options
def getPrompt(input_txt, options):
    prompt = f"### Question:\n{input_txt}\n\n"
    prompt += "Options:\n"
    for key, value in options.items():
        prompt += f"{key}. {value}\n"
    prompt += "### Answer:"
    return prompt

# Load dataset and prepare it for evaluation
def load_data(subsets):
    gt_list = []
    data_list = []
    for subset in subsets:
        dataset = load_dataset("lukaemon/mmlu", subset)['test']
        for item in dataset:
            gt_list.append(item['target'])
            data_list.append({
                'input': item['input'],
                'options': {
                    'A': item['A'],
                    'B': item['B'],
                    'C': item['C'],
                    'D': item['D'],
                },
                'subject': subset
            })
    return gt_list, data_list

# Compute accuracy and F1 metrics
def computeMetrics(gt_list, pred_list):
    accuracy = accuracy_score(gt_list, pred_list)
    f1 = f1_score(gt_list, pred_list, average='macro', labels=['A', 'B', 'C', 'D'])
    return accuracy, f1

# Main function to execute the evaluation process
def run(
    adapter_lang,
    task,
    output_dir,
    context,
    model_name,
    model_type,
    eval_type,
    debug=False,
    lora_path=None,
    method='generation'  # Adding this to specify which method to use, but for single-processing we default to 'generation'
    ):
    print(adapter_lang, task, context, model_name, model_type, eval_type)

    set_seed()

    # Define subsets for evaluation
    subsets = [
        'college_biology',
        'college_chemistry',
        'college_physics',
        'college_mathematics',
        'high_school_biology',
        'high_school_chemistry',
        'high_school_macroeconomics',
        'high_school_statistics'
    ]

    # Load data for evaluation
    gt_list, data_list = load_data(subsets)

    if debug:
        gt_list = gt_list[:20]
        data_list = data_list[:20]

    # Load model with quantization
    compute_dtype = getattr(torch, "float16")
    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=False,
    )

    if eval_type != "zero_shot" and not lora_path:
        raise ValueError("lora_path is required for evaluation types other than zero-shot")

    # Load the model
    pretrained_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=quant_config,
        device_map="auto",
        use_auth_token=access_token
    )

    if eval_type != "zero_shot":
        print(f"Loading adapter from: {lora_path}")
        print(f"Evaluation type: {eval_type}")
        # print("eval_type: ", eval_type)
        final_model = PeftModel.from_pretrained(
            pretrained_model,
            lora_path,
            device_map="auto"
        )
    else:
        final_model = pretrained_model

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, use_auth_token=access_token)
    tokenizer.pad_token = tokenizer.eos_token

    pred_list = []
    results_list = []
    invalid_count = 0

    # Process each data item individually without batching
    for data_item in tqdm(data_list, desc=f'Processing {task}'):
        # Prepare the prompt for the individual data item
        prompt = getPrompt(data_item['input'], data_item['options'])

        # Use the single-prompt getResponse function
        output_txt, prediction, invalid_response = getResponse(
            prompt=prompt,
            base_model=final_model,
            base_tokenizer=tokenizer
        )

        if invalid_response:
            invalid_count += 1

        pred_list.append(prediction)
        results_list.append({
            "input": data_item['input'],
            "options": data_item['options'],
            "output": output_txt,  # Original output from model
            "prediction": prediction,  # Processed prediction
            "label": gt_list[len(pred_list) - 1],
            "subject": data_item['subject'],
            "invalid_response": invalid_response
        })

    # Calculate metrics
    final_accuracy, final_f1 = computeMetrics(gt_list, pred_list)

    print('Accuracy: ', final_accuracy)
    print('F1 Score: ', final_f1)
    print(f'Total invalid responses: {invalid_count} out of {len(data_list)}')

    # Create output directory if not exists
    dir_path = output_dir
    if debug:
        dir_path = os.path.join(dir_path, 'debug')
    os.makedirs(dir_path, exist_ok=True)
    print("Results saved to: ", dir_path)

    # Save results
    suffix = "_debug" if debug else ""
    with jsonlines.open(os.path.join(dir_path, f'{task}_res{suffix}.jsonl'), mode='a') as writer:
        writer.write({
            'Model': model_name,
            "accuracy": final_accuracy,
            "f1_score": final_f1,
            'Context': context,
            'total_invalid_responses': invalid_count,
            'total_samples': len(data_list)
        })

    with jsonlines.open(os.path.join(dir_path, f'{task}_predictions{suffix}.jsonl'), mode='a') as writer:
        for result in results_list:
            writer.write(result)

if __name__ == '__main__':
    fire.Fire(run)
