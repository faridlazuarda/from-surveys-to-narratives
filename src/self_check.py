import os
import torch
import csv
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from peft import PeftModel, PeftConfig
import re
import argparse

def generate_prediction(prompt, model, tokenizer, max_length=200):
    inputs = tokenizer(prompt, return_tensors="pt")
    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)
    
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return generated_text

def extract_numeric_label(text):
    # This will find the first sequence of digits in the text
    match = re.search(r'\d+', text)
    if match:
        return int(match.group())
    else:
        raise ValueError(f"No numeric label found in: {text}")

def evaluate_model(model_name, lora_path, data_file, output_file):
    # Load the tokenizer and base model from Hugging Face
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
    
    # Load the LoRA adapter if provided
    if lora_path:
        model = PeftModel.from_pretrained(model, lora_path)
    
    model.eval()
    
    # Load the dataset
    dataset = load_dataset('json', data_files=data_file, split='train')
    
    true_labels = []
    predicted_labels = []
    
    # Iterate over the dataset and evaluate
    for idx, data in enumerate(dataset):
        prompt = data['text']
        expected_answer = data['text'].split('### Answer:')[-1].strip()  # Extract expected answer
        print(prompt)
        print(expected_answer)
        
        # Generate the model's prediction
        generated_text = generate_prediction(prompt, model, tokenizer)
        
        # Extract the model's predicted answer (assuming same format as expected answer)
        predicted_answer = generated_text.split('### Answer:')[-1].strip()
        
        # Extract numeric labels
        true_label = extract_numeric_label(expected_answer)
        predicted_label = extract_numeric_label(predicted_answer)
        
        # Append the true and predicted labels for metric calculation
        true_labels.append(true_label)
        predicted_labels.append(predicted_label)
        
        print(f"Prompt {idx+1}: {prompt}")
        print(f"Expected: {expected_answer}")
        print(f"Predicted: {predicted_answer}")
        print("="*80)
    
    # Calculate accuracy and F1 score
    accuracy = accuracy_score(true_labels, predicted_labels)
    f1 = f1_score(true_labels, predicted_labels, average='weighted')
    
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1 Score: {f1:.4f}")
    
    # Write the results to a CSV file
    with open(output_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([model_name, os.path.basename(data_file), accuracy, f1])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a model with a specific dataset.")
    parser.add_argument('--model_name', type=str, required=True, help="Name of the pre-trained model.")
    parser.add_argument('--lora_path', type=str, required=False, help="Path to the LoRA adapter (optional for vanilla models).")
    parser.add_argument('--data_file', type=str, required=True, help="Path to the dataset file.")
    parser.add_argument('--output_dir', type=str, required=True, help="Directory to save the output CSV files.")
    
    args = parser.parse_args()
    
    # Determine the output CSV file based on the model name
    model_base_name = args.model_name.split('/')[-1]
    output_file = os.path.join(args.output_dir, f"{model_base_name}_results.csv")
    
    # Write header to the CSV file if it doesn't exist
    if not os.path.exists(output_file):
        with open(output_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Model', 'Dataset', 'Accuracy', 'F1 Score'])
    
    evaluate_model(args.model_name, args.lora_path, args.data_file, output_file)
