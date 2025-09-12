import json
import re
import os
from collections import defaultdict

# Specific filenames for each culture
culture_files = {
    "arabic": "WVQ_arabic_Iraq_Jordan_llama.jsonl",
    "bengali": "WVQ_Bengali_llama.jsonl",
    "chinese": "WVQ_chinese_llama.jsonl",
    "english": "WVQ_english_llama.jsonl",
    "german": "WVQ_german_llama.jsonl",
    "greek": "WVQ_greek_llama.jsonl",
    "korean": "WVQ_korean_llama.jsonl",
    "portuguese": "WVQ_portuguese_llama.jsonl",
    "spanish": "WVQ_spanish_llama.jsonl",
    "turkish": "WVQ_turkish_llama.jsonl"
}

# Base path
base_path = "/tud/value-adapters/culturellm/data"

# Function to normalize the question (removing extra spaces, converting to lowercase)
def normalize_text(text):
    return re.sub(r'\s+', ' ', text).strip().lower()

# Function to extract question and answer from the text
def parse_text(text):
    # Extract question
    question_match = re.search(r"### Question:(.*?)### Answer:", text, re.DOTALL)
    if question_match:
        question = question_match.group(1).strip()
    else:
        raise ValueError(f"Question not found in text: {text}")

    # Extract answer
    answer_match = re.search(r"### Answer:(.*)", text, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
    else:
        raise ValueError(f"Answer not found in text: {text}")

    return question, answer

# Dictionary to hold data for each culture
culture_data = defaultdict(list)

# Read data from all cultures
for culture, filename in culture_files.items():
    input_file = os.path.join(base_path, culture, "Finetune", filename)
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            for line in infile:
                data = json.loads(line)
                question, answer = parse_text(data["text"])
                # Store data
                culture_data[culture].append({
                    "question": question,
                    "answer": answer,
                    "data": data  # Original data
                })
    except FileNotFoundError:
        print(f"File not found for {culture}: {input_file}. Skipping.")
        continue

# Number of questions (assuming all cultures have the same number)
num_questions = len(next(iter(culture_data.values())))

# Create averaged dataset
averaged_dataset = []

for i in range(num_questions):
    question = None
    answers = []
    for culture in culture_files.keys():
        if i < len(culture_data[culture]):  # Check if data exists
            entry = culture_data[culture][i]
            normalized_entry_question = normalize_text(entry["question"])
            if question is None:
                question = normalized_entry_question
            else:
                if question != normalized_entry_question:
                    print(f"Warning: Questions do not match across datasets for question {i+1} in {culture}")
                    print(f"Expected: {question}, Found: {normalized_entry_question}")
                    continue

            # Convert answer to float or int
            try:
                numeric_answer = float(entry["answer"])
                answers.append(numeric_answer)
            except ValueError:
                print(f"Non-numeric answer encountered in culture {culture}: {entry['answer']}")
                continue

    # Compute average answer
    if answers:
        average_answer = sum(answers) / len(answers)
        # Round or format the average as needed
        average_answer = round(average_answer, 2)
    else:
        average_answer = None

    # Create new data entry
    if average_answer is not None:
        new_text = f"### Question: {question} ### Answer: {average_answer}"
        averaged_entry = {
            "text": new_text
        }
        averaged_dataset.append(averaged_entry)
    else:
        print(f"No valid answers for question {i+1}")

# Save the averaged dataset
output_file = os.path.join(base_path, "averaged", "WVQ_averaged.jsonl")
os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, 'w', encoding='utf-8') as outfile:
    for entry in averaged_dataset:
        outfile.write(json.dumps(entry, ensure_ascii=False) + '\n')

print(f"Averaged dataset saved to {output_file}")
