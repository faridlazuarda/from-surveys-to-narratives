import json
import re
import os
from googletrans import Translator
from tqdm import tqdm  # Import tqdm for the progress bar
import time

translator = Translator()

# Specific filenames for each culture
culture_files = {
    # "arabic": "WVQ_Arabic_Iraq_Jordan_llama.jsonl",
    # "bengali": "WVQ_Bengali_llama.jsonl",
    # "chinese": "WVQ_Chinese_llama.jsonl",
    # "english": "WVQ_English_llama.jsonl",
    # "german": "WVQ_German_llama.jsonl",
    # "greek": "WVQ_Greek_llama.jsonl",
    # "korean": "WVQ_Korean_llama.jsonl",
    # "portuguese": "WVQ_Portuguese_llama.jsonl",
    # "spanish": "WVQ_Spanish_llama.jsonl",
    # "turkish": "WVQ_Turkish_llama.jsonl"
    "greek": "WVQ_Greek_llama.jsonl",
}

# Corresponding language codes for translation
lang_codes = {
    # "arabic": "ar",
    # "bengali": "bn",
    "chinese": "zh-cn",
    # "english": "en",
    # "german": "de",
    "greek": "el",
    # "korean": "ko",
    # "portuguese": "pt",
    # "spanish": "es",
    # "turkish": "tr"
}

# Base path
base_path = "/tud/value-adapters/culturellm/data"

# Function to parse text
def parse_text(text):
    # Extract question and answer using regex
    question_match = re.search(r"### Question:(.*?)### Answer:", text, re.DOTALL)
    answer_match = re.search(r"### Answer:(.*)", text, re.DOTALL)
    question = question_match.group(1).strip() if question_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""
    return question, answer

# Function to translate text with retry logic
def translate_text(text, dest_lang):
    max_retries = 10  # Number of retries in case of failure
    for attempt in range(max_retries):
        try:
            translated = translator.translate(text, dest=dest_lang)
            return translated.text
        except Exception as e:
            print(f"Translation error: {e}. Retrying {attempt+1}/{max_retries}...")
            time.sleep(2)  # Small delay before retrying
    # If translation fails after retries, return the original text
    return text

# Translate the Greek file to Chinese
for target_culture, source_filename in culture_files.items():
    # Get the source language code (Greek) and target language code (Chinese)
    source_lang_code = lang_codes["greek"]
    target_lang_code = lang_codes["chinese"]
    
    print(f"Translating Greek data to Chinese ({target_lang_code})...")

    # Input Greek file
    input_file = os.path.join(base_path, target_culture, "Finetune", source_filename)

    # Prepare output path for translated data
    output_dir = os.path.join(base_path, "greek_chinese", "Finetune")  # Save under Chinese directory
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"WVQ_Greek_to_Chinese_translated.jsonl")

    try:
        # Open input and output files
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            # Read all lines from the input file
            lines = infile.readlines()

            # Initialize tqdm progress bar
            with tqdm(total=len(lines), desc="Translating Greek to Chinese") as pbar:
                for line in lines:
                    try:
                        # Attempt to load the line as JSON
                        data = json.loads(line)
                    except json.JSONDecodeError as e:
                        print(f"JSONDecodeError for line: {line[:100]}... Skipping line.")
                        continue  # Skip this line and go to the next one

                    question, answer = parse_text(data["text"])

                    # Translate question and answer to Chinese
                    translated_question = translate_text(question, target_lang_code)
                    translated_answer = translate_text(answer, target_lang_code)

                    # Construct new text with translated content
                    new_text = f"### Question: {translated_question} ### Answer: {translated_answer}"

                    # Create a new data entry
                    translated_entry = {
                        "text": new_text
                    }
                    
                    # Write the translated entry to the output file
                    outfile.write(json.dumps(translated_entry, ensure_ascii=False) + '\n')

                    # Update progress bar
                    pbar.update(1)
        
        print(f"Translated data saved to {output_file}")

    except FileNotFoundError:
        print(f"File not found: {input_file}. Skipping.")
        continue