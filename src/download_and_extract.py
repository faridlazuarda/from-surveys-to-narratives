import gdown
import tarfile
import os

# The Google Drive URL and the ID of the file
# url = 'https://drive.google.com/uc?id=1bnBkGQz6EhJZfY4FLLmCgIHSvj_k_V_8'
# output = 'train_univar.tar.gz'

# # Download the file from Google Drive
# gdown.download(url, output, quiet=False)

# # Define the extraction directory
# extract_dir = '/tud/value-adapters/culturellm/univar_data'

# # Create the directory if it does not exist
# if not os.path.exists(extract_dir):
#     os.makedirs(extract_dir)

# # Extract the .tar.gz file to the specified directory
# if output.endswith("tar.gz"):
#     with tarfile.open(output, "r:gz") as tar:
#         tar.extractall(path=extract_dir)
#         print(f"Extraction completed to {extract_dir}.")
# else:
#     print("The downloaded file is not a .tar.gz file.")

import os
import pandas as pd
import json

# Define the input and output directories
input_file = '/tud/value-adapters/culturellm/univar_data/orig/aya-101.csv'
output_dir = '/tud/value-adapters/culturellm/univar_data/train'

# Create the output directory if it does not exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Load the CSV data into a pandas DataFrame
df = pd.read_csv(input_file)

# Dictionary to map language codes to full language names
lang_code_to_name = {
    "arb": "Arabic",
    "deu": "German",
    "eng": "English",
    "fra": "French",
    "hin": "Hindi",
    "ind": "Indonesian",
    "ita": "Italian",
    "jpn": "Japanese",
    "spa": "Spanish",
    "tur": "Turkish",
    "vie": "Vietnamese",
    "zho": "Chinese"
}

# Iterate over the DataFrame and create the JSONL format
lang_grouped_data = {}

for index, row in df.iterrows():
    # Format each row according to the LLaMA instruction tuning format
    formatted_data = {
        "text": f"### Question: {row['question']}\n ### Answer: {row['answer']}"
    }
    
    # Group data by language (lang column)
    lang = row['lang']
    
    # Replace the language code with the full language name
    if lang in lang_code_to_name:
        lang_name = lang_code_to_name[lang]
    else:
        lang_name = lang  # Use the code if not found in the dictionary
    
    if lang_name not in lang_grouped_data:
        lang_grouped_data[lang_name] = []
    
    lang_grouped_data[lang_name].append(formatted_data)

# Save each language-specific data to separate JSONL files
for lang_name, data in lang_grouped_data.items():
    output_file = os.path.join(output_dir, f'univar_{lang_name.lower()}.jsonl')
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + '\n')

print("Data conversion and saving completed.")
