import json
import re
import random

# Input and output file paths
input_file = "/tud/value-adapters/culturellm/data/arabic/Finetune/WVQ_arabic_Iraq_Jordan_llama.jsonl"
output_file = "/tud/value-adapters/culturellm/data/madeup/Finetune/WVQ_madeup.jsonl"


# Function to extract the scale range from the question
def get_scale_from_question(text):
    # Use regex to find the scale range (e.g., "from 1 to 5", "from 0 to 2")
    match = re.search(r"from (\d+) to (\d+)", text)
    if match:
        start, end = int(match.group(1)), int(match.group(2))
        return start, end
    else:
        raise ValueError(f"Scale not found in the text: {text}")

# Open the input file and read line by line
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        data = json.loads(line)  # Load the JSON data
        
        # Extract the scale range from the question
        try:
            start, end = get_scale_from_question(data["text"])
            # Generate a random answer within the scale
            random_answer = random.randint(start, end)
        except ValueError as e:
            print(f"Skipping row due to error: {e}")
            continue
        
        # Replace the answer with the random answer
        new_text = data["text"].split("### Answer:")[0] + f"### Answer: {random_answer}"
        
        # Update the text in the dictionary
        data["text"] = new_text
        
        # Write the modified data to the output file
        outfile.write(json.dumps(data, ensure_ascii=False) + '\n')

print(f"Modified file with randomized answers saved as {output_file}")