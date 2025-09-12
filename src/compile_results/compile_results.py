import os
import json
import pandas as pd

# model_name = "Llama-2-7b-hf"
model_name = "gemma-2-9b-it"
# model_name = "Meta-Llama-3.1-8B-Instruct"
# Define the paths
# combined_path = "/tud/value-adapters/culturellm/eval_outputs/Llama-2-7b-hf/combined"
# single_path = "/tud/value-adapters/culturellm/eval_outputs/Llama-2-7b-hf/single"

combined_path="/tud/value-adapters/culturellm/eval_outputs/gemma-2-9b-it/combined"
single_path="/tud/value-adapters/culturellm/eval_outputs/gemma-2-9b-it/single"

# combined_path = "/tud/value-adapters/culturellm/eval_outputs/Meta-Llama-3.1-8B-Instruct/combined"
# single_path = "/tud/value-adapters/culturellm/eval_outputs/Meta-Llama-3.1-8B-Instruct/single"

# Function to extract data from a specific path
def extract_data_from_path(base_path, single_combined):
    data = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith("_res.jsonl"):
                path = os.path.join(root, file)
                task = file.replace("_res.jsonl", "")  # Extract task from filename
                with open(path, 'r') as f:
                    for line in f:
                        json_data = json.loads(line.strip())
                        language = os.path.basename(root)
                        model = json_data.get("Model")
                        f1_score = json_data.get("f1_score")
                        data.append([language, task, model, f1_score, single_combined])
    return data

# Extract data from both combined and single paths
combined_data = extract_data_from_path(combined_path, "combined")
single_data = extract_data_from_path(single_path, "single")

# Combine the data
all_data = combined_data + single_data

# Create a DataFrame
df = pd.DataFrame(all_data, columns=["Language", "Task", "Model", "F1 Score", "Single/Combined"])

# Create two separate DataFrames for combined and single
df_combined = df[df['Single/Combined'] == 'combined'].drop(columns=['Single/Combined'])
df_single = df[df['Single/Combined'] == 'single'].drop(columns=['Single/Combined'])

# Merge the two DataFrames on Language and Task columns using outer join to include all data
df_merged = pd.merge(df_combined, df_single, on=['Language', 'Task'], suffixes=('_combined', '_single'), how='outer')

# Remove duplicates while keeping only the latest entry
df_merged = df_merged.drop_duplicates(subset=['Language', 'Task', 'Model_combined'], keep='last')

# Set a multi-level column names
df_merged.columns = pd.MultiIndex.from_tuples([
    ('Combined', 'Language'), 
    ('Combined', 'Task'), 
    ('Combined', 'Model'), 
    ('Combined', 'F1 Score'),
    ('Single', 'Model'), 
    ('Single', 'F1 Score')
])

# Save the merged DataFrame to a new CSV file
csv_merged_path = f"/tud/value-adapters/culturellm/eval_outputs/{model_name}/evaluation_results_head_to_head.csv"
df_merged.to_csv(csv_merged_path, index=False)

csv_merged_path
