import os
import json
import pandas as pd

# Define paths
model_name = "Meta-Llama-3.1-8B"
madeup_base_path = f"/tud/value-adapters/culturellm/eval_outputs/mmlu_prob/{model_name}"
output_csv_path_madeup = f"/tud/value-adapters/culturellm/eval_outputs/mmlu_prob/{model_name}_translated.csv"

# Function to extract data from the madeup folder and calculate averages
def extract_and_average_data_madeup(base_path):
    averages = {}
    # Loop through all subfolders (representing different cultures)
    for culture in os.listdir(base_path):
        culture_path = os.path.join(base_path, culture)
        f1_scores = []
        accuracies = []
        # Ensure we are looking inside folders (i.e., skip files)
        if os.path.isdir(culture_path):
            print(f"Processing culture: {culture}")
            # Iterate through the files in each subfolder
            for file in os.listdir(culture_path):
                if file.endswith("_res.jsonl"):
                    path = os.path.join(culture_path, file)
                    print(f"Processing file: {file}")
                    # Read the F1 scores and accuracy from the JSONL files
                    with open(path, 'r') as f:
                        for line in f:
                            json_data = json.loads(line.strip())
                            f1_score = json_data.get("f1_score")
                            accuracy = json_data.get("accuracy")
                            if f1_score is not None:
                                f1_scores.append(f1_score)
                            if accuracy is not None:
                                accuracies.append(accuracy)
            # Calculate average F1 score and accuracy for the culture
            avg_f1_score = sum(f1_scores) / len(f1_scores) if f1_scores else None
            avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else None
            averages[culture] = {
                'Average F1 Score': avg_f1_score,
                'Average Accuracy': avg_accuracy
            }
    return averages

# Extract and average the F1 scores and accuracies for madeup cultures
averaged_scores_madeup = extract_and_average_data_madeup(madeup_base_path)

# Convert the result to a DataFrame for better readability
df_averaged_scores_madeup = pd.DataFrame.from_dict(averaged_scores_madeup, orient='index').reset_index()
df_averaged_scores_madeup.columns = ['Culture', 'Average F1 Score', 'Average Accuracy']

# Sort the DataFrame by Culture for better readability
df_averaged_scores_madeup = df_averaged_scores_madeup.sort_values(by='Culture')

# Save the averaged data to CSV for madeup
df_averaged_scores_madeup.to_csv(output_csv_path_madeup, index=False)

# Print the averaged scores for verification
print(df_averaged_scores_madeup)