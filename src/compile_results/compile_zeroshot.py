import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Define paths
model_name = "Meta-Llama-3.1-8B"
zero_shot_base_path = f"/tud/value-adapters/culturellm/eval_outputs/{model_name}/zero-shot/"
cross_culture_path = f"/tud/value-adapters/culturellm/eval_outputs/{model_name}/cross_culture_averaged_f1_scores.csv"
output_csv_path = f"/tud/value-adapters/culturellm/eval_outputs/{model_name}/zero-shot/normalized_f1_scores.csv"
output_heatmap_path = f"/tud/value-adapters/culturellm/eval_outputs/{model_name}/zero-shot/normalized_heatmap.png"

# Define the list of languages
languages = ['arabic', 'bengali', 'chinese', 'english', 'german', 'greek', 'korean', 'portuguese', 'spanish', 'turkish']

# Function to extract data from files directly under the language folder and calculate averages
def extract_and_average_data(base_path):
    averages = {}
    for lang in languages:
        lang_path = os.path.join(base_path, lang)
        f1_scores = []
        if os.path.exists(lang_path):
            for file in os.listdir(lang_path):
                if file.endswith("_res.jsonl"):
                    path = os.path.join(lang_path, file)
                    with open(path, 'r') as f:
                        for line in f:
                            json_data = json.loads(line.strip())
                            f1_score = json_data.get("f1_score")
                            if f1_score is not None:
                                f1_scores.append(f1_score)
        # Calculate average F1 score for the language
        if f1_scores:
            averages[lang] = sum(f1_scores) / len(f1_scores)
        else:
            averages[lang] = None  # Handle case where there are no F1 scores
    return averages

# Extract and average the F1 scores
averaged_f1_scores = extract_and_average_data(zero_shot_base_path)

# Convert the result to a DataFrame for better readability
df_averaged_f1_scores = pd.DataFrame(list(averaged_f1_scores.items()), columns=['Language', 'Average F1 Score'])

# Load the cross-culture evaluation data
df_cross_culture = pd.read_csv(cross_culture_path)

# Define the zero-shot F1 scores based on the extraction
zero_shot_f1_scores = averaged_f1_scores

# Add zero-shot F1 scores to the cross-culture DataFrame
df_cross_culture['F1 Score_zero_shot'] = df_cross_culture['Test Language'].map(zero_shot_f1_scores)

# Subtract zero-shot F1 scores from cross-culture F1 scores based on the Test Language to get normalized F1 scores
df_cross_culture['Normalized F1 Score'] = df_cross_culture['F1 Score'] - df_cross_culture['F1 Score_zero_shot']

# Save the normalized data to CSV with the Adapter Language
df_cross_culture.to_csv(output_csv_path, columns=['Adapter Language', 'Test Language', 'F1 Score', 'F1 Score_zero_shot', 'Normalized F1 Score'], index=False)

# Create a pivot table for the heatmap
df_pivot_normalized = df_cross_culture.pivot(index="Adapter Language", columns="Test Language", values="Normalized F1 Score")

# Create and display the heatmap, rounding the annotations to two decimal places
plt.figure(figsize=(10, 8))
sns.heatmap(df_pivot_normalized, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Normalized F1 Score (%)'})
plt.title('Normalized Adapter Performance Across Cultures (Cross-Culture - Zero-Shot)')
plt.xlabel('Test Language')
plt.ylabel('Adapter Language')

# Save the heatmap as a PNG file
plt.savefig(output_heatmap_path)

# Show the heatmap
plt.show()
