import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Define the model name and paths
model_name = "Meta-Llama-3.1-8B-Instruct"
cross_culture_path = f"/tud/value-adapters/culturellm/eval_outputs/offenseval/{model_name}/cross_culture"
single_path_template = f"/tud/value-adapters/culturellm/eval_outputs/offenseval/{model_name}/single/{{}}"
base_result_path = f"/tud/value-adapters/culturellm/eval_outputs/offenseval/{model_name}"

# Define the list of adapter languages
adapter_languages = ['arabic', 'bengali', 'chinese', 'english', 'german', 'greek', 'korean', 'portuguese', 'spanish', 'turkish']

# Function to extract data from a specific path
def extract_data_from_path(base_path):
    data = []
    for root, dirs, files in os.walk(base_path):
        for file in files:
            if file.endswith("_res.jsonl"):
                path = os.path.join(root, file)
                task = file.replace("_res.jsonl", "")  # Extract task from filename
                with open(path, 'r') as f:
                    for line in f:
                        json_data = json.loads(line.strip())
                        adapter_lang = os.path.basename(os.path.dirname(root))  # Get the adapter language
                        test_lang = os.path.basename(root)  # Get the test language
                        model = json_data.get("Model")
                        f1_score = json_data.get("f1_score")
                        data.append([adapter_lang, test_lang, task, model, f1_score])
    return data

# Extract data from cross-culture and single paths
cross_culture_data = extract_data_from_path(cross_culture_path)

# Function to extract and enforce diagonal (self-test) values
def extract_self_test_data(lang):
    single_path = single_path_template.format(lang)
    data = extract_data_from_path(single_path)
    
    # Calculate the average F1 score for the self-test (diagonal) and return it
    df_self = pd.DataFrame(data, columns=["Adapter Language", "Test Language", "Task", "Model", "F1 Score"])
    avg_f1 = df_self['F1 Score'].mean()
    return [lang, lang, avg_f1]

# Extract the self-test data for each adapter language
diagonal_data = [extract_self_test_data(lang) for lang in adapter_languages]

# Convert the diagonal data to a DataFrame
df_diagonal = pd.DataFrame(diagonal_data, columns=["Adapter Language", "Test Language", "F1 Score"])

# Combine the cross-culture data with the diagonal data
df_cross_culture = pd.DataFrame(cross_culture_data, columns=["Adapter Language", "Test Language", "Task", "Model", "F1 Score"])
df_avg = df_cross_culture.groupby(['Adapter Language', 'Test Language']).agg({'F1 Score': 'mean'}).reset_index()

# Merge the diagonal data into the cross-culture data
df_combined = pd.concat([df_avg, df_diagonal], ignore_index=True)

# Convert the F1 Score to numeric, handling errors (convert invalid values to NaN)
df_combined['F1 Score'] = pd.to_numeric(df_combined['F1 Score'], errors='coerce')

# Handle NaN values by filling with 0 or another appropriate value
df_combined.fillna(0, inplace=True)

# Save the combined results to a CSV file
csv_combined_path = f"{base_result_path}/cross_culture_averaged_f1_scores.csv"
df_combined.to_csv(csv_combined_path, index=False)

# Create a pivot table for the heatmap, including the diagonal values
df_pivot = df_combined.pivot(index="Adapter Language", columns="Test Language", values="F1 Score")

# Convert the F1 scores to percentages
df_pivot_percentage = df_pivot * 100

# Create a heatmap with percentage formatting
plt.figure(figsize=(10, 8))
sns.heatmap(df_pivot_percentage, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Average F1 Score (%)'})
plt.title(f'Adapter Performance Across Cultures for {model_name} (Averaged F1 Scores as %)')
plt.xlabel('Test Language')
plt.ylabel('Adapter Language')

# Save the heatmap as a PNG file
heatmap_path = f"{base_result_path}/cross_culture_heatmap.png"
plt.savefig(heatmap_path)

# Show the heatmap
plt.show()

# Now create the second plot: Delta (difference) from the diagonal
# Get diagonal values from the original F1 score pivot table
diagonal_f1_scores = pd.Series({lang: df_pivot_percentage.loc[lang, lang] for lang in df_pivot_percentage.index})

# Calculate the delta for each cell: cell_value - diagonal_value
df_delta = df_pivot_percentage.apply(lambda row: row - diagonal_f1_scores.get(row.name, 0), axis=1)

# Replace NaN values in delta with 0 for better visualization
df_delta.fillna(0, inplace=True)

# Create a heatmap showing the delta values (difference from diagonal)
plt.figure(figsize=(10, 8))
sns.heatmap(df_delta, annot=True, fmt=".2f", cmap="coolwarm", cbar_kws={'label': 'Delta from Diagonal (F1 Score %)'})
plt.title(f'Adapter Performance Delta from Diagonal for {model_name}')
plt.xlabel('Test Language')
plt.ylabel('Adapter Language')

# Save the delta heatmap as a PNG file
delta_heatmap_path = f"{base_result_path}/cross_culture_delta_heatmap.png"
plt.savefig(delta_heatmap_path)

# Show the delta heatmap
plt.show()

# Output paths to confirm where files are saved
csv_combined_path, heatmap_path, delta_heatmap_path
