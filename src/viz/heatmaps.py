import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Load the CSV file
file_path = 'tud/value-adapters/culturellm/eval_outputs/normal/Meta-Llama-3.1-8B/cross_culture_f1_pivot_table.csv'
df = pd.read_csv(file_path, index_col=0)

# Rename the row containing the combination of languages to "combined"
df.rename(index={'arabic,bengali,chinese,english,german,greek,korean,portuguese,spanish,turkish': 'combined'}, inplace=True)

# Extract experiment type and model name from the file path
path_parts = file_path.split('/')
experiment_type = path_parts[-3]  # 'normal'
model_name = path_parts[-2]       # 'Meta-Llama-3.1-8B'

# Plotting the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'F1 Score (%)'})

# Adding title and labels
plt.title(f"Adapter Performance Across Cultures for {model_name}")
plt.xlabel("Test Culture")
plt.ylabel("Adapter Culture")

# Construct the output filename
output_filename = f'tud/value-adapters/culturellm/src/viz/{experiment_type}_{model_name}.pdf'

# Save the figure
plt.savefig(output_filename, dpi=300, bbox_inches='tight')

# Show the plot
plt.show()
