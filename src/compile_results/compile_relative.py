import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


model_name = "Meta-Llama-3.1-8B"
base_save_path = f"/tud/value-adapters/culturellm/eval_outputs/{model_name}/zero-shot/"
zero_shot_base_path = base_save_path
cross_culture_path = f"/tud/value-adapters/culturellm/eval_outputs/{model_name}/cross_culture_averaged_f1_scores.csv"


languages = ['arabic', 'bengali', 'chinese', 'english', 'german', 'greek', 'korean', 'portuguese', 'spanish', 'turkish']


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

        if f1_scores:
            averages[lang] = sum(f1_scores) / len(f1_scores)
        else:
            averages[lang] = None  
    return averages

averaged_f1_scores = extract_and_average_data(zero_shot_base_path)

df_averaged_f1_scores = pd.DataFrame(list(averaged_f1_scores.items()), columns=['Language', 'Average F1 Score'])

df_cross_culture = pd.read_csv(cross_culture_path)

df_cross_culture['F1 Score_zero_shot'] = df_cross_culture['Test Language'].map(averaged_f1_scores)

#  performance difference
df_cross_culture['Performance Difference'] = (df_cross_culture['F1 Score'] - df_cross_culture['F1 Score_zero_shot']) * 100

# max norm.
df_pivot = df_cross_culture.pivot(index="Adapter Language", columns="Test Language", values="F1 Score")
df_pivot_normalized_by_column = df_pivot.div(df_pivot.max(axis=0), axis=1) * 100

# relative change
df_cross_culture['Relative Change (%)'] = ((df_cross_culture['F1 Score'] - df_cross_culture['F1 Score_zero_shot']) / df_cross_culture['F1 Score_zero_shot'])*100


relative_change_csv_path = os.path.join(base_save_path, 'relative_change_f1_scores.csv')
column_normalized_csv_path = os.path.join(base_save_path, 'column_normalized_f1_scores.csv')
df_cross_culture.to_csv(relative_change_csv_path, columns=['Adapter Language', 'Test Language', 'F1 Score', 'F1 Score_zero_shot', 'Relative Change (%)'], index=False)
df_normalized_flattened = df_pivot_normalized_by_column.reset_index().melt(id_vars=["Adapter Language"], var_name="Test Language", value_name="Normalized F1 Score")
df_normalized_flattened.to_csv(column_normalized_csv_path, index=False)


performance_difference_heatmap_path = os.path.join(base_save_path, 'performance_difference_heatmap.png')
plt.figure(figsize=(10, 8))
performance_diff_pivot = df_cross_culture.pivot(index="Adapter Language", columns="Test Language", values="Performance Difference")
sns.heatmap(performance_diff_pivot, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Performance Difference'})
plt.title('Performance Difference from Zero-Shot to Cross-Culture')
plt.xlabel('Test Language')
plt.ylabel('Adapter Language')
plt.savefig(performance_difference_heatmap_path)
plt.close()


column_normalized_heatmap_path = os.path.join(base_save_path, 'column_normalized_heatmap.png')
plt.figure(figsize=(10, 8))
sns.heatmap(df_pivot_normalized_by_column, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Normalized F1 Score (%)'})
plt.title('Column-Wise Normalized Adapter Performance')
plt.xlabel('Test Language')
plt.ylabel('Adapter Language')
plt.savefig(column_normalized_heatmap_path)
plt.close()


relative_change_heatmap_path = os.path.join(base_save_path, 'relative_change_heatmap.png')
plt.figure(figsize=(10, 8))
relative_change_pivot = df_cross_culture.pivot(index="Adapter Language", columns="Test Language", values="Relative Change (%)")
sns.heatmap(relative_change_pivot, annot=True, fmt=".2f", cmap="YlGnBu", cbar_kws={'label': 'Relative Change (%)'})
plt.title('Relative Change from Zero-Shot to Cross-Culture')
plt.xlabel('Test Language')
plt.ylabel('Adapter Language')
plt.savefig(relative_change_heatmap_path)
plt.close()



