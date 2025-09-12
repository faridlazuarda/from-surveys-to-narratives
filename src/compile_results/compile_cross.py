#!/usr/bin/env python3
"""
Cross‑culture adapter evaluation

Outputs per <scenario>/<model> directory
  1) cross_culture_averaged_f1.csv
  2) cross_culture_f1_pivot_table.csv
  3) cross_culture_heatmap.png
  4) cross_culture_delta_heatmap.png
  5) normalized_cross_culture_heatmap.png
  6) cross_culture_rank_heatmap.png
  7) normalized_cross_culture_matrix.csv      ◀︎ NEW (full N×N matrix)
  8) normalized_diagonal_values.csv
  9) overall_normalized_diagonal.txt
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------------ SETTINGS
# SCENARIOS = ["wiki_only", "all_combined"]
SCENARIOS = ["normad_context"]
MODELS = [
    "gemma-2-9b-it",
    "Meta-Llama-3.1-8B",
    "Meta-Llama-3.1-8B-Instruct",
    "Qwen2.5-7B-Instruct",
]

# ----------------------------------------------------------------- HELPERS
def get_adapter_languages(base):
    """Return adapter‑culture directory names found under any “…/complete/.”"""
    for root, dirs, _ in os.walk(base):
        if os.path.basename(root) == "complete":
            return dirs
    return []

def extract_rows(base):
    """Walk *_res.jsonl files → rows (adapter, test, task, model, f1)."""
    rows = []
    for root, _, files in os.walk(base):
        for fn in files:
            if fn.endswith("_res.jsonl"):
                task = fn[:-len("_res.jsonl")]
                with open(os.path.join(root, fn)) as f:
                    for line in f:
                        j = json.loads(line)
                        rows.append([
                            os.path.basename(os.path.dirname(root)),  # adapter
                            os.path.basename(root),                  # test
                            task,
                            j.get("Model"),
                            j.get("f1_score"),
                        ])
    return rows

# ----------------------------------------------------------------- CORE
def process_scenario_model(scenario: str, model: str):
    data_path = os.path.join(BASE_EVAL_DIR, scenario, model)
    scenario_slug = scenario.lower()
    model_slug    = model.lower().replace(" ", "-")
    output_dir    = os.path.join(BASE_FIGS_DIR, scenario_slug, model_slug)
    os.makedirs(output_dir, exist_ok=True)
    tag = f"{scenario_slug}_{model_slug}"

    # 1) ---------------------------------------------------------------- rows
    rows = extract_rows(data_path)
    df_raw = pd.DataFrame(rows, columns=["Adapter", "Test", "Task", "Model", "F1"])

    # 2) ------------------------------------------------------------- diagonals
    for lang in get_adapter_languages(data_path):
        mask = (df_raw["Adapter"] == lang) & (df_raw["Test"] == lang)
        if not mask.any():
            avg_f1 = np.nan
        else:
            avg_f1 = df_raw.loc[mask, "F1"].mean()
        df_raw = pd.concat([df_raw, pd.DataFrame([[lang, lang, np.nan, np.nan, avg_f1]],
                                                 columns=df_raw.columns)],
                           ignore_index=True)

    # 3) ----------------------------------------------- adapter–test averages
    df_mean = (df_raw.groupby(["Adapter", "Test"], as_index=False)
                     .agg({"F1": "mean"}))

    mean_csv = os.path.join(output_dir, f"{tag}_cross_culture_averaged_f1.csv")
    df_mean.to_csv(mean_csv, index=False)

    # 4) ---------------------------------------------------------------- pivot
    df_pivot = df_mean.pivot(index="Adapter", columns="Test", values="F1") * 100
    pivot_csv = os.path.join(output_dir, f"{tag}_cross_culture_f1_pivot_table.csv")
    df_pivot.to_csv(pivot_csv)

    # 5) ------------------------------------------------------------- cleanups
    long_name = "arabic,bengali,chinese,english,german,greek,korean,portuguese,spanish,turkish"
    df_heat = df_pivot.copy()
    df_heat.rename(index={long_name: "combined"}, columns={long_name: "combined"}, inplace=True)
    df_heat.drop(index="combined", columns="combined", errors="ignore", inplace=True)

    # 6) ---------------------------------------------------------- heatmap raw
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_heat, annot=True, fmt=".2f", cmap="YlGnBu",
                cbar_kws={"label": "Average F1 Score (%)"})
    plt.title(f"{model} – Adapter vs. Test‑Culture (avg. F1)")
    plt.xlabel("Test Culture"); plt.ylabel("Adapter Culture")
    plt.savefig(os.path.join(output_dir, f"{tag}_cross_culture_heatmap.png"),
                bbox_inches="tight")
    plt.close()

    # 7) ---------------------------------------------------- Δ from diagonal
    diag_series = pd.Series({lang: df_heat.loc[lang, lang]
                             for lang in df_heat.index if lang in df_heat.columns})
    df_delta = df_heat.sub(diag_series, axis=0).fillna(0)

    plt.figure(figsize=(10, 8))
    sns.heatmap(df_delta, annot=True, fmt=".2f", cmap="coolwarm",
                cbar_kws={"label": "Δ from Diagonal (F1 %)"})
    plt.title(f"{model} – Δ from Self‑Test")
    plt.xlabel("Test Culture"); plt.ylabel("Adapter Culture")
    plt.savefig(os.path.join(output_dir, f"{tag}_cross_culture_delta_heatmap.png"),
                bbox_inches="tight")
    plt.close()

    # 8) ------------------------------------------------ column normalisation
    df_norm = df_heat.div(df_heat.max())     # 0‑1 scale per column

    # 8‑A) ***SAVE FULL N×N MATRIX***  ◀︎ NEW
    norm_matrix_csv = os.path.join(output_dir, f"{tag}_normalized_cross_culture_matrix.csv")
    df_norm.to_csv(norm_matrix_csv)

    # 8‑B) Heatmap of normalised scores
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_norm, annot=True, fmt=".2f", cmap="YlGnBu",
                cbar_kws={"label": "Normalised score"})
    plt.title(f"{model} – Normalised (per test culture)")
    plt.xlabel("Test Culture"); plt.ylabel("Adapter Culture")
    plt.savefig(os.path.join(output_dir, f"{tag}_normalized_cross_culture_heatmap.png"),
                bbox_inches="tight")
    plt.close()

    # 9) ----------------------------------------------------------- ranking
    df_rank = df_norm.rank(method="dense", ascending=False)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df_rank, annot=True, fmt=".0f", cmap="YlOrBr_r",
                cbar_kws={"label": "Rank (1 = best)"})
    plt.title(f"{model} – Rank of Normalised Scores")
    plt.xlabel("Test Culture"); plt.ylabel("Adapter Culture")
    plt.savefig(os.path.join(output_dir, f"{tag}_cross_culture_rank_heatmap.png"),
                bbox_inches="tight")
    plt.close()

    # 10) -------------------------------------------- normalised diagonals
    norm_diag = {l: df_norm.loc[l, l] for l in df_norm.index if l in df_norm.columns}
    df_norm_diag = (pd.Series(norm_diag, name="NormalizedDiagonal")
                      .sort_values(ascending=False)
                      .reset_index()
                      .rename(columns={"index": "Culture"}))
    diag_csv = os.path.join(output_dir, f"{tag}_normalized_diagonal_values.csv")
    df_norm_diag.to_csv(diag_csv, index=False)

    overall_diag = df_norm_diag["NormalizedDiagonal"].mean()
    with open(os.path.join(output_dir, f"{tag}_overall_normalized_diagonal.txt"), "w") as f:
        f.write(f"{overall_diag:.4f}\n")

    # 11) --------------------------------------------- console summary
    print("========================================================")
    print(f"{scenario} / {model}")
    print(f"Saved normalised matrix: {norm_matrix_csv}")
    print(df_norm_diag.to_string(index=False))
    print(f"Average normalised diagonal: {overall_diag:.4f}")
    print("========================================================\n")

# ---------------------------------------------------------------- DRIVER
if __name__ == "__main__":
    for scenario in SCENARIOS:
        for model in MODELS:
            process_scenario_model(scenario, model)
