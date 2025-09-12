#!/usr/bin/env python
# normad_cultures_kde.py

import re
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import umap
from tqdm import tqdm
from sklearn.manifold import TSNE

# Hugging Face
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

#############################################################
# 1) Define mapping: culture -> list of countries
#############################################################
language_to_countries = {
    "arabic": [
        "egypt", "lebanon", "sudan", "iraq", "saudi_arabia", "palestinian_territories", "syria"
    ],
    "bengali": ["bangladesh"],
    "chinese": ["china", "hong_kong", "taiwan", "singapore"],
    "english": [
        "united_kingdom", "united_states_of_america", "australia",
        "new_zealand", "canada", "ireland"
    ],
    "german": ["germany", "austria"],
    "greek": ["greece", "cyprus"],
    "korean": ["south_korea"],
    "portuguese": ["portugal", "brazil"],
    "spanish": [
        "spain", "argentina", "colombia", "chile", "venezuela",
        "mexico", "peru"
    ],
    "turkish": ["türkiye"]
}

#############################################################
# 2) Invert the mapping: country -> culture
#############################################################
country_to_culture = {}
for culture, ctry_list in language_to_countries.items():
    for ctry in ctry_list:
        if ctry not in country_to_culture:
            country_to_culture[ctry] = culture

#############################################################
# 3) Boilerplate Removal Function
#############################################################
def remove_boilerplate(text):
    """
    Removes lines starting with ###,
    triple-dashes, and extra whitespace.
    Adjust or expand this if you have other
    boilerplate patterns to remove.
    """
    # Remove lines like '### Something'
    cleaned = re.sub(r"###.*(\n)?", "", text)
    # Remove triple (or more) dashes
    cleaned = re.sub(r"---+", " ", cleaned)
    # Condense repeated whitespace into single space
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned

#############################################################
# 4) Mean pooling utility for LaBSE
#############################################################
def mean_pooling(model_output, attention_mask):
    """
    Mean pooling of token embeddings.
    """
    token_embeddings = model_output[0]  # (batch_size, seq_len, hidden_dim)
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

#############################################################
# 5) Main function
#############################################################
def main():
    print("Loading full NormAd dataset from Hugging Face...")
    normad_dataset = load_dataset("akhilayerukola/NormAd", split="train")
    print(f"Total NormAd samples: {len(normad_dataset)}")

    # Filter out countries not in our mapping
    def is_valid_country(example):
        return example["Country"] in country_to_culture

    normad_filtered = normad_dataset.filter(is_valid_country)
    print(f"Filtered to relevant countries: {len(normad_filtered)}")

    # Convert to Pandas for easy manipulation
    df = normad_filtered.to_pandas()

    # Create a 'culture' column (e.g., if Country == "iraq", then culture == "arabic")
    df["culture"] = df["Country"].map(country_to_culture)

    # Combine NormAd fields into a single text column, then remove boilerplate
    combined_texts = []
    for idx, row in df.iterrows():
        background     = str(row.get("Background", ""))
        rule_of_thumb  = str(row.get("Rule-of-Thumb", ""))
        story          = str(row.get("Story", ""))
        explanation    = str(row.get("Explanation", ""))

        combined = " ".join([background, rule_of_thumb, story, explanation])
        cleaned  = remove_boilerplate(combined)  # <---- Remove boilerplate
        combined_texts.append(cleaned.strip())

    df["text"] = combined_texts

    print("Preview of final data:\n", df.head())

    #############################################################
    # 6) Load LaBSE model + tokenizer
    #############################################################
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # labse_model_name = "sentence-transformers/all-MiniLM-L6-v2"
    labse_model_name = "sentence-transformers/LaBSE"
    tokenizer = AutoTokenizer.from_pretrained(labse_model_name)
    model = AutoModel.from_pretrained(labse_model_name).to(device)
    model.eval()

    print("Encoding text with LaBSE embeddings...")
    batch_size = 16
    all_embeddings = []

    texts = df["text"].tolist()
    for start_idx in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[start_idx : start_idx + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            model_output = model(input_ids, attention_mask=attention_mask)
        embeddings = mean_pooling(model_output, attention_mask)
        all_embeddings.append(embeddings.cpu())

    # Shape: (num_samples, 768)
    all_embeddings = torch.cat(all_embeddings, dim=0).numpy()

    #############################################################
    # 7) UMAP or t-SNE Dimensionality Reduction
    #############################################################
    print("Reducing embeddings to 2D via UMAP...")
    reducer = umap.UMAP(
        n_neighbors=15,
        min_dist=0.1,
        n_components=2,
        random_state=42
    )
    umap_2d = reducer.fit_transform(all_embeddings)

    # If you want t-SNE instead, comment out above lines and uncomment:
    # tsne = TSNE(
    #     n_components=2,
    #     perplexity=30,
    #     learning_rate='auto',
    #     init='random',
    #     random_state=42
    # )
    # umap_2d = tsne.fit_transform(all_embeddings)

    df["UMAP1"] = umap_2d[:, 0]
    df["UMAP2"] = umap_2d[:, 1]

    #############################################################
    # 8) Plot: One KDE per Culture (WVS styling)
    #############################################################
    # matplotlib.rcParams['savefig.transparent'] = True

    # Define a more vibrant color palette for each culture
    language_palette = {
        "arabic":     "#FF4B4B",  # Bright red
        "bengali":    "#FF8C42",  # Bright orange
        "chinese":    "#FFD93D",  # Bright yellow
        "english":    "#4B7BFF",  # Bright blue
        "german":     "#0091FF",  # Electric blue
        "greek":      "#00C896",  # Bright turquoise
        "korean":     "#FFB302",  # Golden yellow
        "portuguese": "#00B4DB",  # Ocean blue
        "spanish":    "#2ECC71",  # Bright green
        "turkish":    "#A44DFF"   # Bright purple
    }

    sns.set_context("talk", font_scale=1.2)
    sns.set_style("white")
    plt.figure(figsize=(14, 10))

    unique_cultures = sorted(df["culture"].unique())
    for cult in unique_cultures:
        subset = df[df["culture"] == cult]
        color = language_palette.get(cult, "#999999")  # Gray if missing
        sns.kdeplot(
            x=subset["UMAP1"],
            y=subset["UMAP2"],
            fill=True,
            alpha=0.5,
            color=color,
            linewidth=2,
            levels=10,
            bw_adjust=2
        )

    # Build legend manually with patches
    handles = [
        mpatches.Patch(color=language_palette[cult], label=cult.capitalize(), alpha=0.8)
        for cult in unique_cultures if cult in language_palette
    ]
    plt.legend(
        handles=handles,
        title="Culture",
        loc="lower right",
        frameon=False,
        fontsize=18
    )

    plt.title("NormAd Data Clustering: UMAP + KDE by Culture", fontsize=30, pad=20)
    plt.xlabel("Dimension 1 (UMAP)", fontsize=24)
    plt.ylabel("Dimension 2 (UMAP)", fontsize=24)

    # Remove top and right spines
    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Optional: Set a subtle grid or background color
    ax.set_facecolor('#FAFAFA')

    plt.tight_layout()
    plt.savefig("./normad_umap_kde.png", dpi=300, bbox_inches="tight", transparent=True)
    plt.show()

if __name__ == "__main__":
    main()
