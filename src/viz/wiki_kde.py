#!/usr/bin/env python
# wikipedia_cultures_kde.py

import os
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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from transformers import AutoTokenizer, AutoModel
from sklearn.manifold import TSNE

#############################################################
# 1) Configure Language File Paths
#############################################################
# For example, you might have 10 files:
# "arabic_wiki.txt", "bengali_wiki.txt", ...
# Adjust the paths to your local files.

FILE_PATHS = {
    "arabic":      "tud/value-adapters/culturellm/data/cultural_context/arabic.txt",
    "bengali":     "tud/value-adapters/culturellm/data/cultural_context/bengali.txt",
    "chinese":     "tud/value-adapters/culturellm/data/cultural_context/chinese.txt",
    "english":     "tud/value-adapters/culturellm/data/cultural_context/english.txt",
    "german":      "tud/value-adapters/culturellm/data/cultural_context/german.txt",
    "greek":       "tud/value-adapters/culturellm/data/cultural_context/greek.txt",
    "korean":      "tud/value-adapters/culturellm/data/cultural_context/korean.txt",
    "portuguese":  "tud/value-adapters/culturellm/data/cultural_context/portuguese.txt",
    "spanish":     "tud/value-adapters/culturellm/data/cultural_context/spanish.txt",
    "turkish":     "tud/value-adapters/culturellm/data/cultural_context/turkish.txt"
}

# (Optional) If some files are missing or partial, remove or adjust accordingly.

#############################################################
# 2) Helper: Mean Pooling
#############################################################
def mean_pooling(model_output, attention_mask):
    """
    Mean pool the token embeddings from the final layer.
    model_output[0] shape: (batch_size, seq_len, hidden_dim)
    attention_mask: (batch_size, seq_len)
    """
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )

#############################################################
# 3) Load Wikipedia Text (Local Files)
#############################################################
def load_wiki_texts(file_paths):
    """
    Reads each file, splits by lines, and returns a DataFrame
    with columns [text, language].
    """
    rows = []
    for lang, path in file_paths.items():
        if not os.path.isfile(path):
            print(f"Warning: missing file for '{lang}': {path}")
            continue
        with open(path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        # Filter out empty lines
        lines = [ln.strip() for ln in lines if ln.strip()]

        # Create rows
        for ln in lines:
            rows.append({"text": ln, "language": lang})

    df = pd.DataFrame(rows)
    return df

#############################################################
# 4) Embed Text in Batches
#############################################################
def embed_texts(texts, tokenizer, model, device, batch_size=16, max_length=512):
    """
    Tokenizes and encodes text in batches with mean pooling.
    """
    all_embeddings = []
    for start_idx in tqdm(range(0, len(texts), batch_size), desc="Embedding"):
        batch_texts = texts[start_idx : start_idx + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt"
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)

        with torch.no_grad():
            output = model(input_ids, attention_mask=attention_mask)
        embeddings = mean_pooling(output, attention_mask)
        all_embeddings.append(embeddings.cpu())

    return torch.cat(all_embeddings, dim=0).numpy()

#############################################################
# 5) Main Pipeline
#############################################################
def main():
    # A) Load local Wikipedia data
    print("Loading local Wikipedia text for each language...")
    df = load_wiki_texts(FILE_PATHS)
    print(f"Total lines loaded: {len(df)}")

    # Optional: remove duplicates or sample if huge
    # df.drop_duplicates(subset=["text"], keep="first", inplace=True)
    # df = df.sample(n=5000, random_state=42).reset_index(drop=True)

    print("DataFrame head:\n", df.head())

    # B) Load multilingual embedding model (e.g., LaBSE or any other)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model_name = "sentence-transformers/all-MiniLM-L6-v2"  # or "sentence-transformers/xlm-r-100langs-bert-base-nli-stsb-mean-tokens"
    model_name = "sentence-transformers/LaBSE"

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device)
    model.eval()

    # C) Embed text
    print("Embedding texts...")
    texts = df["text"].tolist()
    embeddings = embed_texts(texts, tokenizer, model, device, batch_size=16, max_length=512)

    # D) (Optional) Preprocessing: scale + PCA
    # Sometimes scaling and reducing dimensions first helps UMAP
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    print("Scaling + PCA on embeddings...")
    scaler = StandardScaler()
    embeddings_scaled = scaler.fit_transform(embeddings)

    pca = PCA(n_components=50, random_state=42)
    embeddings_pca = pca.fit_transform(embeddings_scaled)

    # E) UMAP
    # print("Applying UMAP...")
    reducer = umap.UMAP(
        n_neighbors=15,       # smaller => more local separation
        min_dist=0.1,
        n_components=2,
        # metric="cosine",
        random_state=42
    )
    umap_2d = reducer.fit_transform(embeddings_pca)

    # tsne = TSNE(
    #     n_components=2,
    #     perplexity=30,
    #     learning_rate='auto',
    #     init='random',
    #     random_state=42
    # )
    # umap_2d = tsne.fit_transform(embeddings_pca)

    df["UMAP1"] = umap_2d[:, 0]
    df["UMAP2"] = umap_2d[:, 1]

    # F) Plot: KDE by language
    # matplotlib.rcParams['savefig.transparent'] = True

    # Vibrant color palette
    language_palette = {
        "arabic":     "#FF4B4B",
        "bengali":    "#FF8C42",
        "chinese":    "#FFD93D",
        "english":    "#4B7BFF",
        "german":     "#0091FF",
        "greek":      "#00C896",
        "korean":     "#FFB302",
        "portuguese": "#00B4DB",
        "spanish":    "#2ECC71",
        "turkish":    "#A44DFF"
    }

    sns.set_context("talk", font_scale=1.2)
    sns.set_style("white")
    plt.figure(figsize=(14, 10))

    unique_langs = sorted(df["language"].unique())
    for lang in unique_langs:
        subset = df[df["language"] == lang]
        color = language_palette.get(lang, "#999999")
        sns.kdeplot(
            x=subset["UMAP1"],
            y=subset["UMAP2"],
            fill=True,
            alpha=0.5,
            color=color,
            linewidth=2,
            levels=10,
            bw_adjust=1.5
        )

    # Legend
    handles = [
        mpatches.Patch(color=language_palette[lang], label=lang.capitalize(), alpha=0.8)
        for lang in unique_langs if lang in language_palette
    ]
    plt.legend(
        handles=handles,
        title="Language",
        loc="lower right",
        frameon=False,
        fontsize=18
    )

    plt.title("Wikipedia Data Clustering by Language: UMAP + KDE", fontsize=24, pad=15)
    plt.xlabel("UMAP Dimension 1", fontsize=18)
    plt.ylabel("UMAP Dimension 2", fontsize=18)

    ax = plt.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#FAFAFA')

    plt.tight_layout()
    plt.savefig("./wikipedia_umap_kde.png", dpi=300, bbox_inches="tight", transparent=True)
    plt.show()

if __name__ == "__main__":
    main()
