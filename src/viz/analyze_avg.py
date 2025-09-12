import pandas as pd
import numpy as np
import scipy.spatial.distance as distance  # Imported as 'distance'
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from umap import UMAP
import os

# --- Beautification Settings ---
sns.set_style("whitegrid")
sns.set_context("talk", font_scale=1.1)  # Adjust font scale as desired

# Step 1: Read the Data from CSV
df = pd.read_csv("tud/value-adapters/culturellm/data/cultural_avg.csv")

# Step 2: Extract Cultural Data (assuming columns Q27 to Q233)
cultural_data = df.iloc[:, 1:].values

# 1. Cosine Similarity Calculation
cosine_similarity_matrix = 1 - distance.squareform(distance.pdist(cultural_data, metric='cosine'))

# 2. Pearson Correlation Calculation
df_features = df.set_index('Country')
pearson_corr_matrix = df_features.T.corr(method='pearson').values

# 3. Euclidean Distance Calculation
euclidean_distance_matrix = distance.squareform(distance.pdist(cultural_data, metric='euclidean'))

# Define the directory to save figures
save_dir = 'tud/value-adapters/culturellm/figs'
os.makedirs(save_dir, exist_ok=True)

# ------------------ 1. Cosine Similarity Heatmap ------------------
plt.figure(figsize=(12, 10))
sns.heatmap(
    cosine_similarity_matrix, 
    xticklabels=df['Country'], 
    yticklabels=df['Country'], 
    annot=False,  # Set to False if you prefer cleaner look; True if you want to see numeric values
    cmap='coolwarm', 
    fmt=".2f"
)
plt.title('Cosine Similarity between Cultures', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'cosine_similarity_heatmap.pdf'), format='pdf', dpi=300)
plt.close()

# ------------------ 2. Pearson Correlation Heatmap ------------------
plt.figure(figsize=(12, 10))
sns.heatmap(
    pearson_corr_matrix, 
    xticklabels=df['Country'], 
    yticklabels=df['Country'], 
    annot=False,  # Toggle this if you prefer values
    cmap='coolwarm', 
    center=0, 
    fmt=".2f"
)
plt.title('Pearson Correlation between Cultures', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'pearson_correlation_heatmap.pdf'), format='pdf', dpi=300)
plt.close()

# ------------------ 3. Euclidean Distance Heatmap ------------------
plt.figure(figsize=(12, 10))
sns.heatmap(
    euclidean_distance_matrix, 
    xticklabels=df['Country'], 
    yticklabels=df['Country'], 
    annot=False,  # Toggle if you need numeric values
    cmap='viridis', 
    fmt=".2f"
)
plt.title('Euclidean Distance between Cultures', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'euclidean_distance_heatmap.pdf'), format='pdf', dpi=300)
plt.close()

# ------------------ 4. PCA Visualization ------------------
pca = PCA(n_components=2)
pca_result = pca.fit_transform(cultural_data)
df['PCA1'] = pca_result[:, 0]
df['PCA2'] = pca_result[:, 1]

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='PCA1', y='PCA2', hue='Country', data=df, 
    palette='tab10', s=100, edgecolor='black'
)
plt.title('PCA of Cultural Data', fontsize=16)
plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'pca_cultural_data.pdf'), format='pdf', dpi=300)
plt.close()

# ------------------ 5. t-SNE Visualization ------------------
tsne = TSNE(n_components=2, perplexity=5, random_state=42)
tsne_result = tsne.fit_transform(cultural_data)
df['TSNE1'] = tsne_result[:, 0]
df['TSNE2'] = tsne_result[:, 1]

plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='TSNE1', y='TSNE2', hue='Country', data=df, 
    palette='tab10', s=100, edgecolor='black'
)
# Optionally, add labels next to points
for i, txt in enumerate(df['Country']):
    plt.text(
        df['TSNE1'].iloc[i] + 0.1, df['TSNE2'].iloc[i] + 0.1, 
        txt, fontsize=9
    )
plt.title('t-SNE of Cultural Data', fontsize=16)
plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'tsne_cultural_data_with_labels.pdf'), format='pdf', dpi=300)
plt.close()

# ------------------ 6. K-Means Clustering ------------------
inertia = []
K = range(1, 10)
for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(cultural_data)
    inertia.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(K, inertia, 'bo-')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method for Optimal k', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'elbow_method.pdf'), format='pdf', dpi=300)
plt.close()

# Choose optimal_k after reviewing elbow method
optimal_k = 3  # Update based on your elbow plot
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Cluster'] = kmeans.fit_predict(cultural_data)

# Visualize Clusters using PCA
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x='PCA1', y='PCA2', hue='Country', data=df, 
    palette='tab10', s=100, edgecolor='black',
    legend=False
)
for i, txt in enumerate(df['Country']):
    plt.text(
        df['PCA1'].iloc[i] + 0.05, df['PCA2'].iloc[i] + 0.05,
        txt, fontsize=9
    )
plt.title('PCA of Cultural Data (with labels)', fontsize=16)
# plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)

# plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'pca_cultural_data_with_labels.pdf'), format='pdf', dpi=300)
plt.close()

# ------------------ 7. UMAP Visualization ------------------
umap = UMAP(n_components=2, n_neighbors=5, min_dist=0.3, metric='euclidean', random_state=42)
umap_result = umap.fit_transform(cultural_data)
df['UMAP1'] = umap_result[:, 0]
df['UMAP2'] = umap_result[:, 1]

# UMAP Visualization Colored by Country
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='UMAP1', y='UMAP2', hue='Country', data=df, 
    palette='tab10', s=100, edgecolor='black'
)
for i, txt in enumerate(df['Country']):
    plt.text(
        df['UMAP1'].iloc[i] + 0.05, df['UMAP2'].iloc[i] + 0.05,
        txt, fontsize=9
    )
plt.title('UMAP of Cultural Data Colored by Country', fontsize=16)
plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'umap_cultural_data_with_labels.pdf'), format='pdf', dpi=300)
plt.close()

# UMAP Visualization Colored by Cluster
plt.figure(figsize=(10, 8))
sns.scatterplot(
    x='UMAP1', y='UMAP2', hue='Cluster', data=df, 
    palette='tab10', s=100, edgecolor='black'
)
plt.title('UMAP of Cultural Data with K-Means Clusters', fontsize=16)
plt.legend(loc='best', bbox_to_anchor=(1.05, 1), borderaxespad=0)
plt.tight_layout()
plt.savefig(os.path.join(save_dir, 'umap_cultural_data_clusters.pdf'), format='pdf', dpi=300)
plt.close()
