from sentence_transformers import SentenceTransformer
from Contentreading.pdf_reader import get_all_content
from Contentreading.txtreader import get_all_content_txt
import numpy as np
import hdbscan
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_distances

model = SentenceTransformer('all-mpnet-base-v2') 
folder_path=r"C:\Users\aditya\Desktop\filefolder"

file_texts=get_all_content_txt(folder_path)
file_names = list(file_texts.items())  # [("file1.pdf", "text..."), ...]

texts = [text for _, text in file_names]
names = [name for name, _ in file_names]

for name, text in file_texts.items():
    print(f"{name}: {len(text)} chars | preview: {text[:80]}")

embeddings = model.encode(texts, show_progress_bar=True)
distance_matrix = cosine_distances(embeddings).astype('float64')

clusterer = hdbscan.HDBSCAN(
    min_cluster_size=2,
    min_samples=1,
    metric='precomputed'
)
labels = clusterer.fit_predict(distance_matrix)
clusters = defaultdict(list)

for name, label in zip(names, labels):
    clusters[label].append(name)

for label, files in clusters.items():
    cluster_name = "NOISE / Uncategorised" if label == -1 else f"Cluster {label}"
    print(f"\n{cluster_name}:")
    for f in files:
        print(f"  - {f}")
# embeddings.shape → (num_files, 384)  ← 384-dim vector per file