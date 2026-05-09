from sentence_transformers import SentenceTransformer
from content_reader import get_all_content
import numpy as np
import hdbscan
from sklearn.metrics.pairwise import cosine_distances
import os

model = SentenceTransformer('all-MiniLM-L6-v2')


def get_representative_chunks(cluster_id, labels, chunk_texts, embeddings, top_n=5):
    cluster_indices = [i for i, l in enumerate(labels) if l == cluster_id]
    if not cluster_indices:
        return []
    cluster_embeddings = embeddings[cluster_indices]
    centroid = cluster_embeddings.mean(axis=0, keepdims=True)
    distances = cosine_distances(centroid, cluster_embeddings)[0]
    top_indices = np.argsort(distances)[:top_n]
    return [chunk_texts[cluster_indices[i]] for i in top_indices]


def getans(folder_path):

    file_texts = get_all_content(folder_path)  # now returns {filename: [chunks]}

    if len(file_texts) == 0:
        raise ValueError("No files were read. Check your folder path and reader functions.")

    chunk_texts = []
    chunk_to_doc = []

    # ✅ CHANGED: file_texts values are already lists of chunks, no chunk_text() call needed
    for name, chunks in file_texts.items():
        if not chunks:
            continue
        for chunk in chunks:
            chunk_texts.append(chunk)
            chunk_to_doc.append(name)

    print(f"\nTotal chunks: {len(chunk_texts)}")

    if len(chunk_texts) == 0:
        raise ValueError("No chunks created. Your files may be empty or unreadable.")

    embeddings = model.encode(chunk_texts, show_progress_bar=True)
    embeddings = embeddings.astype(np.float64)

    distance_matrix = cosine_distances(embeddings)
    distance_matrix = distance_matrix.astype(np.float64)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=3,
        min_samples=2,
        metric='precomputed'
    )
    labels = clusterer.fit_predict(distance_matrix)

    doc_cluster_count = {}
    for i in range(len(labels)):
        label = labels[i]
        if label == -1:
            continue
        doc = chunk_to_doc[i]
        weight = len(chunk_texts[i].split())
        if doc not in doc_cluster_count:
            doc_cluster_count[doc] = {}
        if label not in doc_cluster_count[doc]:
            doc_cluster_count[doc][label] = 0
        doc_cluster_count[doc][label] += weight

    doc_final_cluster = {}
    for doc in doc_cluster_count:
        cluster_counts = doc_cluster_count[doc]
        total = sum(cluster_counts.values())
        best_cluster = max(cluster_counts, key=cluster_counts.get)
        confidence = cluster_counts[best_cluster] / total
        if confidence < 0.4:
            doc_final_cluster[doc] = -1
        else:
            doc_final_cluster[doc] = best_cluster

    # ✅ CHANGED: iterate file_texts keys (not file_texts itself, same thing but explicit)
    for doc in file_texts:
        if doc not in doc_final_cluster:
            doc_final_cluster[doc] = -1

    final_clusters = {}
    for doc, cluster in doc_final_cluster.items():
        if cluster not in final_clusters:
            final_clusters[cluster] = []
        final_clusters[cluster].append(doc)

    cluster_representative_text = {}
    for cluster_id in final_clusters:
        if cluster_id == -1:
            continue

        rep_chunks = get_representative_chunks(
            cluster_id, labels, chunk_texts, embeddings, top_n=5
        )

        doc_names = final_clusters[cluster_id]
        doc_list = "\n".join(f"- {d}" for d in doc_names)
        chunk_list = "\n\n".join(
            f"[Excerpt {i+1}]: {c[:300]}" for i, c in enumerate(rep_chunks)
        )

        cluster_representative_text[cluster_id] = (
            f"Files in group:\n{doc_list}\n\nRepresentative excerpts:\n{chunk_list}"
        )

    toreturn = {}
    for cluster_id, docs in final_clusters.items():
        rep_text = cluster_representative_text.get(cluster_id, "")
        toreturn[str(cluster_id)] = {
            "files": docs,
            "llm_input": rep_text if cluster_id != -1 else "other"
        }

    return toreturn


if __name__ == "__main__":
    folder_path = r"C:\Users\aditya\Desktop\my"
    ans = getans(folder_path)
    print(f"the answer is: {ans}")