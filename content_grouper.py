from sentence_transformers import SentenceTransformer
from content_reader import get_all_content
import numpy as np
import hdbscan
from sklearn.metrics.pairwise import cosine_distances
import os
import hashlib
import pickle

model = SentenceTransformer('all-MiniLM-L6-v2')
CACHE_FILE = "embedding_cache.pkl"


def get_hash(chunks):
    return hashlib.md5("||".join(chunks).encode(errors="ignore")).hexdigest()


def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            return pickle.load(f)
    return {}


def save_cache(cache):
    with open(CACHE_FILE, "wb") as f:
        pickle.dump(cache, f)


def get_representative_chunks(cluster_id, labels, chunk_texts, embeddings, top_n=5):
    indices = [i for i, l in enumerate(labels) if l == cluster_id]
    if not indices:
        return []
    cluster_embs = embeddings[indices]
    centroid = cluster_embs.mean(axis=0, keepdims=True)
    distances = cosine_distances(centroid, cluster_embs)[0]
    top = np.argsort(distances)[:top_n]
    return [chunk_texts[indices[i]] for i in top]


def getans(folder_path):

    file_texts = get_all_content(folder_path)
    if not file_texts:
        raise ValueError("No files were read.")

    cache = load_cache()

    chunk_texts = []
    chunk_to_doc = []
    all_embeddings = []

    for name, chunks in file_texts.items():
        if not chunks:
            continue

        file_hash = get_hash(chunks)

        # cache hit — reuse stored embeddings
        if name in cache and cache[name]["hash"] == file_hash:
            embs = cache[name]["embeddings"]
            print(f"[cache hit]  {name}")

        # cache miss — encode and store
        else:
            print(f"[cache miss] {name}")
            embs = model.encode(chunks, show_progress_bar=True).astype(np.float64)
            cache[name] = {"hash": file_hash, "chunks": chunks, "embeddings": embs}

        for chunk, emb in zip(chunks, embs):
            chunk_texts.append(chunk)
            chunk_to_doc.append(name)
            all_embeddings.append(emb)

    save_cache(cache)

    if not chunk_texts:
        raise ValueError("No chunks created.")

    embeddings = np.array(all_embeddings)

    distance_matrix = cosine_distances(embeddings).astype(np.float64)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)

    labels = hdbscan.HDBSCAN(
        min_cluster_size=3,
        min_samples=2,
        metric='precomputed'
    ).fit_predict(distance_matrix)

    # vote: each chunk votes for its cluster, weighted by word count
    doc_cluster_count = {}
    for i, label in enumerate(labels):
        if label == -1:
            continue
        doc = chunk_to_doc[i]
        weight = len(chunk_texts[i].split())
        doc_cluster_count.setdefault(doc, {})
        doc_cluster_count[doc][label] = doc_cluster_count[doc].get(label, 0) + weight

    # assign each doc to its winning cluster
    doc_final_cluster = {}
    for doc, counts in doc_cluster_count.items():
        total = sum(counts.values())
        best = max(counts, key=counts.get)
        doc_final_cluster[doc] = best if counts[best] / total >= 0.4 else -1

    # any doc with no chunks in any cluster → noise
    for doc in file_texts:
        if doc not in doc_final_cluster:
            doc_final_cluster[doc] = -1

    # group docs by cluster
    final_clusters = {}
    for doc, cluster in doc_final_cluster.items():
        final_clusters.setdefault(cluster, []).append(doc)

    # build llm_input per cluster
    toreturn = {}
    for cluster_id, docs in final_clusters.items():
        if cluster_id == -1:
            toreturn["-1"] = {"files": docs, "llm_input": "other"}
            continue

        rep_chunks = get_representative_chunks(
            cluster_id, labels, chunk_texts, embeddings, top_n=5
        )
        doc_list = "\n".join(f"- {d}" for d in docs)
        chunk_list = "\n\n".join(
            f"[Excerpt {i+1}]: {c[:300]}" for i, c in enumerate(rep_chunks)
        )
        toreturn[str(cluster_id)] = {
            "files": docs,
            "llm_input": f"Files in group:\n{doc_list}\n\nRepresentative excerpts:\n{chunk_list}"
        }

    return toreturn


if __name__ == "__main__":
    folder_path = r"C:\Users\aditya\Desktop\my"
    ans = getans(folder_path)
    print(f"the answer is: {ans}")