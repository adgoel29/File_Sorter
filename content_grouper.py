from sentence_transformers import SentenceTransformer
from content_reader import get_all_content
import numpy as np
import hdbscan
import umap
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


def weighted_average_embedding(chunks, embeddings):
    """Weighted average: longer chunks contribute more (by word count)."""
    weights = np.array([len(c.split()) for c in chunks], dtype=np.float64)
    total = weights.sum()
    if total == 0:
        return embeddings.mean(axis=0)
    return (embeddings * weights[:, None]).sum(axis=0) / total


def get_umap_params(n_docs):
    if n_docs < 5:
        return None
    elif n_docs <= 20:
        return dict(n_neighbors=min(3, n_docs - 1), min_dist=0.02, n_components=2)
        # return None
    elif n_docs <= 200:
        return dict(n_neighbors=min(15, n_docs - 1), min_dist=0.05, n_components=2)
    else:
        return dict(n_neighbors=min(30, n_docs - 1), min_dist=0.0, n_components=2)


def get_hdbscan_params(n_docs):
    if n_docs <= 100:
        return dict(min_cluster_size=2, min_samples=1)
    else:
        size = max(5, int(np.sqrt(n_docs)))
        return dict(min_cluster_size=size, min_samples=2)


def build_cosine_distance_matrix(embeddings):
    """Symmetric, zero-diagonal cosine distance matrix for HDBSCAN precomputed."""
    dist = cosine_distances(embeddings).astype(np.float64)
    dist = (dist + dist.T) / 2          # enforce perfect symmetry
    np.fill_diagonal(dist, 0.0)
    np.clip(dist, 0.0, None, out=dist)  # kill any tiny negatives from float errors
    return dist


def getans(folder_path):

    file_texts = get_all_content(folder_path)
    if not file_texts:
        raise ValueError("No files were read.")

    cache = load_cache()

    # ── 1. Chunking + Sentence Embeddings (cached per file) ──────────────────
    doc_names = []
    doc_avg_embeddings = []
    doc_rep_chunks = {}

    for name, chunks in file_texts.items():
        if not chunks:
            print(f"[skip] {name} — no chunks")
            continue

        file_hash = get_hash(chunks)

        if name in cache and cache[name]["hash"] == file_hash:
            embs = cache[name]["embeddings"]
            print(f"[cache hit]  {name}")
        else:
            print(f"[cache miss] {name}")
            embs = model.encode(chunks, show_progress_bar=False).astype(np.float64)
            cache[name] = {"hash": file_hash, "chunks": chunks, "embeddings": embs}

        # ── 2. Weighted average embedding per document ────────────────────
        avg_emb = weighted_average_embedding(chunks, embs)

        doc_names.append(name)
        doc_avg_embeddings.append(avg_emb)
        doc_rep_chunks[name] = chunks

    save_cache(cache)

    if not doc_names:
        raise ValueError("No documents with chunks found.")

    n_docs = len(doc_names)
    doc_matrix = np.array(doc_avg_embeddings)  # shape: (n_docs, emb_dim)

    # ── 3. UMAP (dimensionality reduction only — for structure, not clustering) 
    umap_params = get_umap_params(n_docs)

    if umap_params is None:
        print(f"[umap] skipped (only {n_docs} docs)")
    else:
        print(f"[umap] reducing {n_docs} docs with params {umap_params}")
        reducer = umap.UMAP(**umap_params, metric="cosine", random_state=42)
        umap_output = reducer.fit_transform(doc_matrix)  # shape: (n_docs, 2)
        doc_matrix = umap_output

    # #     # Lift UMAP output back to unit vectors so cosine distance is meaningful.
    # #     # Normalize each 2D point onto the unit circle before distance computation.
        norms = np.linalg.norm(umap_output, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        doc_matrix = umap_output / norms  # replace doc_matrix with normalized UMAP coords

    # ── 4. Precomputed cosine distance matrix ─────────────────────────────────
    print("[hdbscan] building cosine distance matrix...")
    dist_matrix = build_cosine_distance_matrix(doc_matrix)

    # ── 5. HDBSCAN with precomputed metric ────────────────────────────────────
    hdb_params = get_hdbscan_params(n_docs)
    print(f"[hdbscan] params {hdb_params} | metric=precomputed (cosine)")

    labels = hdbscan.HDBSCAN(
        **hdb_params,
        metric="precomputed"
    ).fit_predict(dist_matrix)

    print(f"[hdbscan] labels: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # ── 6. Build output clusters ──────────────────────────────────────────────
    final_clusters = {}
    for doc, label in zip(doc_names, labels):
        final_clusters.setdefault(int(label), []).append(doc)

    # ── 7. Pick representative excerpts per cluster ───────────────────────────
    def pick_excerpts(docs, top_n=5):
        all_chunks = []
        for d in docs:
            all_chunks.extend(doc_rep_chunks.get(d, []))
        all_chunks.sort(key=lambda c: len(c.split()), reverse=True)
        return all_chunks[:top_n]

    toreturn = {}
    for cluster_id, docs in final_clusters.items():
        if cluster_id == -1:
            toreturn["-1"] = {"files": docs, "llm_input": "other"}
            continue

        excerpts = pick_excerpts(docs)
        doc_list = "\n".join(f"- {d}" for d in docs)
        chunk_list = "\n\n".join(
            f"[Excerpt {i+1}]: {c[:300]}" for i, c in enumerate(excerpts)
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