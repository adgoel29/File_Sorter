from sentence_transformers import SentenceTransformer
from Contentreading.pdf_reader import get_all_content
from Contentreading.txtreader import get_all_content_txt
import numpy as np
import hdbscan
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_distances
import os

model = SentenceTransformer('allenai-specter')


def chunk_text(text, chunk_size=200, overlap=50):
    words = text.split()
    chunks = []

    for i in range(0, len(words), chunk_size - overlap):
        chunk = words[i:i + chunk_size]
        if len(chunk) < 10:
            continue
        chunks.append(" ".join(chunk))

    return chunks

def getans(folder_path):

    
    
    file_texts = get_all_content_txt(folder_path)

    if len(file_texts) == 0:
        raise ValueError("No files were read. Check your folder path and reader functions.")

    # print(f"Files loaded: {len(file_texts)}")
    # for name, text in file_texts.items():
        # print(f"  {name}: {len(text)} chars | preview: {text[:80]}")


    chunk_texts = []
    chunk_to_doc = []

    for name, text in file_texts.items():
        chunks = chunk_text(text)

        if len(chunks) == 0:
            chunks = [text] if text.strip() else []

        for chunk in chunks:
            chunk_texts.append(chunk)
            chunk_to_doc.append(name)

    print(f"\nTotal chunks: {len(chunk_texts)}")
    print(f"\nTotal chunks: {chunk_to_doc}")

    if len(chunk_texts) == 0:
        raise ValueError("No chunks created. Your files may be empty or unreadable.")


    embeddings = model.encode(chunk_texts, show_progress_bar=True)
    embeddings = embeddings.astype(np.float64)

    # print(f"Embeddings shape: {embeddings.shape}")


    distance_matrix = cosine_distances(embeddings)
    distance_matrix = distance_matrix.astype(np.float64)
    distance_matrix = (distance_matrix + distance_matrix.T) / 2
    np.fill_diagonal(distance_matrix, 0)

    # -----------------------------
    # 7. Noise diagnostics
    # -----------------------------
    # Tune min_cluster_size based on total chunks:
    # ~50 chunks  -> min_cluster_size=5,  min_samples=2
    # ~100 chunks -> min_cluster_size=8,  min_samples=3
    # ~200 chunks -> min_cluster_size=12, min_samples=4

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=2,
        min_samples=2,
        metric='precomputed'
    )

    labels = clusterer.fit_predict(distance_matrix)

    # -----------------------------------
    # STEP 1: Count clusters per document
    # -----------------------------------

    # Structure:
    # {
    #   "doc1.txt": {0: 120, 1: 50},
    #   "doc2.txt": {2: 200}
    # }
    doc_cluster_count = {}

    for i in range(len(labels)):
        label = labels[i]

        # Ignore noise chunks (cluster = -1)
        if label == -1:
            continue

        doc = chunk_to_doc[i]  # which document this chunk belongs to

        # Weight = number of words in chunk
        weight = len(chunk_texts[i].split())

        # If document not seen before → create empty dict
        if doc not in doc_cluster_count:
            doc_cluster_count[doc] = {}

        # If cluster not seen for this doc → initialize to 0
        if label not in doc_cluster_count[doc]:
            doc_cluster_count[doc][label] = 0

        # Add weight to that cluster
        doc_cluster_count[doc][label] += weight
        
    # -----------------------------------
    # STEP 2: Decide final cluster per doc
    # -----------------------------------

    doc_final_cluster = {}

    for doc in doc_cluster_count:

        cluster_counts = doc_cluster_count[doc]

        # Total weight of all clusters
        total = sum(cluster_counts.values())

        # Find cluster with highest weight
        best_cluster = max(cluster_counts, key=cluster_counts.get)

        # Confidence = how dominant that cluster is
        confidence = cluster_counts[best_cluster] / total

        # If not dominant enough → mark as noise
        if confidence < 0.4:
            doc_final_cluster[doc] = -1
        else:
            doc_final_cluster[doc] = best_cluster

    # print(doc_final_cluster)
    # -----------------------------------
    # STEP 3: Ensure all docs are included
    # -----------------------------------

    for doc in file_texts:
        if doc not in doc_final_cluster:
            doc_final_cluster[doc] = -1


    # -----------------------------------
    # STEP 4: Group documents by cluster
    # -----------------------------------

    # Structure:
    # {
    #   0: ["doc1", "doc5"],
    #   1: ["doc2"],
    #   -1: ["doc3"]
    # }
    final_clusters = {}

    for doc, cluster in doc_final_cluster.items():

        if cluster not in final_clusters:
            final_clusters[cluster] = []

        final_clusters[cluster].append(doc)


    toreturn={
        str(key):[filename for filename in values]
        for key,values in final_clusters.items()
    }



    return toreturn


if __name__=="__main__":
    folder_path = r"C:\Users\aditya\Desktop\filefolder"
    ans=getans(folder_path)
    print(f"the answer is:{ans}")