from transformers import CLIPProcessor, CLIPModel, BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import numpy as np
import hdbscan
from pathlib import Path
import json
from sklearn.preprocessing import normalize
import umap
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from itertools import product


def get_image_clusters(
    image_dir,
    embedding_cache_path,
    caption_cache_path,
    text_embedding_cache_path,
    clustering_mode  = "general",   # "strict" | "medium" | "general"
    noise_assignment = "none",      # "none"   | "soft"   | "cosine"
    clip_weight      = 0.5,
    blip_weight      = 0.5,
    dedup_threshold  = 0.02,
):
    """
    Clusters images and returns a dict keyed by cluster label (str), each containing:
      - "files"     : list of image filenames in that cluster
      - "llm_input" : formatted string of filename + BLIP caption pairs (for LLM naming)

    Noise cluster is returned under key "-1" with llm_input="other".
    Mirrors the structure returned by getans().
    """

    assert abs(clip_weight + blip_weight - 1.0) < 1e-6, \
        "clip_weight + blip_weight must equal 1.0"
    assert clustering_mode in ("strict", "medium", "general"), \
        "clustering_mode must be 'strict', 'medium', or 'general'"
    assert noise_assignment in ("none", "soft", "cosine"), \
        "noise_assignment must be 'none', 'soft', or 'cosine'"

    IMAGE_DIR            = Path(image_dir)
    EMBEDDING_CACHE      = Path(embedding_cache_path)
    CAPTION_CACHE        = Path(caption_cache_path)
    TEXT_EMBEDDING_CACHE = Path(text_embedding_cache_path)

    # ── Mode profiles ──────────────────────────────────────────
    MODE_PROFILES = {
        "strict": dict(
            umap_neighbors  = 10,
            umap_min_dist   = 0.0,
            mcs_fractions   = [0.005, 0.008, 0.012, 0.018, 0.025],
            ms_values       = [1, 2],
            noise_target    = 0.30,
            giant_threshold = 0.05,
            clust_cap       = 300,
        ),
        "medium": dict(
            umap_neighbors  = 15,
            umap_min_dist   = 0.05,
            mcs_fractions   = [0.010, 0.015, 0.020, 0.030, 0.040],
            ms_values       = [1, 2, 3],
            noise_target    = 0.20,
            giant_threshold = 0.08,
            clust_cap       = 150,
        ),
        "general": dict(
            umap_neighbors  = 30,
            umap_min_dist   = 0.1,
            mcs_fractions   = [0.025, 0.040, 0.060, 0.080, 0.100],
            ms_values       = [1, 2, 3, 5],
            noise_target    = 0.10,
            giant_threshold = 0.15,
            clust_cap       = 80,
        ),
    }

    cfg = MODE_PROFILES[clustering_mode]

    # ── Load caches ────────────────────────────────────────────
    embedding_cache      = json.loads(EMBEDDING_CACHE.read_text())      if EMBEDDING_CACHE.exists()      else {}
    caption_cache        = json.loads(CAPTION_CACHE.read_text())        if CAPTION_CACHE.exists()        else {}
    text_embedding_cache = json.loads(TEXT_EMBEDDING_CACHE.read_text()) if TEXT_EMBEDDING_CACHE.exists() else {}

    # ── Collect image paths ────────────────────────────────────
    exts        = ("*.jpg", "*.jpeg", "*.png", "*.webp", "*.bmp")
    image_paths = [p for ext in exts for p in IMAGE_DIR.glob(ext)]

    missing_visual_emb = [p for p in image_paths if p.name not in embedding_cache]
    missing_captions   = [p for p in image_paths if p.name not in caption_cache]
    missing_text_emb   = [p for p in image_paths if p.name not in text_embedding_cache]

    needs_clip = len(missing_visual_emb) > 0 or len(missing_text_emb) > 0
    needs_blip = len(missing_captions) > 0

    # ── Load models ────────────────────────────────────────────
    clip_model, clip_processor = None, None
    blip_model, blip_processor = None, None
    device = "cpu"

    if needs_clip or needs_blip:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if needs_clip:
        print("Loading CLIP...")
        clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        clip_model.eval()
    else:
        print("All visual + text embeddings cached — skipping CLIP load")

    if needs_blip:
        print("Loading BLIP...")
        blip_model     = BlipForConditionalGeneration.from_pretrained(
                             "Salesforce/blip-image-captioning-base").to(device)
        blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        blip_model.eval()
    else:
        print("All captions cached — skipping BLIP load")

    def get_clip():
        nonlocal clip_model, clip_processor, device
        if clip_model is None:
            device         = "cuda" if torch.cuda.is_available() else "cpu"
            print("Loading CLIP for text embeddings...")
            clip_model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
            clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
            clip_model.eval()
        return clip_model, clip_processor

    # ── Embed ──────────────────────────────────────────────────
    embeddings, valid_paths = [], []

    print(f"Processing {len(image_paths)} images...")
    for path in image_paths:
        try:
            key = path.name

            if key in embedding_cache:
                clip_feat_np = np.array(embedding_cache[key], dtype=np.float32)
            else:
                img      = Image.open(path).convert("RGB")
                cm, cp   = get_clip()
                clip_inp = cp(images=img, return_tensors="pt").to(device)
                with torch.no_grad():
                    clip_feat = cm.get_image_features(**clip_inp)
                    clip_feat = clip_feat / clip_feat.norm(dim=-1, keepdim=True)
                clip_feat_np = clip_feat.squeeze().cpu().numpy()
                embedding_cache[key] = clip_feat_np.tolist()
                EMBEDDING_CACHE.write_text(json.dumps(embedding_cache))
                print(f"  [new visual]  {key}")

            if key in caption_cache:
                caption = caption_cache[key]
            else:
                img = Image.open(path).convert("RGB")
                inp = blip_processor(img, return_tensors="pt").to(device)
                with torch.no_grad():
                    out = blip_model.generate(**inp, max_new_tokens=30)
                caption = blip_processor.decode(out[0], skip_special_tokens=True)
                caption_cache[key] = caption
                CAPTION_CACHE.write_text(json.dumps(caption_cache, indent=2))
                print(f"  [new caption] {key} → \"{caption}\"")

            if key in text_embedding_cache:
                txt_feat_np = np.array(text_embedding_cache[key], dtype=np.float32)
            else:
                cm, cp  = get_clip()
                txt_inp = cp(text=caption, return_tensors="pt",
                             padding=True, truncation=True).to(device)
                with torch.no_grad():
                    txt_feat = cm.get_text_features(**txt_inp)
                    txt_feat = txt_feat / txt_feat.norm(dim=-1, keepdim=True)
                txt_feat_np = txt_feat.squeeze().cpu().numpy()
                text_embedding_cache[key] = txt_feat_np.tolist()
                TEXT_EMBEDDING_CACHE.write_text(json.dumps(text_embedding_cache))
                print(f"  [new text]    {key}")

            fused = clip_weight * clip_feat_np + blip_weight * txt_feat_np
            fused = fused / np.linalg.norm(fused)
            embeddings.append(fused)
            valid_paths.append(path)

        except Exception as e:
            print(f"  Skipping {path.name}: {e}")

    embeddings = normalize(np.array(embeddings))
    n          = len(embeddings)
    print(f"\nEmbedded {n} images  (CLIP×{clip_weight} + BLIP×{blip_weight})")

    # ── Deduplicate ────────────────────────────────────────────
    sim_full = cosine_similarity(embeddings).astype(np.float64)
    np.fill_diagonal(sim_full, -1)

    seen, keep_idx, dup_map = set(), [], {}
    for i in range(n):
        if i in seen:
            continue
        keep_idx.append(i)
        for j in range(i + 1, n):
            if j not in seen and (1.0 - sim_full[i, j]) < dedup_threshold:
                seen.add(j)
                dup_map[j] = i

    print(f"Dedup: {n} → {len(keep_idx)} unique, {len(dup_map)} removed")

    uniq_emb   = embeddings[keep_idx]
    uniq_paths = [valid_paths[i] for i in keep_idx]
    n_uniq     = len(keep_idx)

    # ── Dynamic scaling ────────────────────────────────────────
    def scale_mcs(frac, n):
        raw = max(2, int(round(frac * n)))
        return min(raw, max(2, n // 2))

    mcs_values   = sorted(set(scale_mcs(f, n_uniq) for f in cfg["mcs_fractions"]))
    n_neighbors  = min(cfg["umap_neighbors"], n_uniq - 1)
    # n_components = min(10, n_uniq - 1)
    n_components = min(10, max(2, n_uniq // 2))

    print(f"\n{'═'*57}")
    print(f"  CLUSTERING MODE  : {clustering_mode.upper()}")
    print(f"  NOISE ASSIGNMENT : {noise_assignment.upper()}")
    print(f"  Dataset size     : {n_uniq} unique images")
    print(f"  CLIP weight      : {clip_weight}  |  BLIP weight : {blip_weight}")
    print(f"  umap_neighbors   : {n_neighbors}  (requested {cfg['umap_neighbors']})")
    print(f"  umap_min_dist    : {cfg['umap_min_dist']}")
    print(f"  umap_components  : {n_components}")
    print(f"  mcs range        : {mcs_values}  (scaled from fractions)")
    print(f"  noise target     : ≤{int(cfg['noise_target']*100)}%")
    print(f"{'═'*57}\n")

    # ── Diagnostic: raw space ──────────────────────────────────
    s_idx = np.random.choice(n_uniq, min(500, n_uniq), replace=False)
    s_sim = cosine_similarity(uniq_emb[s_idx])
    np.fill_diagonal(s_sim, 0)
    print(f"Before UMAP (sample={len(s_idx)}): "
          f"mean_sim={s_sim.mean():.3f}  p90={np.percentile(s_sim, 90):.3f}")

    # ── UMAP ───────────────────────────────────────────────────
    print(f"\nRunning UMAP (n_neighbors={n_neighbors}, "
          f"min_dist={cfg['umap_min_dist']}, n_components={n_components})...")

    reduced = umap.UMAP(
        n_components = n_components,
        n_neighbors  = n_neighbors,
        min_dist     = cfg["umap_min_dist"],
        metric       = "cosine",
        random_state = 42,
    ).fit_transform(uniq_emb)

    print(f"UMAP done → {reduced.shape[1]}D")

    d_umap = euclidean_distances(reduced[s_idx])
    np.fill_diagonal(d_umap, 0)
    print(f"After UMAP (sample={len(s_idx)}): "
          f"mean_dist={d_umap.mean():.3f}  "
          f"p90={np.percentile(d_umap[d_umap > 0], 90):.3f}")

    # ── Scoring function ───────────────────────────────────────
    def score(labels, n_total, noise_target, giant_threshold, clust_cap):
        noise_r  = list(labels).count(-1) / n_total
        clusters = set(labels) - {-1}
        if not clusters:
            return -999
        sizes      = [list(labels).count(l) for l in clusters]
        singletons = sum(1 for s in sizes if s == 1) / n_total
        n_clust    = len(clusters)
        max_size   = max(sizes)

        excess_noise = max(0, noise_r - noise_target) * 15
        giant_pen    = max(0, max_size / n_total - giant_threshold) * 8
        clust_reward = min(np.log1p(n_clust), np.log1p(clust_cap)) * 2.5

        return (
            -(noise_r * 2)
            -(singletons * 3)
            + clust_reward
            - excess_noise
            - giant_pen
        )

    # ── Grid search HDBSCAN ────────────────────────────────────
    methods = ["leaf", "eom"]

    def valid_ms(mcs, ms_list):
        return [ms for ms in ms_list if ms < mcs] or [1]

    n_configs = sum(len(valid_ms(mcs, cfg["ms_values"])) * len(methods) for mcs in mcs_values)
    print(f"\nGrid searching {n_configs} HDBSCAN configs [{clustering_mode} mode]...")

    best_score, best_labels, best_cfg = -999, None, None

    for mcs, method in product(mcs_values, methods):
        for ms in valid_ms(mcs, cfg["ms_values"]):
            try:
                c = hdbscan.HDBSCAN(
                    min_cluster_size          = mcs,
                    min_samples               = ms,
                    metric                    = "euclidean",
                    cluster_selection_method  = method,
                    cluster_selection_epsilon = 0.0,
                    prediction_data           = False,
                )
                lbs     = c.fit_predict(reduced)
                s       = score(lbs, n_uniq,
                                cfg["noise_target"], cfg["giant_threshold"], cfg["clust_cap"])
                n_clust = len(set(lbs) - {-1})
                noise   = list(lbs).count(-1)
                sizes   = sorted([list(lbs).count(l) for l in set(lbs) - {-1}], reverse=True)
                print(f"  mcs={mcs:<4} ms={ms} {method:<4} → "
                      f"score={s:7.2f}  clusters={n_clust:<4}  "
                      f"noise={noise:<4} ({noise/n_uniq*100:.0f}%)  "
                      f"top3={sizes[:3]}")
                if s > best_score:
                    best_score, best_labels = s, lbs
                    best_cfg = dict(mcs=mcs, ms=ms, method=method)
            except Exception as e:
                print(f"  FAILED mcs={mcs} ms={ms} {method}: {e}")

    if best_labels is None:
        raise RuntimeError("All HDBSCAN configs failed — dataset may be too small or too uniform.")

    labels     = best_labels
    n_clusters = len(set(labels) - {-1})
    n_noise    = list(labels).count(-1)
    print(f"\n{'─'*57}")
    print(f"  Best config  : mcs={best_cfg['mcs']}  ms={best_cfg['ms']}  method={best_cfg['method']}")
    print(f"  Clusters     : {n_clusters}")
    print(f"  Noise (raw)  : {n_noise} ({n_noise/n_uniq*100:.1f}%)")
    print(f"  Score        : {best_score:.2f}")
    print(f"{'─'*57}")

    # ── Noise assignment ───────────────────────────────────────
    final_labels = labels.copy()

    if noise_assignment != "none" and n_noise > 0:
        noise_mask  = labels == -1
        cluster_ids = sorted(set(labels) - {-1})

        if noise_assignment == "soft":
            print(f"\nNoise assignment: SOFT (nearest UMAP centroid)...")
            centroids  = np.array([reduced[labels == cid].mean(axis=0) for cid in cluster_ids])
            noise_vecs = reduced[noise_mask]
            dists      = euclidean_distances(noise_vecs, centroids)
            nearest    = np.argmin(dists, axis=1)
            final_labels[noise_mask] = np.array([cluster_ids[i] for i in nearest])

        elif noise_assignment == "cosine":
            print(f"\nNoise assignment: COSINE (fused 512D embeddings)...")
            centroids_fused  = normalize(
                np.array([uniq_emb[labels == cid].mean(axis=0) for cid in cluster_ids])
            )
            noise_vecs_fused = uniq_emb[noise_mask]
            sim_mat          = cosine_similarity(noise_vecs_fused, centroids_fused)
            nearest          = np.argmax(sim_mat, axis=1)
            final_labels[noise_mask] = np.array([cluster_ids[i] for i in nearest])

        n_remaining = (final_labels == -1).sum()
        print(f"  Noise before : {n_noise}  →  after : {n_remaining}  "
              f"(reassigned {n_noise - n_remaining})")

    else:
        print("\nNoise assignment: NONE — keeping noise folder as-is")

    # ── Label map (propagate duplicates) ──────────────────────
    orig_idx_to_label = {keep_idx[i]: final_labels[i] for i in range(n_uniq)}
    for dup_idx, src_idx in dup_map.items():
        orig_idx_to_label[dup_idx] = orig_idx_to_label[src_idx]

    # ── Summary ────────────────────────────────────────────────
    all_labels = [orig_idx_to_label[i] for i in range(n)]
    print("\nFinal cluster sizes:")
    for label in sorted(set(all_labels)):
        name  = f"cluster_{int(label):03d}" if label != -1 else "noise"
        count = all_labels.count(label)
        bar   = "█" * min(count // max(1, n // 200), 50)
        print(f"  {name:14s}: {count:4d}  {bar}")

    final_noise = all_labels.count(-1)
    print(f"\n  Total images   : {n}")
    print(f"  Total clusters : {len(set(all_labels) - {-1})}")
    print(f"  Noise remaining: {final_noise} ({final_noise/n*100:.1f}%)")

    # ── Top similar pairs audit ────────────────────────────────
    sim_audit = cosine_similarity(uniq_emb)
    np.fill_diagonal(sim_audit, -1)
    pairs = sorted(
        [(sim_audit[i, j], uniq_paths[i].name, uniq_paths[j].name)
         for i in range(n_uniq) for j in range(i + 1, n_uniq)],
        reverse=True,
    )[:5]
    print("\nTop 5 most similar pairs (fused space):")
    for s, a, b in pairs:
        print(f"  {s:.3f}  {a} ↔ {b}")

    # ── Build and return cluster data (mirrors getans toreturn) 
    toreturn = {}
    cluster_groups = {}
    for i, path in enumerate(valid_paths):
        label = orig_idx_to_label[i]
        cluster_groups.setdefault(int(label), []).append(path)

    for label, paths in cluster_groups.items():
        filenames = [p.name for p in paths]

        if label == -1:
            toreturn["-1"] = {
                "files": filenames,
                "llm_input": "other"
            }
            continue

        caption_lines = []
        for p in paths:
            caption = caption_cache.get(p.name, "[no caption]")
            caption_lines.append(f"- {p.name}: {caption}")

        caption_block = "\n".join(caption_lines)

        toreturn[str(label)] = {
            "files": filenames,
            "llm_input": (
                f"Images in group:\n{caption_block}\n\n"
                f"Based on these image captions, suggest a short descriptive folder name."
            )
        }

    return toreturn


# ── Example usage ──────────────────────────────────────────────
if __name__ == "__main__":
    result = get_image_clusters(
        image_dir                  = r"C:\Users\aditya\Desktop\fileimag",
        embedding_cache_path       = r"C:\Users\aditya\Downloads\embedding_cacheok.json",
        caption_cache_path         = r"C:\Users\aditya\Downloads\caption_cacheok.json",
        text_embedding_cache_path  = r"C:\Users\aditya\Downloads\text_embedding_cacheok.json",
        clustering_mode            = "general",
        noise_assignment           = "none",
        clip_weight                = 0.5,
        blip_weight                = 0.5,
    )

    # Preview output
    for label, info in sorted(result.items(), key=lambda x: int(x[0])):
        print(f"\n[{'noise' if label == '-1' else f'cluster_{int(label):03d}'}]  "
              f"({len(info['files'])} images)")
        if label != "-1":
            for line in info["llm_input"].split("\n")[1:4]:
                print(f"  {line}")