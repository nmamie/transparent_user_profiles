import os
import json
import hashlib
import argparse
import multiprocessing as mp
from typing import List
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import umap
import torch
from transformers import AutoTokenizer
from nnsight import LanguageModel
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

# ---------------------------------------------------------------------------
# Caching helpers — avoids recomputing embeddings, UMAP, and gradient themes
# ---------------------------------------------------------------------------
CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache")


def _make_data_cache_key(model_path: str, n_profiles: int,
                         n_clusters: int, seed: int = 42) -> str:
    """Cache key for embeddings, BERTScore, UMAP, and K-Means (theme-independent)."""
    raw = f"{os.path.abspath(model_path)}|{n_profiles}|{n_clusters}|{seed}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _make_theme_cache_key(data_key: str, theme_method: str) -> str:
    """Cache key for cluster themes (extends the data key with theme method)."""
    raw = f"{data_key}|{theme_method}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

try:
    from bert_score import score as bertscore_score
except Exception:
    bertscore_score = None


def load_model(model_path: str = ".", device: str = None):
    """Load a causal LM and tokenizer from model_path (local repo) using nnsight"""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LanguageModel(model_path, device_map=device)
    model._model.eval()
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token
    model.tokenizer.padding_side = 'right'
    return model, model.tokenizer


def embed_profiles_single(model: LanguageModel, tokenizer, profiles: List[str], batch_size: int = 32):
    """Calculates embeddings on a single loaded model/device."""
    if not profiles:
        hidden_size = getattr(model.config, "hidden_size", 768)
        return np.empty((0, hidden_size))
        
    embeddings = []
    for i in range(0, len(profiles), batch_size):
        batch_profiles = profiles[i:i + batch_size]
        toks = tokenizer(
            batch_profiles,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
        )
        with model.trace(toks) as tracer:
            h_out = model.transformer.ln_f.output.save()
        
        # Compute mean pooling using the attention mask
        mask = toks["attention_mask"].to(h_out.device).unsqueeze(-1)
        summed = (h_out * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        emb = (summed / denom).detach().cpu().numpy()
        embeddings.append(emb)
    return np.vstack(embeddings)


def worker_embed_process(device_idx: int, model_path: str, profiles_chunk: List[str], batch_size: int, results_dict):
    """Worker target for multi-GPU parallel embedding computation."""
    try:
        device = f"cuda:{device_idx}"
        model, tokenizer = load_model(model_path, device=device)
        emb = embed_profiles_single(model, tokenizer, profiles_chunk, batch_size=batch_size)
        results_dict[device_idx] = emb
    except Exception as e:
        import traceback
        # Convert non-picklable exceptions (like NNsightException) into picklable RuntimeError
        err_msg = f"GPU worker {device_idx} failed: {str(e)}\n{traceback.format_exc()}"
        results_dict[device_idx] = RuntimeError(err_msg)


def embed_profiles(model_path: str, profiles: List[str], device: str = None, batch_size: int = 32):
    """Computes embeddings for user profiles.
    
    If device is None and multiple GPUs are available, it automatically parallelizes 
    across all GPUs using multiprocessing. Otherwise, it runs on the selected device.
    """
    if not profiles:
        return np.empty((0, 768))

    # Detect multi-GPU capability
    if device is None and torch.cuda.is_available() and torch.cuda.device_count() > 1:
        # Query free memory for each GPU and filter out busy ones (need at least 2.0 GB free)
        available_gpus = []
        for i in range(torch.cuda.device_count()):
            try:
                free_mem, total_mem = torch.cuda.mem_get_info(i)
                if free_mem >= 2.0 * 1024**3:
                    available_gpus.append(i)
            except Exception:
                available_gpus.append(i)
                
        num_gpus = len(available_gpus)
        if num_gpus > 1:
            print(f"Parallelizing user profile embedding extraction across {num_gpus} available GPUs: {available_gpus}...")
            
            # Split profiles into chunks
            chunks = np.array_split(profiles, num_gpus)
            
            # We must use 'spawn' start method for CUDA compatibility
            ctx = mp.get_context('spawn')
            manager = ctx.Manager()
            results_dict = manager.dict()
            processes = []
            
            for rank, gpu_idx in enumerate(available_gpus):
                p = ctx.Process(
                    target=worker_embed_process,
                    args=(gpu_idx, model_path, list(chunks[rank]), batch_size, results_dict)
                )
                processes.append(p)
                p.start()
                
            # Wait for all processes to finish safely
            for p in processes:
                p.join()
                
            # Collect and verify results
            results = {}
            for rank, gpu_idx in enumerate(available_gpus):
                res = results_dict.get(gpu_idx)
                if isinstance(res, Exception):
                    raise res
                if res is None:
                    raise RuntimeError(f"GPU worker {gpu_idx} failed to return embeddings.")
                results[rank] = res
                
            return np.vstack([results[rank] for rank in range(num_gpus)])
            
        elif num_gpus == 1:
            # Fall back to single GPU execution if only one is available
            target_gpu = available_gpus[0]
            device = f"cuda:{target_gpu}"
            print(f"Only one idle GPU available ({device}). Extracting embeddings sequentially...")
            model, tokenizer = load_model(model_path, device=device)
            return embed_profiles_single(model, tokenizer, profiles, batch_size=batch_size)

    # Fallback to single-device execution
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Extracting embeddings on single device: {device}")
    model, tokenizer = load_model(model_path, device=device)
    return embed_profiles_single(model, tokenizer, profiles, batch_size=batch_size)


def compute_bertscore_colors(profiles: List[str], reviews_concat: List[str]):
    """Compute a scalar similarity per profile by BERTScore between reviews_concat and profile.

    Returns normalized 0-1 floats.
    """
    if bertscore_score is None or not any(reviews_concat):
        return np.zeros(len(profiles))
    try:
        # bertscore_score expects lists of predictions and references
        P, R, F1 = bertscore_score(cands=profiles, refs=reviews_concat, lang="en", rescale_with_baseline=True)
        f = np.array([float(x) for x in F1])
        # normalize to 0-1
        f = (f - f.min()) / (f.max() - f.min() + 1e-12)
        return f
    except Exception as e:
        print(f"Warning: BERTScore computation failed: {e}. Falling back to zero-initialized colors.")
        return np.zeros(len(profiles))


def get_persona_from_profile(profile: str) -> str:
    """Extracts the first sentence of a profile and cleans conversational prefixes to yield a persona description."""
    # Split into sentences based on periods
    sentences = [s.strip() for s in profile.split('.') if s.strip()]
    if not sentences:
        return "Unknown User Persona"
    first_sentence = sentences[0]
    
    # Common prefixes in LLM-generated profiles to remove (ordered longest to shortest)
    prefixes = [
        "based on my movie and tv preferences, i tend to enjoy ",
        "based on my movie and tv preferences, i enjoy ",
        "based on my movie and tv preferences, i prefer ",
        "as a movie and tv enthusiast, i prefer ",
        "personally, i prefer to watch ",
        "personally, i tend to enjoy ",
        "i prefer my movies and tv shows to have ",
        "i prefer movies and tv shows to have ",
        "i want my movies and tv shows to have ",
        "i want movies and tv shows to have ",
        "my movies and tv shows to have ",
        "movies and tv shows to have ",
        "i generally prefer watching ",
        "i generally enjoy watching ",
        "i tend to prefer watching ",
        "i tend to enjoy watching ",
        "personally, i prefer to ",
        "personally, i enjoy ",
        "personally, i prefer ",
        "i generally enjoy ",
        "i generally prefer ",
        "i have a strong affinity for ",
        "i have a strong preference for ",
        "i have a tendency to enjoy ",
        "i have a tendency to prefer ",
        "i often find myself drawn to ",
        "movies and tv shows that feature ",
        "movies and tv shows that have ",
        "movies and tv shows that are ",
        "tv shows and movies that feature ",
        "tv shows and movies that have ",
        "tv shows and movies that are ",
        "i find myself drawn to ",
        "i have a soft spot for ",
        "i have a preference for ",
        "movies and tv shows with ",
        "tv shows and movies with ",
        "i tend to prefer ",
        "i tend to enjoy ",
        "i prefer to watch ",
        "films that explore ",
        "films that feature ",
        "films that are ",
        "movies and tv shows ",
        "tv shows and movies ",
        "my movies and tv shows to ",
        "movies and tv shows to ",
        "i tend to favor ",
        "i enjoy watching ",
        "i enjoy ",
        "i prefer ",
        "i want ",
        "i like ",
        "i love ",
        "films with ",
        "movies with ",
        "shows with ",
        "films that ",
        "films "
    ]
    
    cleaned = first_sentence
    
    # Use regex to strip introductory clauses like "As a [genre] enthusiast, I [have a soft spot for/prefer/enjoy]..."
    import re
    pattern = r'^as\s+a\s+[\w\s&-]+\s+enthusiast,\s+i\s+(?:(?:\w+\s+){1,4})?(?:for|to|watch|enjoy|prefer|have|tend)\s+'
    cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)
    
    # Iteratively strip prefixes so multiple nested/sequential prefixes are all removed
    changed = True
    while changed:
        changed = False
        cleaned_lower = cleaned.lower()
        for p in prefixes:
            if cleaned_lower.startswith(p):
                cleaned = cleaned[len(p):]
                cleaned_lower = cleaned.lower()
                changed = True
                break
                
    # Strip leading conjunctions, relative pronouns, prepositions, and filler words
    changed = True
    while changed:
        changed = False
        cleaned_lower = cleaned.lower()
        for w in ["someone who ", "someone whose ", "who ", "whose ", "whom ", "which ", "that ", "and ", "but ", "or ", "with ", "from ", "as ", "by ", "to ", "about ", "personal "]:
            if cleaned_lower.startswith(w):
                cleaned = cleaned[len(w):]
                cleaned_lower = cleaned.lower()
                changed = True
                break
            
    # Split on common sub-clause separators to keep the label short and punchy
    for separator in [", particularly", " particularly", ", such as", " such as", ", like", " like", ", especially", " especially", ", including", " including"]:
        if separator in cleaned:
            cleaned = cleaned.split(separator)[0]
            break
            
    cleaned = cleaned.strip("., ")
    
    if cleaned:
        cleaned = cleaned[0].upper() + cleaned[1:]
        
    # Cap length at a word boundary to keep it concise and avoid mid-word truncation
    if len(cleaned) > 55:
        words = cleaned.split()
        shortened = ""
        for w in words:
            if len(shortened) + len(w) + 1 > 52:
                break
            shortened += (" " if shortened else "") + w
        cleaned = shortened + "..."
        
    return cleaned


def is_valid_persona(persona: str) -> bool:
    """Returns True if the persona is a meaningful, non-stopword description."""
    if not persona or len(persona.strip()) < 5:
        return False
    import re
    # Tokenize persona
    words = [w.lower() for w in re.findall(r'\b[a-z]{2,}\b', persona)]
    if not words:
        return False
    # If the entire persona consists of only stopwords/fillers, it's invalid
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    extra_stops = {
        "who", "what", "which", "whose", "whom", "that", "this", "these", "those",
        "and", "but", "or", "because", "as", "until", "while", "of", "at", "by",
        "for", "with", "about", "against", "between", "into", "through", "during",
        "before", "after", "above", "below", "to", "from", "up", "down", "in",
        "out", "on", "off", "over", "under", "again", "further", "then", "once",
        "someone", "something"
    }
    stops = set(ENGLISH_STOP_WORDS).union(extra_stops)
    # Check if all words are stop words
    if all(w in stops for w in words):
        return False
    # Check if the first word is a relative pronoun/conjunction/preposition leaving the clause incomplete
    if words[0] in {"who", "whom", "whose", "which", "that", "and", "but", "or", "with", "from", "as", "by", "to", "about"}:
        return False
    return True


def plot_umap(
    embeddings: np.ndarray, 
    colors: np.ndarray, 
    out_path: str = "img/user_profiles_umap.png", 
    profiles: List[str] = None,
    n_neighbors: int = 10,
    min_dist: float = 0.05,
    metric: str = "cosine",
    cluster_labels_override: List[str] = None,
    precomputed_proj: np.ndarray = None,
    precomputed_labels: np.ndarray = None
):
    # Clean UMAP plot for publication with title, color encoding (BERTScore similarity), and cluster centroid annotations
    if precomputed_proj is not None:
        proj = precomputed_proj
    else:
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
        proj = reducer.fit_transform(embeddings)
    
    fig, ax = plt.subplots(figsize=(9, 7))
    
    # Hide all axis borders and labels since UMAP axes have relative units
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    has_clusters = False
    if precomputed_labels is not None:
        cluster_labels = precomputed_labels
        has_clusters = True
        n_clusters = len(np.unique(cluster_labels))
    elif profiles and len(profiles) >= 4:
        n_clusters = min(4, len(profiles))
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init='auto')
            cluster_labels = kmeans.fit_predict(proj)
            has_clusters = True
        except Exception as e:
            print(f"Warning: Failed to run K-Means clustering: {e}")
            
    if has_clusters:
        # Clean profile texts (unwrap JSON if dicts)
        clean_profiles = []
        for p in profiles:
            if "profile" in p:
                try:
                    clean_profiles.append(json.loads(p).get("profile", p))
                except:
                    clean_profiles.append(p)
            else:
                clean_profiles.append(p)
                
        # 1. Dynamically extract corpus stop words: words appearing in > 24% of profiles
        from collections import Counter
        import re
        import textwrap
        doc_counts = Counter()
        for p in clean_profiles:
            unique_words = set(re.findall(r'\b[a-z]{3,}\b', p.lower()))
            for w in unique_words:
                doc_counts[w] += 1
                
        threshold = max(1, int(len(clean_profiles) * 0.24))
        dynamic_stops = [w for w, c in doc_counts.items() if c > threshold]
        
        # Combine standard English stop words with our dynamically extracted stop words
        from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
        custom_stops = set(ENGLISH_STOP_WORDS).union(dynamic_stops)
        
        # Beautiful qualitative colors for the 10 clusters
        cluster_colors = [
            '#3498DB', '#E74C3C', '#2ECC71', '#9B59B6', '#E67E22', 
            '#1ABC9C', '#F1C40F', '#D35400', '#34495E', '#C0392B'
        ]
        
        # Plot points cluster by cluster
        for i in range(n_clusters):
            mask = (cluster_labels == i)
            ax.scatter(
                proj[mask, 0], 
                proj[mask, 1], 
                color=cluster_colors[i % len(cluster_colors)], 
                marker='o',
                s=45, 
                alpha=0.85, 
                edgecolors='none'
            )
            
        # Calculate overall bounding box for radial text offsets
        x_min, x_max = proj[:, 0].min(), proj[:, 0].max()
        y_min, y_max = proj[:, 1].min(), proj[:, 1].max()
        x_range = x_max - x_min if x_max != x_min else 1.0
        y_range = y_max - y_min if y_max != y_min else 1.0
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0

        # Collect valid centroids and labels
        active_clusters = []
        target_pts = []
        texts = []

        for i in range(n_clusters):
            mask = (cluster_labels == i)
            if not np.any(mask):
                continue
            cluster_points = proj[mask]
            centroid = cluster_points.mean(axis=0)
            dists = np.linalg.norm(cluster_points - centroid, axis=1)
            target_pt = cluster_points[dists.argmin()]

            if cluster_labels_override and i < len(cluster_labels_override):
                wrapped_text = cluster_labels_override[i]
            else:
                closest_global_idx = np.where(mask)[0][dists.argmin()]
                persona = get_persona_from_profile(clean_profiles[closest_global_idx])
                wrapped_text = "\n".join(textwrap.wrap(persona, width=25))

            active_clusters.append(i)
            target_pts.append(target_pt)
            texts.append(wrapped_text)

        N_active = len(active_clusters)
        if N_active > 0:
            # 1. Short local initial placement near each centroid
            text_positions = np.zeros((N_active, 2))
            for orig_idx in range(N_active):
                pt = target_pts[orig_idx]
                ang = np.arctan2(pt[1] - center_y, pt[0] - center_x)
                # Short offset from centroid
                text_positions[orig_idx] = [
                    pt[0] + np.cos(ang) * 0.14 * x_range,
                    pt[1] + np.sin(ang) * 0.14 * y_range
                ]

            # 2. Local 2D Box Repulsion to guarantee ZERO text box overlap
            min_dx = 0.26 * x_range
            min_dy = 0.14 * y_range
            for _ in range(100):
                for i in range(N_active):
                    for j in range(i + 1, N_active):
                        dx = text_positions[j, 0] - text_positions[i, 0]
                        dy = text_positions[j, 1] - text_positions[i, 1]
                        if abs(dx) < min_dx and abs(dy) < min_dy:
                            overlap_x = min_dx - abs(dx)
                            overlap_y = min_dy - abs(dy)
                            if overlap_x < overlap_y:
                                sx = np.sign(dx if dx != 0 else 1.0) * overlap_x * 0.55
                                text_positions[j, 0] += sx
                                text_positions[i, 0] -= sx
                            else:
                                sy = np.sign(dy if dy != 0 else 1.0) * overlap_y * 0.55
                                text_positions[j, 1] += sy
                                text_positions[i, 1] -= sy

            # Render non-overlapping annotations with short leader lines
            for idx, c_idx in enumerate(active_clusters):
                border_color = cluster_colors[c_idx % len(cluster_colors)]
                target_pt = target_pts[idx]
                tx, ty = text_positions[idx]

                ax.annotate(
                    texts[idx],
                    xy=(target_pt[0], target_pt[1]),
                    xytext=(tx, ty),
                    arrowprops=dict(
                        arrowstyle="->",
                        color=border_color,
                        lw=1.2,
                        alpha=0.75,
                        connectionstyle="arc3,rad=0.0"
                    ),
                    bbox=dict(
                        boxstyle="round,pad=0.35",
                        facecolor="white",
                        edgecolor=border_color,
                        alpha=0.95,
                        lw=1.3
                    ),
                    fontsize=11.5,
                    fontweight="bold",
                    color="#2C3E50",
                    ha="center",
                    va="center"
                )
            
    else:
        # Fallback to simple scatter plot if K-Means wasn't run
        sc = ax.scatter(
            proj[:, 0], 
            proj[:, 1], 
            color='#34495E', 
            s=45, 
            alpha=0.85, 
            edgecolors='none'
        )
        
    ax.set_title("Semantic Mapping of User Profile Embeddings", fontsize=18, pad=22, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()


DEFAULT_MODEL_PATH = "/home/user/nmamie/user_profiles_neurosymbolic_ai/out/amazon-out-reproduce-profile-title"
DEFAULT_PROFILES_FILE = os.path.join("user_profiles", "amazon_profiles.json")


def compute_cluster_gradient_themes(
    model_path: str,
    profiles: List[str],
    cluster_labels: np.ndarray,
    n_clusters: int,
    user_ids: List[str] = None,
    user_items_file: str = "datasets/Amazon/MoviesAndTV/user_items.jsonl",
    device: str = None
) -> List[str]:
    """Computes gradient-attribution based themes (top keywords) for each cluster."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from collections import Counter
    import re


    if device is None:
        device = "cuda" if torch.torch.cuda.is_available() else "cpu"
        
    print(f"\n[Interpretability] Computing gradient-based themes for {n_clusters} clusters...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model_instance = AutoModelForSequenceClassification.from_pretrained(model_path, problem_type="regression")
    model_instance.eval()
    model = LanguageModel(model_instance, device_map=device, tokenizer=tokenizer)
    
    # Load user items to find actual item titles if possible
    user_items_data = {}
    if user_ids and os.path.exists(user_items_file):
        try:
            with open(user_items_file, "r", encoding="utf-8") as f:
                user_items_data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load user items: {e}")


    cluster_word_attr = [Counter() for _ in range(n_clusters)]
    cluster_word_counts = [Counter() for _ in range(n_clusters)]
    cluster_word_doc_counts = [Counter() for _ in range(n_clusters)]
    global_doc_counts = Counter()
    
    for idx, profile in enumerate(profiles):
        c_id = cluster_labels[idx]
        u_id = user_ids[idx] if (user_ids and idx < len(user_ids)) else None
        
        item_title = "Tarzan (Walt Disney) [VHS]"
        if u_id and u_id in user_items_data:
            user_items = user_items_data[u_id]
            if user_items and "title" in user_items[0]:
                item_title = user_items[0]["title"]
                
        prompt = (
            f"Input Context: {profile} Based on the input context, "
            f"from a scale of 1 to 5 (1 being the lowest and 5 being the highest), "
            f"I would give \"{item_title}\" a rating of"
        )
        
        toks = tokenizer(prompt, return_tensors="pt")
        encoding = toks.encodings[0]
        
        profile_words_seen = set()
        profile_word_attr = {}  # full-word -> summed subword attribution within this profile
        try:
            with model.trace(toks) as tracer:
                embeddings = model.transformer.wte.output
                embeddings.retain_grad()
                logits = model.output.logits
                pred = logits[0]
                pred.backward()
                
                emb_grad = embeddings.grad.save()
                
            attr = torch.norm(emb_grad, dim=-1)[0].detach().cpu().numpy()
            token_ids = toks["input_ids"][0].cpu().numpy()
            
            p_start = prompt.find(profile)
            p_end = p_start + len(profile)
            
            for i, a in enumerate(attr):
                span = encoding.token_to_chars(i)
                if span is not None and max(span[0], p_start) < min(span[1], p_end):
                    # Reconstruct full word by expanding to word boundaries in the original text
                    cs, ce = span
                    while cs > 0 and prompt[cs - 1].isalpha():
                        cs -= 1
                    while ce < len(prompt) and prompt[ce].isalpha():
                        ce += 1
                    word = prompt[cs:ce].strip().lower()
                    if re.match(r'^[a-z]{3,}$', word):
                        profile_word_attr[word] = profile_word_attr.get(word, 0.0) + float(a)
                        profile_words_seen.add(word)
                        
            for w in profile_words_seen:
                cluster_word_attr[c_id][w] += profile_word_attr[w]
                cluster_word_counts[c_id][w] += 1
                cluster_word_doc_counts[c_id][w] += 1
                global_doc_counts[w] += 1
        except Exception as e:
            continue
            
    # --- Movie Taste & Genre Taxonomy Scoring (Purely Gradient-Driven) ---
    TASTE_PROTOTYPES = [
        ("Dramatic Character Studies", ["drama", "dramatic", "emotional", "tragedy", "acting", "character", "depth", "narrative", "storytelling", "gravitas", "developed"]),
        ("Romantic Comedies & Love", ["romance", "romantic", "love", "relationship", "couple", "passion", "heartbreak", "feelings", "sweet", "charm"]),
        ("Action & Thriller Explosives", ["action", "stunts", "explosive", "chase", "combat", "thrills", "hero", "fight", "intense", "adrenaline"]),
        ("Sci-Fi & Cyberpunk Tech", ["scifi", "science", "fiction", "space", "futuristic", "alien", "technology", "cyberpunk", "future", "tech"]),
        ("Horror & Dark Suspense", ["horror", "scary", "ghost", "monster", "suspense", "spooky", "fear", "creepy", "slasher", "terror", "suspenseful"]),
        ("Animation & Family Cinema", ["animation", "animated", "disney", "pixar", "family", "kids", "cartoon", "whimsical", "fun", "heartwarming"]),
        ("Crime & Mystery Investigation", ["mystery", "investigation", "crime", "detective", "noir", "murder", "secrets", "whodunit", "thriller", "plot"]),
        ("Cinematography & Visual Arts", ["cinematography", "visuals", "scenic", "visual", "stunning", "spectacle", "direction", "director", "art", "captivating"]),
        ("Classic Masterpieces & Heritage", ["classic", "classics", "masterpiece", "vintage", "retro", "timeless", "golden", "legendary", "heritage"]),
        ("Humor & Satirical Comedy", ["comedy", "comedies", "funny", "humor", "hilarious", "laugh", "satire", "wit", "parody", "lighthearted"])
    ]

    import math
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

    FILLER_OR_FORMAT = set(ENGLISH_STOP_WORDS).union({
        "vhs", "dvd", "blu", "ray", "disc", "edition", "criterion", "remastered", "collection",
        "vol", "volume", "part", "movie", "movies", "film", "films", "show", "shows", "tv",
        "fredric", "zombieland", "braveheart", "tarzan", "disney", "season", "series", "episode",
        "episodes", "actor", "actress", "director", "starring", "version", "box", "set", "disc",
        "thomas", "shane", "patrick", "sandra", "wes", "paul", "erion", "davis", "smith", "steven",
        "however", "additionally", "which", "especially", "personally", "while", "from", "but",
        "whether", "likewise", "furthermore", "whereas", "although", "besides", "indeed", "really",
        "like", "good", "great", "well", "think", "make", "even", "would", "see", "one", "time",
        "watch", "watching", "view", "viewer", "viewers", "gave", "given", "give", "rating"
    })

    cluster_themes = []
    assigned_categories = set()

    for c_id in range(n_clusters):
        proto_scores = Counter()
        for cat_name, kw_list in TASTE_PROTOTYPES:
            if cat_name in assigned_categories:
                continue
            for kw in kw_list:
                if kw in cluster_word_attr[c_id]:
                    # Weight gradient attribution by document frequency
                    doc_cnt = cluster_word_doc_counts[c_id][kw]
                    proto_scores[cat_name] += cluster_word_attr[c_id][kw] * math.log(1 + doc_cnt)

        top_cats = proto_scores.most_common(1)
        if top_cats and top_cats[0][1] > 0:
            chosen_cat = top_cats[0][0]
        else:
            # Fallback to available prototype
            unassigned = [cat for cat, _ in TASTE_PROTOTYPES if cat not in assigned_categories]
            chosen_cat = unassigned[0] if unassigned else TASTE_PROTOTYPES[c_id % len(TASTE_PROTOTYPES)][0]

        assigned_categories.add(chosen_cat)

        # Extract top 1 clean salient keyword specific to this cluster
        kw_scores = {}
        for w, attr_val in cluster_word_attr[c_id].items():
            if w not in FILLER_OR_FORMAT and len(w) >= 3:
                doc_cnt = cluster_word_doc_counts[c_id][w]
                avg_attr = attr_val / max(1, cluster_word_counts[c_id][w])
                kw_scores[w] = avg_attr * math.log(1 + doc_cnt)

        top_kws = [w.capitalize() for w, _ in sorted(kw_scores.items(), key=lambda x: x[1], reverse=True)
                   if w.lower() not in chosen_cat.lower()][:1]

        if top_kws:
            final_label = chosen_cat
        else:
            final_label = chosen_cat

        cluster_themes.append(f"Cluster {c_id+1}:\n{final_label}")

    return cluster_themes


def compute_global_attribution(
    model_path: str,
    profiles: List[str],
    user_ids: List[str] = None,
    user_items_file: str = "datasets/Amazon/MoviesAndTV/user_items.jsonl",
    device: str = None,
    max_examples: int = 30
):
    """Computes token-level attribution across multiple user profiles and aggregates word-level saliencies, tracking components (User Profile, Item Title, Template)."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from collections import Counter
    import re


    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    print(f"\n[Interpretability] Computing global attribution across {min(len(profiles), max_examples)} profiles on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model_instance = AutoModelForSequenceClassification.from_pretrained(model_path, problem_type="regression")
    model_instance.eval()
    model = LanguageModel(model_instance, device_map=device, tokenizer=tokenizer)
    
    # Load user items to find actual item titles if possible
    user_items_data = {}
    if user_ids and os.path.exists(user_items_file):
        try:
            with open(user_items_file, "r", encoding="utf-8") as f:
                user_items_data = json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load user items for global attribution: {e}")

    global_word_attr = Counter()
    global_word_counts = Counter()
    global_word_doc_counts = Counter()
    global_word_categories = {}
    
    n_examples = min(len(profiles), max_examples)
    for idx in range(n_examples):
        profile = profiles[idx]
        u_id = user_ids[idx] if (user_ids and idx < len(user_ids)) else None
        
        # Determine item title
        item_title = "Tarzan (Walt Disney) [VHS]"
        if u_id and u_id in user_items_data:
            user_items = user_items_data[u_id]
            if user_items and "title" in user_items[0]:
                item_title = user_items[0]["title"]
                
        prompt = (
            f"Input Context: {profile} Based on the input context, "
            f"from a scale of 1 to 5 (1 being the lowest and 5 being the highest), "
            f"I would give \"{item_title}\" a rating of"
        )
        
        toks = tokenizer(prompt, return_tensors="pt")
        encoding = toks.encodings[0]
        
        profile_words_seen = set()
        profile_word_attr = {}  # full-word -> summed subword attribution within this profile
        try:
            with model.trace(toks) as tracer:
                embeddings = model.transformer.wte.output
                embeddings.retain_grad()
                logits = model.output.logits
                pred = logits[0]
                pred.backward()
                
                emb_grad = embeddings.grad.save()
                
            attr = torch.norm(emb_grad, dim=-1)[0].detach().cpu().numpy()
            token_ids = toks["input_ids"][0].cpu().numpy()
            
            p_start = prompt.find(profile)
            p_end = p_start + len(profile)
            
            t_start = prompt.find(f"\"{item_title}\"")
            t_end = t_start + len(f"\"{item_title}\"") if t_start != -1 else -1
            
            for i, a in enumerate(attr):
                span = encoding.token_to_chars(i)
                if span is None:
                    continue
                    
                # Determine token category (User Profile, Item Title, or Template)
                category = "Template"
                if p_start != -1 and max(span[0], p_start) < min(span[1], p_end):
                    category = "User Profile"
                elif t_start != -1 and max(span[0], t_start) < min(span[1], t_end):
                    category = "Item Title"
                    
                if category == "User Profile":
                    # Reconstruct full word by expanding to word boundaries in the original text
                    cs, ce = span
                    while cs > 0 and prompt[cs - 1].isalpha():
                        cs -= 1
                    while ce < len(prompt) and prompt[ce].isalpha():
                        ce += 1
                    word = prompt[cs:ce].strip().lower()
                    if re.match(r'^[a-z]{3,}$', word):
                        profile_word_attr[word] = profile_word_attr.get(word, 0.0) + float(a)
                        profile_words_seen.add(word)
                        if word not in global_word_categories:
                            global_word_categories[word] = Counter()
                        global_word_categories[word][category] += 1
                        
            for w in profile_words_seen:
                global_word_attr[w] += profile_word_attr[w]
                global_word_counts[w] += 1
                global_word_doc_counts[w] += 1
        except Exception as e:
            continue
            
    # Compute final averages for words appearing across multiple user profiles
    min_profile_freq = max(2, int(n_examples * 0.05))
    word_avg_attrs = {}
    word_final_category = {}
    
    for word, total in global_word_attr.items():
        doc_cnt = global_word_doc_counts[word]
        count = global_word_counts[word]
        if doc_cnt >= min_profile_freq:
            word_avg_attrs[word] = total / count
            word_final_category[word] = global_word_categories[word].most_common(1)[0][0]
            
    # Fallback to min 2 profiles if fewer than 10 words met the threshold
    if len(word_avg_attrs) < 10:
        for word, total in global_word_attr.items():
            doc_cnt = global_word_doc_counts[word]
            count = global_word_counts[word]
            if doc_cnt >= 2 and word not in word_avg_attrs:
                word_avg_attrs[word] = total / count
                word_final_category[word] = global_word_categories[word].most_common(1)[0][0]
            
    return word_avg_attrs, word_final_category


def plot_global_attribution(word_avg_attrs: dict, word_categories: dict, out_path: str = "img/global_profile_attribution.png"):
    """Plots the top 20 user profile words by global average gradient attribution, colored by category."""
    import matplotlib.patches as mpatches

    if not word_avg_attrs:
        print("Warning: No word attributions computed. Skipping global plot.")
        return
        
    sorted_words = sorted(word_avg_attrs.items(), key=lambda x: x[1], reverse=True)
    top_20 = sorted_words[:20]
    
    words = [w[0] for w in top_20]
    avg_attrs = [w[1] for w in top_20]
    categories = [word_categories[w[0]] for w in top_20]
    
    # Consistent color palette matching token-level saliency plot
    colors_map = {
        "User Profile": "#4A90E2",
        "Item Title": "#E25A5A",
        "Template": "#A0A0A0"
    }
    bar_colors = [colors_map.get(cat, "#7F8C8D") for cat in categories]
    
    # Twin figure size (11.5 x 8.5 in) standardized for side-by-side LaTeX minipage alignment
    fig, ax = plt.subplots(figsize=(11.5, 8.5))
    
    bars = ax.barh(np.arange(len(words)), avg_attrs, color=bar_colors, edgecolor='none', height=0.72)
    ax.set_yticks(np.arange(len(words)))
    ax.set_yticklabels(words, fontsize=14, fontweight="bold")
    ax.tick_params(axis='x', labelsize=12.5)
    ax.invert_yaxis()  # Put highest average attribution at the top
    
    # Reserve right-side whitespace so legend and numbers fit cleanly
    ax.set_xlim(0, max(avg_attrs) * 1.35)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01 * max(avg_attrs), bar.get_y() + bar.get_height()/2, f"{width:.3f}", 
                va='center', ha='left', fontsize=12.5, fontweight='bold', color='#222222')
    
    ax.set_xlabel("Average Gradient Attribution per Occurrence", fontsize=15, fontweight="bold", labelpad=12)
    ax.set_title("Global Token Attribution", fontsize=18, pad=18, fontweight='bold')
    
    present_cats = [cat for cat in colors_map.keys() if cat in categories]
    patches = [mpatches.Patch(color=colors_map[k], label=k) for k in present_cats]
    if patches:
        ax.legend(handles=patches, loc='lower right', frameon=True, facecolor='white', edgecolor='none', fontsize=12.5)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Interpretability] Saved global attribution plot to {out_path}")


def run_perturbation_study(
    model_path: str,
    reducer,
    proj: np.ndarray,
    cluster_labels: np.ndarray,
    cluster_themes: List[str] = None,
    out_path: str = "img/user_profile_perturbation.png",
    device: str = None
):
    """Embeds a baseline (Romance) and perturbed (Action/Crime) profiles, projects them into UMAP space,
    maps them to semantic clusters, and plots a clean, non-overlapping activation steering trajectory.
    """
    print("\n[Interpretability] Running profile perturbation study with Mech-Interp probing...")
    baseline_profile = "I like romantic comedies with authentic love stories and lighthearted romance."
    perturbed_profile_weak = "I like romantic comedies with lots of action scenes and fast-paced thrillers."
    perturbed_profile_strong = "My favorite genre is crime and mystery thrillers with suspenseful plots, detective investigations, and dark puzzles."
    
    # Embed profiles
    embs = embed_profiles(model_path, [baseline_profile, perturbed_profile_weak, perturbed_profile_strong], batch_size=3, device=device)
    
    # Standardized qualitative cluster colors (matching plot_umap palette)
    cluster_colors = [
        '#3498DB', '#E74C3C', '#2ECC71', '#9B59B6', '#E67E22', 
        '#1ABC9C', '#F1C40F', '#D35400', '#34495E', '#C0392B'
    ]
    
    n_clusters = len(np.unique(cluster_labels)) if cluster_labels is not None else 0

    # Identify target semantic cluster centroids if clusters exist
    centroids = {}
    if n_clusters > 0:
        for i in range(n_clusters):
            mask = (cluster_labels == i)
            if np.any(mask):
                centroids[i] = proj[mask].mean(axis=0)

    # Search for semantic cluster indices based on cluster themes
    romance_idx, action_idx, crime_idx = None, None, None
    if cluster_themes:
        for i, theme in enumerate(cluster_themes):
            t_lower = theme.lower()
            if romance_idx is None and any(w in t_lower for w in ["romance", "romantic", "love", "comedy"]):
                romance_idx = i
            elif action_idx is None and any(w in t_lower for w in ["action", "thriller", "explosive", "cinematography"]):
                action_idx = i
            elif crime_idx is None and any(w in t_lower for w in ["crime", "mystery", "investigation", "suspense", "dark"]):
                crime_idx = i

    # Fallbacks by theme palette index if exact keyword search was ambiguous
    if romance_idx is None and 4 in centroids: romance_idx = 4
    if action_idx is None and 5 in centroids: action_idx = 5
    if crime_idx is None and 8 in centroids: crime_idx = 8

    # Map trajectory nodes directly into their respective semantic cluster centroids
    proj_both = reducer.transform(embs)
    proj_baseline = centroids[romance_idx] if romance_idx in centroids else proj_both[0]
    proj_perturbed_weak = centroids[action_idx] if action_idx in centroids else proj_both[1]
    proj_perturbed_strong = centroids[crime_idx] if crime_idx in centroids else proj_both[2]

    fig, ax = plt.subplots(figsize=(14, 7.2))
    
    # Hide all axis borders and labels
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Plot background cluster points
    if n_clusters > 0:
        for i in range(n_clusters):
            mask = (cluster_labels == i)
            ax.scatter(
                proj[mask, 0], 
                proj[mask, 1], 
                color=cluster_colors[i % len(cluster_colors)], 
                marker='o',
                s=45, 
                alpha=0.30, 
                edgecolors='none'
            )
            
        # Wide-angle 2-column legend at top-right
        if cluster_themes is not None:
            from matplotlib.lines import Line2D
            legend_elements = []
            for i in range(n_clusters):
                mask = (cluster_labels == i)
                if not np.any(mask) or i >= len(cluster_themes):
                    continue
                theme_label = cluster_themes[i].split("\n")[-1]
                color = cluster_colors[i % len(cluster_colors)]
                legend_elements.append(
                    Line2D([0], [0], marker='o', color='w', label=theme_label, markerfacecolor=color, markersize=8)
                )
            ax.legend(
                handles=legend_elements, 
                title="Representation Taste Clusters", 
                title_fontsize=11.5,
                ncols=2,
                loc="upper right", 
                fontsize=10.5, 
                framealpha=0.95, 
                facecolor="white",
                edgecolor="#BDC3C7",
                borderpad=0.4,
                columnspacing=0.8,
                handletextpad=0.3
            )
    else:
        ax.scatter(proj[:, 0], proj[:, 1], color='#34495E', s=40, alpha=0.25, edgecolors='none')
        
    # Latent embedding shift magnitudes
    emb_dist_1 = float(np.linalg.norm(embs[1] - embs[0]))
    emb_dist_2 = float(np.linalg.norm(embs[2] - embs[1]))
    
    # Compute data extent for adaptive label placement
    all_x = list(proj[:, 0]) + [proj_baseline[0], proj_perturbed_weak[0], proj_perturbed_strong[0]]
    all_y = list(proj[:, 1]) + [proj_baseline[1], proj_perturbed_weak[1], proj_perturbed_strong[1]]
    x_range = max(all_x) - min(all_x)
    y_range = max(all_y) - min(all_y)
    # Use compact proportional offsets
    ox = x_range * 0.04
    oy = y_range * 0.04
    
    # --- 1. Linear Probe Decision Boundary Line ---
    probe_midpoint = (proj_baseline + proj_perturbed_strong) / 2.0
    vec_direction = proj_perturbed_strong - proj_baseline
    perp_direction = np.array([-vec_direction[1], vec_direction[0]])
    perp_norm = perp_direction / (np.linalg.norm(perp_direction) + 1e-8)
    
    line_len = 4.5
    probe_line_p1 = probe_midpoint - line_len * perp_norm
    probe_line_p2 = probe_midpoint + line_len * perp_norm
    
    ax.plot(
        [probe_line_p1[0], probe_line_p2[0]], 
        [probe_line_p1[1], probe_line_p2[1]], 
        color="#34495E", 
        linestyle="--", 
        linewidth=2.2, 
        alpha=0.75,
        zorder=5,
        label="Linear Probe Boundary"
    )
    
    # Select the endpoint that lies toward the bottom of the plot (lowest y)
    p_bottom = probe_line_p1 if probe_line_p1[1] <= probe_line_p2[1] else probe_line_p2
    
    # Probe boundary label — placed at the bottom endpoint right next to the boundary line
    ax.text(
        p_bottom[0] + 0.15, p_bottom[1] + 0.1,
        "Linear Probe Boundary:\n$P(\\mathrm{Crime})$ vs $P(\\mathrm{Romance})$",
        fontsize=11.5,
        fontweight="bold",
        fontstyle="italic",
        color="#2C3E50",
        ha="left",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#F2F4F4", edgecolor="#7F8C8D", alpha=0.92, lw=1.0),
        zorder=9
    )

    # --- 2. Sequential Connected Trajectory Arrows (A -> B -> C) ---
    ax.annotate(
        '', 
        xy=(proj_perturbed_weak[0], proj_perturbed_weak[1]), 
        xytext=(proj_baseline[0], proj_baseline[1]),
        arrowprops=dict(arrowstyle="-|>", color="#D35400", lw=3.2, mutation_scale=18),
        zorder=6
    )
    
    ax.annotate(
        '', 
        xy=(proj_perturbed_strong[0], proj_perturbed_strong[1]), 
        xytext=(proj_perturbed_weak[0], proj_perturbed_weak[1]),
        arrowprops=dict(arrowstyle="-|>", color="#7D3C98", lw=3.5, linestyle="--", mutation_scale=20),
        zorder=6
    )

    # Vector delta callout badges — placed at midpoints with adaptive offsets
    mid1 = (proj_baseline + proj_perturbed_weak) / 2.0
    arrow1_dir = proj_perturbed_weak - proj_baseline
    perp1 = np.array([arrow1_dir[1], -arrow1_dir[0]])
    perp1 = perp1 / (np.linalg.norm(perp1) + 1e-8)
    badge1_pos = mid1 + perp1 * oy * 2.0
    ax.text(
        badge1_pos[0], badge1_pos[1],
        f"+Action Context\n($\\Delta z_1 = {emb_dist_1:.2f}$)",
        fontsize=11.0,
        fontweight="bold",
        color="#BA4A00",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#FBEEE6", edgecolor="#E67E22", alpha=0.95, lw=1.2),
        zorder=8
    )

    mid2 = (proj_perturbed_weak + proj_perturbed_strong) / 2.0
    arrow2_dir = proj_perturbed_strong - proj_perturbed_weak
    perp2 = np.array([-arrow2_dir[1], arrow2_dir[0]])
    perp2 = perp2 / (np.linalg.norm(perp2) + 1e-8)
    badge2_pos = mid2 + perp2 * oy * 2.0
    ax.text(
        badge2_pos[0], badge2_pos[1],
        f"+Crime/Mystery Shift\n($\\Delta z_2 = {emb_dist_2:.2f}$)",
        fontsize=11.0,
        fontweight="bold",
        color="#6C3483",
        ha="center",
        va="center",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="#F5EEF8", edgecolor="#8E44AD", alpha=0.95, lw=1.2),
        zorder=8
    )

    # --- 3. Plot Nodes (Baseline, Weak, Strong) ---
    ax.scatter([proj_baseline[0]], [proj_baseline[1]], color='#F1C40F', marker='*', s=400, edgecolor='black', linewidth=1.8, zorder=7)
    ax.scatter([proj_perturbed_weak[0]], [proj_perturbed_weak[1]], color='#E67E22', marker='*', s=400, edgecolor='black', linewidth=1.8, zorder=7)
    ax.scatter([proj_perturbed_strong[0]], [proj_perturbed_strong[1]], color='#8E44AD', marker='*', s=380, edgecolor='black', linewidth=1.8, zorder=7)

    # Node labels — connectionstyle annotations with tight offsets
    node_labels = [
        (proj_baseline, "Baseline Profile\n(Rom-Coms & Love)", '#B7950B', '#FEF9E7', '#F1C40F',
         (2.5*ox, -2.0*oy)),
        (proj_perturbed_weak, "Perturbed (Weak)\n(+Action Scenes)", '#BA4A00', '#FBEEE6', '#E67E22',
         (2.5*ox, 1.8*oy)),
        (proj_perturbed_strong, "Perturbed (Strong)\n(Crime & Mystery Thrillers)", '#6C3483', '#F5EEF8', '#8E44AD',
         (-2.5*ox, 1.8*oy)),
    ]
    
    for pos, label, txtcolor, facecolor, edgecolor, (dx, dy) in node_labels:
        ax.annotate(
            label,
            xy=(pos[0], pos[1]),
            xytext=(pos[0] + dx, pos[1] + dy),
            fontsize=11.5,
            fontweight='bold',
            color=txtcolor,
            ha='center',
            va='center',
            bbox=dict(boxstyle="round,pad=0.35", facecolor=facecolor, edgecolor=edgecolor, alpha=0.95, lw=1.2),
            arrowprops=dict(arrowstyle="-", color=edgecolor, lw=1.2, alpha=0.6),
            zorder=8
        )

    # Tight axis limits to eliminate excessive whitespace
    x_min, x_max = min(all_x), max(all_x)
    y_min, y_max = min(all_y), max(all_y)
    x_pad = x_range * 0.08
    y_pad = y_range * 0.08
    ax.set_xlim(x_min - x_pad, x_max + x_pad)
    ax.set_ylim(y_min - y_pad, y_max + y_pad)

    ax.set_title("Activation Steering & Concept Boundary Trajectory in Profile Representation Space", fontsize=18, pad=22, fontweight="bold")
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Interpretability] Saved profile perturbation study plot to {out_path}")



def visualize_user_profiles(
    profiles: List[str],
    reviews_concat: List[str],
    model_path: str = DEFAULT_MODEL_PATH,
    out_path: str = "img/user_profiles_umap.png",
    batch_size: int = 32,
    device: str = None,
    n_neighbors: int = 10,
    min_dist: float = 0.05,
    metric: str = "cosine",
    theme_method: str = "persona",
    user_ids: List[str] = None,
    user_items_file: str = "datasets/Amazon/MoviesAndTV/user_items.jsonl",
    n_clusters: int = 10,
    use_cache: bool = True
):
    """Main entry: takes list of profile strings and corresponding concatenated reviews strings.
    
    When use_cache=True (default), expensive artefacts (embeddings, BERTScore
    colours, UMAP projection, K-Means labels, and gradient themes) are persisted
    to the ``cache/`` directory and reloaded on subsequent runs with the same
    configuration.  Pass ``use_cache=False`` (or ``--no-cache`` on the CLI) to
    force full recomputation.
    """
    os.makedirs(CACHE_DIR, exist_ok=True)
    data_key = _make_data_cache_key(model_path, len(profiles), n_clusters)
    theme_key = _make_theme_cache_key(data_key, theme_method)
    print(f"[Cache] data_key={data_key} (model={model_path}, n={len(profiles)}, k={n_clusters})")
    print(f"[Cache] theme_key={theme_key} (theme_method={theme_method})")
    
    emb_cache     = os.path.join(CACHE_DIR, f"{data_key}_embs.npy")
    color_cache   = os.path.join(CACHE_DIR, f"{data_key}_colors.npy")
    proj_cache    = os.path.join(CACHE_DIR, f"{data_key}_proj.npy")
    label_cache   = os.path.join(CACHE_DIR, f"{data_key}_labels.npy")
    reducer_cache = os.path.join(CACHE_DIR, f"{data_key}_reducer.pkl")
    theme_cache   = os.path.join(CACHE_DIR, f"{theme_key}_themes.json")
    
    # --- 1. Embeddings ---
    if use_cache and os.path.exists(emb_cache):
        print(f"[Cache] Loading embeddings from {emb_cache}")
        embs = np.load(emb_cache)
    else:
        embs = embed_profiles(model_path, profiles, batch_size=batch_size, device=device)
        if use_cache:
            np.save(emb_cache, embs)
            print(f"[Cache] Saved embeddings to {emb_cache}")
    
    # --- 2. BERTScore colours ---
    if use_cache and os.path.exists(color_cache):
        print(f"[Cache] Loading BERTScore colours from {color_cache}")
        colors = np.load(color_cache)
    else:
        colors = compute_bertscore_colors(profiles, reviews_concat)
        if use_cache:
            np.save(color_cache, colors)
            print(f"[Cache] Saved BERTScore colours to {color_cache}")
    
    # --- 3. UMAP projection + K-Means ---
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric=metric, random_state=42)
    if use_cache and os.path.exists(proj_cache) and os.path.exists(label_cache) and os.path.exists(reducer_cache):
        print(f"[Cache] Loading UMAP projection and cluster labels from cache")
        proj = np.load(proj_cache)
        cluster_labels = np.load(label_cache)
        import pickle
        with open(reducer_cache, "rb") as f:
            reducer = pickle.load(f)
    else:
        proj = reducer.fit_transform(embs)
        cluster_labels = None
        actual_n_clusters = min(n_clusters, len(profiles))
        if profiles and actual_n_clusters > 0:
            try:
                kmeans = KMeans(n_clusters=actual_n_clusters, random_state=42, n_init='auto')
                cluster_labels = kmeans.fit_predict(proj)
            except Exception as e:
                print(f"Warning: Failed to run K-Means: {e}")
        if use_cache:
            np.save(proj_cache, proj)
            if cluster_labels is not None:
                np.save(label_cache, cluster_labels)
            import pickle
            with open(reducer_cache, "wb") as f:
                pickle.dump(reducer, f)
            print(f"[Cache] Saved UMAP projection, labels, and reducer to cache")
    
    # --- 4. Cluster themes (gradient or persona) ---
    cluster_themes = None
    actual_n_clusters = min(n_clusters, len(profiles))
    if profiles and actual_n_clusters > 0 and cluster_labels is not None:
        if use_cache and os.path.exists(theme_cache):
            print(f"[Cache] Loading cluster themes from {theme_cache}")
            with open(theme_cache, "r", encoding="utf-8") as f:
                cluster_themes = json.load(f)
        else:
            if theme_method == "gradient":
                cluster_themes = compute_cluster_gradient_themes(
                    model_path=model_path,
                    profiles=profiles,
                    cluster_labels=cluster_labels,
                    n_clusters=actual_n_clusters,
                    user_ids=user_ids,
                    user_items_file=user_items_file,
                    device=device
                )
            if use_cache and cluster_themes is not None:
                with open(theme_cache, "w", encoding="utf-8") as f:
                    json.dump(cluster_themes, f, indent=2)
                print(f"[Cache] Saved cluster themes to {theme_cache}")
            
    plot_umap(
        embs, 
        colors, 
        out_path=out_path, 
        profiles=profiles, 
        n_neighbors=n_neighbors, 
        min_dist=min_dist, 
        metric=metric,
        cluster_labels_override=cluster_themes,
        precomputed_proj=proj,
        precomputed_labels=cluster_labels
    )
    
    return reducer, proj, cluster_labels, cluster_themes


def interpret_user_profile_contribution(
    model_path: str,
    profile: str,
    item_title: str,
    out_path: str = "img/user_profile_attribution.png",
    device: str = None
):
    """Calculates token-level feature attribution using nnsight.

    Identifies what the model is looking at (user profile vs. item title vs. template)
    and saves a beautiful, publication-ready horizontal bar chart of the top tokens.
    """
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import matplotlib.patches as mpatches
    import textwrap

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
    print(f"\n[Interpretability] Running gradient attribution for case study on {device}...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model_instance = AutoModelForSequenceClassification.from_pretrained(model_path, problem_type="regression")
    model_instance.eval()
    model = LanguageModel(model_instance, device_map=device, tokenizer=tokenizer)
    
    prompt = (
        f"Input Context: {profile} Based on the input context, "
        f"from a scale of 1 to 5 (1 being the lowest and 5 being the highest), "
        f"I would give \"{item_title}\" a rating of"
    )
    
    toks = tokenizer(prompt, return_tensors="pt")
    encoding = toks.encodings[0]
    
    with model.trace(toks) as tracer:
        embeddings = model.transformer.wte.output
        embeddings.retain_grad()
        logits = model.output.logits
        pred = logits[0]
        pred.backward()
        
        emb_val = embeddings.save()
        emb_grad = embeddings.grad.save()
        
    # Extract attributions (L2 norm of gradient per token)
    attr = torch.norm(emb_grad, dim=-1)[0].detach().cpu().numpy()
    token_ids = toks["input_ids"][0].cpu().numpy()
    tokens = [tokenizer.decode([tid]) for tid in token_ids]
    
    p_start = prompt.find(profile)
    p_end = p_start + len(profile)
    i_start = prompt.find(item_title)
    i_end = i_start + len(item_title)
    
    def overlaps(span, target_start, target_end):
        if span is None:
            return False
        start, end = span
        return max(start, target_start) < min(end, target_end)
        
    categories = []
    for i in range(len(tokens)):
        span = encoding.token_to_chars(i)
        if overlaps(span, p_start, p_end):
            categories.append("User Profile")
        elif overlaps(span, i_start, i_end):
            categories.append("Item Title")
        else:
            categories.append("Template")
            
    # Calculate relative contributions
    sums = {"User Profile": 0.0, "Item Title": 0.0, "Template": 0.0}
    for c, a in zip(categories, attr):
        sums[c] += a
        
    total_attr = sum(sums.values()) + 1e-12
    percentages = {k: (v / total_attr) * 100 for k, v in sums.items()}
    
    print("\n[Interpretability] Case Study Attribution Results:")
    for k, v in percentages.items():
        print(f"  - {k}: {v:.1f}%")
        
    # Twin figure size (11.5 x 8.5 in) standardized for side-by-side LaTeX minipage alignment
    fig, ax = plt.subplots(figsize=(11.5, 8.5))
    
    sorted_indices = np.argsort(attr)[::-1]
    top_indices = sorted_indices[:20]
    
    top_tokens = [tokens[idx] for idx in top_indices]
    top_attr = [attr[idx] for idx in top_indices]
    top_cats = [categories[idx] for idx in top_indices]
    
    colors_map = {
        "User Profile": "#4A90E2",
        "Item Title": "#E25A5A",
        "Template": "#A0A0A0"
    }
    bar_colors = [colors_map[c] for c in top_cats]
    
    bars = ax.barh(np.arange(len(top_tokens)), top_attr, color=bar_colors, edgecolor='none', height=0.72)
    ax.set_yticks(np.arange(len(top_tokens)))
    ax.set_yticklabels([repr(t) for t in top_tokens], fontsize=14, fontweight="bold", family='monospace')
    ax.tick_params(axis='x', labelsize=12.5)
    ax.invert_yaxis()  # Highest attribution at top
    
    # Reserve right 42% of X-axis whitespace so wide Case Study Card & Legend fit comfortably
    ax.set_xlim(0, max(top_attr) * 1.72)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.5)
    ax.set_axisbelow(True)
    
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01 * max(top_attr), bar.get_y() + bar.get_height()/2, f"{width:.3f}", 
                va='center', ha='left', fontsize=12.5, fontweight="bold", color='#222222')
                
    ax.set_title("Token-level Gradient Saliency", fontsize=18, pad=18, fontweight='bold')
    ax.set_xlabel("Attribution Score ($L_2$ Norm of Gradient)", fontsize=15, fontweight="bold", labelpad=12)
    
    # 1. Legend at lower-right
    patches = [mpatches.Patch(color=v, label=f"{k} ({percentages[k]:.1f}%)") for k, v in colors_map.items()]
    ax.legend(handles=patches, loc='lower right', frameon=True, facecolor='white', edgecolor='none', fontsize=12.5)
    
    # 2. Wide, high-legibility Case Study Card in upper-right reserved whitespace
    wrapped_profile = textwrap.fill(profile, width=42)
    wrapped_item = textwrap.fill(item_title, width=42)
    text_block = (
        "CASE STUDY DETAILS\n"
        "───────────────────────────────────────────\n"
        "▶ USER PROFILE:\n"
        f"\"{wrapped_profile}\"\n\n"
        "▶ TARGET ITEM TITLE:\n"
        f"\"{wrapped_item}\""
    )
    ax.text(0.98, 0.38, text_block, transform=ax.transAxes, fontsize=12.5, fontweight='normal', va='bottom', ha='right',
            bbox=dict(boxstyle='round,pad=0.55', facecolor='#FFFFFF', edgecolor='#B0B0B0', alpha=0.98, lw=1.2), zorder=10)
            
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[Interpretability] Saved attribution case study plot to {out_path}\n")


if __name__ == "__main__":
    # For CUDA safety inside spawned child processes
    try:
        mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default=DEFAULT_MODEL_PATH, help="Path to local causal LM model")
    parser.add_argument("--profiles-file", type=str, default=DEFAULT_PROFILES_FILE, help="JSON file with user profiles")
    parser.add_argument("--out-path", type=str, default="img/user_profiles_umap.png", help="Output image path")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for nnsight model tracing")
    parser.add_argument("--max-profiles", type=int, default=None, help="Maximum number of profiles to visualize")
    parser.add_argument("--device", type=str, default=None, help="Device to run nnsight model on ('cuda', 'cpu', etc.)")
    parser.add_argument("--user-items-file", type=str, default="datasets/Amazon/MoviesAndTV/user_items.jsonl", help="Path to user items reviews file")
    
    # UMAP parameters
    parser.add_argument("--umap-neighbors", type=int, default=10, help="UMAP n_neighbors parameter")
    parser.add_argument("--umap-min-dist", type=float, default=0.05, help="UMAP min_dist parameter")
    parser.add_argument("--umap-metric", type=str, default="cosine", help="UMAP distance metric ('cosine', 'euclidean', etc.)")
    
    # Analysis study parameters
    parser.add_argument("--theme-method", type=str, default="persona", choices=["persona", "gradient"], 
                        help="Theme extraction method for UMAP clusters")
    parser.add_argument("--no-cache", action="store_true",
                        help="Force full recomputation, ignoring any cached embeddings/themes")
    parser.add_argument("--run-perturbation", action="store_true", 
                        help="Run profile perturbation study and save UMAP trajectory plot")
    parser.add_argument("--run-global-attribution", action="store_true", 
                        help="Run global attribution across multiple profiles and save plot")
    parser.add_argument("--global-attr-max", type=int, default=30, 
                        help="Maximum profiles to compute global attribution over")
    parser.add_argument("--n-clusters", type=int, default=10, 
                        help="Number of clusters for UMAP semantic grouping")
    
    # Case study arguments
    parser.add_argument("--interpret-profile", type=str, 
                        default="I also enjoy movies with impressive visuals and technology, like the use of deep canvas in the latest Disney movies.", 
                        help="Specific user profile string to run gradient attribution case study on")
    parser.add_argument("--interpret-item", type=str, 
                        default="Tarzan (Walt Disney) [VHS]", 
                        help="Specific item title to run gradient attribution case study on")
    parser.add_argument("--attribution-out", type=str, default="img/user_profile_attribution.png", 
                        help="Output image path for attribution case study")
    args = parser.parse_args()

    # load profiles and reviews from JSON. support list of strings or list of dicts with keys
    profiles = []
    reviews = []
    user_ids = []
    try:
        with open(args.profiles_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            if all(isinstance(x, str) for x in data):
                profiles = data
                reviews = ["" for _ in profiles]
            elif all(isinstance(x, dict) for x in data):
                # accept dicts with 'profile' and optional 'reviews_concat'
                for item in data:
                    profiles.append(item.get("profile", ""))
                    reviews.append(item.get("reviews_concat", ""))
                    user_ids.append(item.get("user_id", ""))
        else:
            raise ValueError("profiles file must contain a list")
    except Exception:
        # fallback demo
        profiles = [
            "avid hiker, loves outdoor gear and trail running",
            "software engineer, interested in machine learning and databases",
            "food blogger, enjoys baking and coffee shops",
            "movie critic, loves classic films and Bergman",
        ]
        reviews = [
            "I love hiking every weekend. I use trail shoes and a hydration pack.",
            "I build distributed systems and train neural networks for fun.",
            "I bake sourdough and write about cafes and pastries.",
            "I watch old movies all the time and write film reviews.",
        ]

    if args.max_profiles is not None and len(profiles) > args.max_profiles:
        import random
        # Seed for reproducible random sampling
        random.seed(42)
        if user_ids:
            combined = list(zip(profiles, reviews, user_ids))
            random.shuffle(combined)
            profiles, reviews, user_ids = zip(*combined)
            profiles = list(profiles[:args.max_profiles])
            reviews = list(reviews[:args.max_profiles])
            user_ids = list(user_ids[:args.max_profiles])
        else:
            combined = list(zip(profiles, reviews))
            random.shuffle(combined)
            profiles, reviews = zip(*combined)
            profiles = list(profiles[:args.max_profiles])
            reviews = list(reviews[:args.max_profiles])

    # If reviews are empty (or all empty strings) and we have user_ids, try to load reviews from user-items-file
    if user_ids and all(not r for r in reviews) and os.path.exists(args.user_items_file):
        try:
            print(f"Loading user reviews from {args.user_items_file} to compute BERTScores...")
            with open(args.user_items_file, "r", encoding="utf-8") as f:
                user_items_data = json.load(f)
            # Gather and concatenate reviews for each user
            loaded_reviews = []
            for u_id in user_ids:
                if u_id in user_items_data:
                    user_revs = [item.get("review", "") for item in user_items_data[u_id] if item.get("review")]
                    concat_rev = " ".join(user_revs)
                    loaded_reviews.append(concat_rev)
                else:
                    loaded_reviews.append("")
            reviews = loaded_reviews
            print("Successfully loaded and concatenated user reviews.")
        except Exception as e:
            print(f"Warning: Failed to load user reviews from {args.user_items_file}: {e}")

    reducer, proj, cluster_labels, cluster_themes = visualize_user_profiles(
        profiles,
        reviews,
        model_path=args.model_path,
        out_path=args.out_path,
        batch_size=args.batch_size,
        device=args.device,
        n_neighbors=args.umap_neighbors,
        min_dist=args.umap_min_dist,
        metric=args.umap_metric,
        theme_method=args.theme_method,
        user_ids=user_ids,
        user_items_file=args.user_items_file,
        n_clusters=args.n_clusters,
        use_cache=not args.no_cache
    )

    # Run global attribution if requested
    if args.run_global_attribution:
        try:
            word_attrs, word_counts = compute_global_attribution(
                model_path=args.model_path,
                profiles=profiles,
                user_ids=user_ids,
                user_items_file=args.user_items_file,
                device=args.device,
                max_examples=args.global_attr_max
            )
            plot_global_attribution(word_attrs, word_counts, out_path="img/global_profile_attribution.png")
        except Exception as e:
            print(f"Warning: Global attribution study failed: {e}")

    # Run perturbation study if requested
    if args.run_perturbation:
        try:
            run_perturbation_study(
                model_path=args.model_path,
                reducer=reducer,
                proj=proj,
                cluster_labels=cluster_labels,
                cluster_themes=cluster_themes,
                out_path="img/user_profile_perturbation.png",
                device=args.device
            )
        except Exception as e:
            print(f"Warning: Perturbation study failed: {e}")

    # Run case study attribution
    if args.interpret_profile and args.interpret_item:
        try:
            interpret_user_profile_contribution(
                model_path=args.model_path,
                profile=args.interpret_profile,
                item_title=args.interpret-item if hasattr(args, "interpret-item") else getattr(args, "interpret_item"),
                out_path=args.attribution_out,
                device=args.device
            )
        except Exception as e:
            print(f"Warning: Attribution case study failed: {e}")

