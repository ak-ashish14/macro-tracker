"""
One-time script: builds a FAISS index from foods.json.

Run from the backend/ directory:
    python build_index.py

Outputs:
    foods_index.faiss   – FAISS flat inner-product index (cosine after L2 norm)
    foods_meta.json     – food metadata aligned with index row positions
"""

import json
import os
import sys
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

FOODS_JSON = os.path.join(os.path.dirname(__file__), '..', 'foods.json')
INDEX_PATH = os.path.join(os.path.dirname(__file__), 'foods_index.faiss')
META_PATH  = os.path.join(os.path.dirname(__file__), 'foods_meta.json')
MODEL_NAME = 'all-MiniLM-L6-v2'


def make_embedding_text(food: dict) -> str:
    """
    Build a rich natural-language description for each food so the embedding
    captures both the dish identity and its macronutrient profile.
    The query will be the user's raw ingredient list, so we want the food text
    to surface ingredient-like keywords (drawn from the dish name) alongside
    nutritional context.
    """
    kcal = round(food['up'] * 4 + food['uc'] * 4 + food['uf'] * 9, 1)
    return (
        f"{food['n']}. "
        f"Served as 1 {food['u']}. "
        f"Per serving: {food['up']}g protein, {food['uc']}g carbs, "
        f"{food['uf']}g fat, {kcal} kcal."
    )


def build():
    # ── Load foods ────────────────────────────────────────────────────────
    if not os.path.exists(FOODS_JSON):
        print(f"ERROR: {FOODS_JSON} not found. Run from the macro-tracker root.")
        sys.exit(1)

    with open(FOODS_JSON) as f:
        foods = json.load(f)
    print(f"Loaded {len(foods)} foods from foods.json")

    # ── Embed ─────────────────────────────────────────────────────────────
    print(f"Loading embedding model '{MODEL_NAME}' …")
    model = SentenceTransformer(MODEL_NAME)

    texts = [make_embedding_text(food) for food in foods]
    print("Encoding food descriptions …")
    embeddings = model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        convert_to_numpy=True,
    ).astype('float32')

    # Normalise to unit length so inner product == cosine similarity
    faiss.normalize_L2(embeddings)

    # ── Build FAISS index ─────────────────────────────────────────────────
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)   # inner product (cosine after normalisation)
    index.add(embeddings)
    print(f"FAISS index: {index.ntotal} vectors, dim={dim}")

    faiss.write_index(index, INDEX_PATH)
    print(f"Saved → {INDEX_PATH}")

    with open(META_PATH, 'w') as f:
        json.dump(foods, f, separators=(',', ':'))
    print(f"Saved → {META_PATH}")
    print("Done.")


if __name__ == '__main__':
    build()
