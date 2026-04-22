import re
import json
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def normalize_text(x):
    if pd.isna(x):
        return x
    x = str(x).strip()
    x = re.sub(r"\s+", " ", x)
    return x


def normalize_category(cat):
    if pd.isna(cat):
        return cat
    cat = normalize_text(cat)
    parts = [p.strip() for p in cat.split("_")] if "_" in str(cat) else [p.strip() for p in cat.split("-")]
    parts = [p.capitalize() if p else p for p in parts]
    return "-".join(parts)


def infer_season_from_month(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4]:
        return "Spring"
    elif month in [5, 6, 7]:
        return "Summer"
    elif month in [8, 9, 10]:
        return "Autumn"
    return "LateAutumn"


def week_of_month(dt):
    return ((dt.day - 1) // 7) + 1


def cosine_sim(vec1, vec2):
    if vec1 is None or vec2 is None:
        return 0.0

    v1 = np.array(vec1, dtype=float).reshape(1, -1)
    v2 = np.array(vec2, dtype=float).reshape(1, -1)

    if np.allclose(v1, 0) or np.allclose(v2, 0):
        return 0.0

    return float(cosine_similarity(v1, v2)[0][0])


def mean_pool_vectors(vectors):
    valid = [np.array(v, dtype=float) for v in vectors if v is not None]
    if len(valid) == 0:
        return None
    return np.mean(valid, axis=0)


def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)