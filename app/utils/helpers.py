import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def normalize_text(value):
    if pd.isna(value):
        return ""

    value = str(value).strip()
    value = re.sub(r"\s+", " ", value)

    return value


def load_json(path):
    path = Path(path)

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_pickle(path):
    path = Path(path)

    with open(path, "rb") as f:
        return pickle.load(f)


def infer_season_from_month(month):
    month = int(month)

    if month in [11, 12, 1, 2]:
        return "Winter"

    if month in [3, 4, 5, 6]:
        return "Summer"

    return "Rainy"


def season_to_label(season):
    mapping = {
        "Winter": 1,
        "Summer": 2,
        "Rainy": 3
    }

    return mapping.get(str(season).strip(), 0)


def timeslot_to_label(timeslot):
    mapping = {
        "Morning": 1,
        "Noon": 2,
        "Afternoon": 3,
        "Evening": 4,
        "Night": 5
    }

    return mapping.get(str(timeslot).strip(), 0)

def infer_timeslot_from_hour(hour):
    hour = int(hour)

    if 6 <= hour <= 11:
        return "Morning"

    if 12 <= hour <= 13:
        return "Noon"

    if 14 <= hour <= 16:
        return "Afternoon"

    if 17 <= hour <= 20:
        return "Evening"

    return "Night"


def month_part_label(dt):
    day = int(dt.day)

    if day <= 10:
        return 1

    if day <= 20:
        return 2

    return 3


def week_of_month(dt):
    first_day = dt.replace(day=1)
    adjusted_day = dt.day + first_day.weekday()

    return int(np.ceil(adjusted_day / 7.0))


def cosine_sim(vec1, vec2):
    if vec1 is None or vec2 is None:
        return 0.0

    v1 = np.array(vec1, dtype=float).reshape(1, -1)
    v2 = np.array(vec2, dtype=float).reshape(1, -1)

    if np.allclose(v1, 0) or np.allclose(v2, 0):
        return 0.0

    return float(cosine_similarity(v1, v2)[0][0])


def mean_pool_vectors(vectors):
    valid_vectors = []

    for vector in vectors:
        if vector is not None:
            valid_vectors.append(np.array(vector, dtype=float))

    if len(valid_vectors) == 0:
        return None

    return np.mean(valid_vectors, axis=0)