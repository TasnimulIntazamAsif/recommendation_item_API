from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import xgboost as xgb

from app.utils.helpers import (
    load_json,
    load_pickle,
    normalize_text,
    infer_season_from_month,
    infer_timeslot_from_hour,
    season_to_label,
    timeslot_to_label,
    month_part_label,
    week_of_month,
    cosine_sim,
    mean_pool_vectors
)


class RecommenderService:
    def __init__(self, config):
        self.config = config

        self.ranker = xgb.XGBRanker()
        self.ranker.load_model(str(config["MODEL_FILE"]))

        feature_artifact = load_json(config["FEATURE_FILE"])

        if isinstance(feature_artifact, dict):
            self.trained_feature_columns = feature_artifact.get(
                "final_feature_cols",
                []
            )
        else:
            self.trained_feature_columns = feature_artifact

        if not self.trained_feature_columns:
            raise ValueError("ranker_feature_columns.json has no final feature columns")

        self.item_catalog_df = pd.read_csv(config["ITEM_CATALOG_FILE"])

        if config["NGCF_CANDIDATE_FILE"].exists():
            self.ngcf_candidates_df = pd.read_csv(config["NGCF_CANDIDATE_FILE"])
        else:
            self.ngcf_candidates_df = pd.DataFrame(
                columns=["customerId", "itemId", "ngcf_score", "ngcf_rank"]
            )

        self.rule_artifacts = load_json(config["CATEGORY_RULE_FILE"])
        self.category_allowed_map = self.rule_artifacts["category_allowed_map"]

        self.prepare_data()
        self.load_stage1_artifacts()
        self.build_ngcf_lookup()
        self.build_category_rule_cache()

    def prepare_data(self):
        self.item_catalog_df.columns = [
            c.strip() for c in self.item_catalog_df.columns
        ]

        self.item_catalog_df["itemId"] = self.item_catalog_df["itemId"].astype(int)
        self.item_catalog_df["itemName"] = self.item_catalog_df["itemName"].apply(normalize_text)
        self.item_catalog_df["category"] = self.item_catalog_df["category"].apply(normalize_text)

        self.catalog_by_category = {
            category: group.copy()
            for category, group in self.item_catalog_df.groupby("category", sort=False)
        }

    def load_stage1_artifacts(self):
        artifact_dir = self.config["STAGE1_ARTIFACT_DIR"]

        self.item_pair_counter = load_pickle(artifact_dir / "item_pair_counter.pkl")
        self.context_item_counter = load_pickle(artifact_dir / "context_item_counter.pkl")
        self.user_item_counter = load_pickle(artifact_dir / "user_item_counter.pkl")
        self.user_category_counter = load_pickle(artifact_dir / "user_category_counter.pkl")
        self.category_popularity_counter = load_pickle(artifact_dir / "category_popularity_counter.pkl")

        self.pair_to_related_items = load_pickle(artifact_dir / "pair_to_related_items.pkl")
        self.context_to_items = load_pickle(artifact_dir / "context_to_items.pkl")
        self.user_to_items = load_pickle(artifact_dir / "user_to_items.pkl")

        self.customer_default_timeslot = load_pickle(artifact_dir / "customer_default_timeslot.pkl")
        self.item_to_category = load_pickle(artifact_dir / "item_to_category.pkl")
        self.item_to_name = load_pickle(artifact_dir / "item_to_name.pkl")
        self.category_to_vector = load_pickle(artifact_dir / "category_to_vector.pkl")

        family_map_file = artifact_dir / "category_family_map.pkl"

        if family_map_file.exists():
            self.category_family_map = load_pickle(family_map_file)
        else:
            self.category_family_map = self.rule_artifacts.get("category_family_map", {})

    def build_ngcf_lookup(self):
        self.ngcf_candidate_lookup = defaultdict(list)
        self.ngcf_score_lookup = {}
        self.ngcf_rank_lookup = {}

        if self.ngcf_candidates_df.empty:
            return

        self.ngcf_candidates_df.columns = [
            c.strip() for c in self.ngcf_candidates_df.columns
        ]

        self.ngcf_candidates_df["customerId"] = self.ngcf_candidates_df["customerId"].astype(int)
        self.ngcf_candidates_df["itemId"] = self.ngcf_candidates_df["itemId"].astype(int)

        self.ngcf_candidates_df["ngcf_score"] = pd.to_numeric(
            self.ngcf_candidates_df["ngcf_score"],
            errors="coerce"
        ).fillna(0.0)

        self.ngcf_candidates_df["ngcf_rank"] = pd.to_numeric(
            self.ngcf_candidates_df["ngcf_rank"],
            errors="coerce"
        ).fillna(999).astype(int)

        self.ngcf_candidates_df = self.ngcf_candidates_df.sort_values(
            ["customerId", "ngcf_rank"]
        )

        for _, row in self.ngcf_candidates_df.iterrows():
            customer_id = int(row["customerId"])
            item_id = int(row["itemId"])
            score = float(row["ngcf_score"])
            rank = int(row["ngcf_rank"])

            self.ngcf_candidate_lookup[customer_id].append(item_id)
            self.ngcf_score_lookup[(customer_id, item_id)] = score
            self.ngcf_rank_lookup[(customer_id, item_id)] = rank

    def build_category_rule_cache(self):
        self.category_allowed_cache = {}

        for source_category, rule in self.category_allowed_map.items():
            source_category = str(source_category).strip()
            any_allowed = rule.get("Any", [])

            for time_slot in ["Morning", "Noon", "Afternoon", "Evening", "Night"]:
                if time_slot in rule:
                    self.category_allowed_cache[(source_category, time_slot)] = list(rule[time_slot])
                else:
                    self.category_allowed_cache[(source_category, time_slot)] = list(any_allowed)

            self.category_allowed_cache[(source_category, None)] = list(any_allowed)

    def get_category_family(self, category):
        return self.category_family_map.get(str(category).strip(), "other")

    def get_rule_allowed_categories(self, source_category, time_slot=None):
        source_category = str(source_category).strip()

        if (source_category, time_slot) in self.category_allowed_cache:
            return self.category_allowed_cache[(source_category, time_slot)]

        if (source_category, None) in self.category_allowed_cache:
            return self.category_allowed_cache[(source_category, None)]

        return []

    def build_dynamic_category_weights(self, context_categories, time_slot=None):
        context_categories = [
            str(c).strip() for c in context_categories
            if str(c).strip()
        ]

        context_categories = list(dict.fromkeys(context_categories))

        allowed_weight_counter = Counter()
        source_family_counter = Counter()

        for source_category in context_categories:
            source_family = self.get_category_family(source_category)
            source_family_counter[source_family] += 1

            source_allowed = self.get_rule_allowed_categories(
                source_category,
                time_slot
            )

            if not source_allowed:
                continue

            for rank, target_category in enumerate(source_allowed, start=1):
                target_family = self.get_category_family(target_category)

                base_weight = 1.0 / rank

                if target_family == source_family:
                    base_weight += 0.45

                if source_category in ["Meat-Fresh", "Fish-Fresh"]:
                    if target_category in [
                        "pantry salt",
                        "Pantry-Oils",
                        "general_cooking_vegetables",
                        "Pantry-Rice",
                        "Spices-Cooking",
                        "sauce item"
                    ]:
                        base_weight += 0.65

                if source_category == "Bakery-Bread":
                    if time_slot == "Morning":
                        if target_category in [
                            "Beverage-Hot",
                            "Dairy-Milk",
                            "Dairy-Other",
                            "Protein-Egg"
                        ]:
                            base_weight += 0.60
                    else:
                        if target_category in [
                            "Snacks-General",
                            "Meat-Processed",
                            "sauce item",
                            "noodles_pasta_and_haleem"
                        ]:
                            base_weight += 0.45

                if source_category in [
                    "clothing - male",
                    "clothing female",
                    "clothing babies"
                ]:
                    if target_family == "clothing":
                        base_weight += 0.70

                allowed_weight_counter[target_category] += base_weight

        return allowed_weight_counter, source_family_counter

    def build_family_quota(self, source_family_counter, top_n=10):
        if not source_family_counter:
            return {}

        total_sources = sum(source_family_counter.values())
        family_quota = {}

        for family, count in source_family_counter.items():
            ratio = count / total_sources
            quota = max(1, int(round(ratio * top_n)))
            family_quota[family] = quota

        if len(family_quota) > 1:
            for family in family_quota:
                family_quota[family] = max(2, family_quota[family])

        return family_quota

    def build_allowed_categories_from_context(self, context_categories, time_slot=None):
        context_categories = [
            str(c).strip() for c in context_categories
            if str(c).strip()
        ]

        context_categories = list(dict.fromkeys(context_categories))
        context_category_set = set(context_categories)

        allowed_weight_counter, source_family_counter = self.build_dynamic_category_weights(
            context_categories=context_categories,
            time_slot=time_slot
        )

        allowed_categories = [
            category for category, _ in allowed_weight_counter.most_common()
        ]

        strict_different_category_only = False

        if "Fish-Fresh" in context_category_set or "Meat-Fresh" in context_category_set:
            strict_different_category_only = True

        if strict_different_category_only:
            allowed_categories = [
                category for category in allowed_categories
                if category not in context_category_set
            ]

        if not allowed_categories:
            for source_category in context_categories:
                source_allowed = self.get_rule_allowed_categories(
                    source_category,
                    time_slot
                )
                allowed_categories.extend(source_allowed)

            allowed_categories = list(dict.fromkeys(allowed_categories))

        if not allowed_categories:
            allowed_categories = self.item_catalog_df["category"].dropna().unique().tolist()

        rule_mode = "mixed_basket" if len(context_categories) > 1 else (
            context_categories[0] if context_categories else "fallback_all"
        )

        family_quota = self.build_family_quota(
            source_family_counter=source_family_counter,
            top_n=10
        )

        return allowed_categories, strict_different_category_only, rule_mode, dict(allowed_weight_counter), family_quota

    def filter_catalog_by_allowed_categories(
        self,
        allowed_categories,
        input_categories=None,
        strict_different_category_only=False
    ):
        if input_categories is None:
            input_categories = []

        frames = []

        for category in allowed_categories:
            if category in self.catalog_by_category:
                frames.append(self.catalog_by_category[category])

        if not frames:
            return self.item_catalog_df.iloc[0:0].copy()

        temp = pd.concat(frames, ignore_index=True)

        if strict_different_category_only:
            input_category_set = set(input_categories)
            temp = temp[~temp["category"].isin(input_category_set)].copy()

        return temp

    def build_request_context(self, customer_id, date_str, requested_time_slot=None):
        dt = pd.to_datetime(date_str, errors="coerce")

        if pd.isna(dt):
            raise ValueError("date parse failed")

        season = infer_season_from_month(dt.month)

        if requested_time_slot:
            time_slot = str(requested_time_slot).strip()
        else:
            time_slot = infer_timeslot_from_hour(dt.hour)

        return {
            "date": str(dt),
            "dateOnly": str(dt.date()),
            "season": season,
            "seasonLabel": season_to_label(season),
            "isHoliday": 0,
            "isFestival": 0,
            "timeSlot": time_slot,
            "timeSlotLabel": timeslot_to_label(time_slot),
            "monthPartLabel": month_part_label(dt),
            "weekOfMonth": week_of_month(dt)
        }

    def get_input_categories(self, input_item_ids):
        categories = []

        for item_id in input_item_ids:
            category = self.item_to_category.get(int(item_id))

            if category:
                categories.append(category)

        return list(dict.fromkeys(categories))

    def get_top_copurchase_candidates(self, context_item_ids, top_n=30):
        score_counter = Counter()

        for context_item_id in context_item_ids:
            related_counter = self.pair_to_related_items.get(
                int(context_item_id),
                Counter()
            )

            score_counter.update(related_counter)

        return [item_id for item_id, _ in score_counter.most_common(top_n)]

    def get_top_context_candidates(self, context_key, top_n=30):
        score_counter = self.context_to_items.get(context_key, Counter())
        return [item_id for item_id, _ in score_counter.most_common(top_n)]

    def get_top_user_history_candidates(self, customer_id, top_n=30):
        score_counter = self.user_to_items.get(int(customer_id), Counter())
        return [item_id for item_id, _ in score_counter.most_common(top_n)]

    def get_top_ngcf_candidates(self, customer_id, top_n=80):
        return self.ngcf_candidate_lookup.get(int(customer_id), [])[:top_n]

    def get_allowed_category_candidates(
        self,
        allowed_categories,
        input_categories,
        strict_different_category_only,
        top_n=80
    ):
        temp = self.filter_catalog_by_allowed_categories(
            allowed_categories=allowed_categories,
            input_categories=input_categories,
            strict_different_category_only=strict_different_category_only
        )

        if temp.empty:
            return []

        temp = temp.copy()

        temp["categoryPopularity"] = temp["category"].map(
            lambda c: self.category_popularity_counter.get(c, 0)
        )

        if "totalRowsSeen" not in temp.columns:
            temp["totalRowsSeen"] = 1

        temp["totalRowsSeen"] = pd.to_numeric(
            temp["totalRowsSeen"],
            errors="coerce"
        ).fillna(1)

        temp = temp.sort_values(
            ["categoryPopularity", "totalRowsSeen"],
            ascending=[False, False]
        )

        return temp["itemId"].astype(int).head(top_n).tolist()

    def build_candidate_pool(
        self,
        customer_id,
        context_item_ids,
        context_key,
        allowed_categories,
        input_categories,
        strict_different_category_only
    ):
        candidate_items = []

        candidate_items.extend(
            self.get_top_copurchase_candidates(context_item_ids, top_n=30)
        )

        candidate_items.extend(
            self.get_top_context_candidates(context_key, top_n=30)
        )

        candidate_items.extend(
            self.get_top_user_history_candidates(customer_id, top_n=30)
        )

        candidate_items.extend(
            self.get_top_ngcf_candidates(customer_id, top_n=80)
        )

        candidate_items.extend(
            self.get_allowed_category_candidates(
                allowed_categories=allowed_categories,
                input_categories=input_categories,
                strict_different_category_only=strict_different_category_only,
                top_n=80
            )
        )

        candidate_items = list(dict.fromkeys([int(x) for x in candidate_items]))

        context_item_set = set([int(x) for x in context_item_ids])
        allowed_category_set = set(allowed_categories)
        input_category_set = set(input_categories)

        filtered_items = []

        for item_id in candidate_items:
            category = self.item_to_category.get(int(item_id), "")

            if category not in allowed_category_set:
                continue

            if strict_different_category_only and category in input_category_set:
                continue

            if item_id in context_item_set:
                continue

            filtered_items.append(item_id)

        return filtered_items

    def get_item_cooccurrence_score(self, candidate_item_id, context_item_ids):
        if len(context_item_ids) == 0:
            return 0.0

        scores = []

        for context_item_id in context_item_ids:
            scores.append(
                self.item_pair_counter.get(
                    (int(candidate_item_id), int(context_item_id)),
                    0
                )
            )

        return float(np.mean(scores)) if scores else 0.0

    def get_customer_history_score(self, customer_id, candidate_item_id):
        return float(
            self.user_item_counter.get(
                (int(customer_id), int(candidate_item_id)),
                0
            )
        )

    def get_category_affinity_score(self, customer_id, candidate_category):
        return float(
            self.user_category_counter.get(
                (int(customer_id), candidate_category),
                0
            )
        )

    def get_context_popularity_score(self, context_key, candidate_item_id):
        return float(
            self.context_item_counter.get(
                (context_key, int(candidate_item_id)),
                0
            )
        )

    def get_category_popularity_score(self, candidate_category):
        return float(
            self.category_popularity_counter.get(candidate_category, 0)
        )

    def build_context_embedding_from_items(self, item_ids):
        vectors = []

        for item_id in item_ids:
            category = self.item_to_category.get(int(item_id), "")

            if category in self.category_to_vector:
                vectors.append(self.category_to_vector[category])

        return mean_pool_vectors(vectors)

    def get_embedding_similarity_score(self, context_vector, candidate_item_id):
        candidate_category = self.item_to_category.get(int(candidate_item_id), "")
        candidate_vector = self.category_to_vector.get(candidate_category)

        return cosine_sim(context_vector, candidate_vector)

    def get_ngcf_score(self, customer_id, candidate_item_id):
        return float(
            self.ngcf_score_lookup.get(
                (int(customer_id), int(candidate_item_id)),
                0.0
            )
        )

    def get_ngcf_rank(self, customer_id, candidate_item_id):
        return int(
            self.ngcf_rank_lookup.get(
                (int(customer_id), int(candidate_item_id)),
                999
            )
        )

    def build_scoring_rows(
        self,
        customer_id,
        request_context,
        input_item_ids,
        candidate_pool,
        allowed_categories,
        input_categories,
        rule_mode,
        allowed_category_weight_map
    ):
        context_embedding_vec = self.build_context_embedding_from_items(input_item_ids)

        context_key = (
            request_context["season"],
            int(request_context["isHoliday"]),
            int(request_context["isFestival"]),
            request_context["timeSlot"],
            int(request_context["weekOfMonth"])
        )

        input_families = [
            self.get_category_family(category)
            for category in input_categories
        ]
        input_families = list(dict.fromkeys(input_families))

        mixed_basket_flag = 1 if len(input_families) > 1 else 0
        input_family_count = len(input_families)

        rows = []

        for candidate_item_id in candidate_pool:
            candidate_item_id = int(candidate_item_id)
            candidate_category = self.item_to_category.get(candidate_item_id, "Unknown")
            candidate_family = self.get_category_family(candidate_category)

            if candidate_category in allowed_categories:
                allowed_category_rank = allowed_categories.index(candidate_category) + 1
            else:
                allowed_category_rank = 999

            candidate_family_in_input = 1 if candidate_family in input_families else 0

            rows.append({
                "customerId": int(customer_id),
                "candidate_item_id": candidate_item_id,
                "candidate_item_name": self.item_to_name.get(candidate_item_id, f"Item_{candidate_item_id}"),
                "candidate_category": candidate_category,
                "candidate_family": candidate_family,
                "basket_size": int(len(input_item_ids)),
                "item_cooccurrence_score": self.get_item_cooccurrence_score(candidate_item_id, input_item_ids),
                "category_affinity_score": self.get_category_affinity_score(customer_id, candidate_category),
                "context_popularity_score": self.get_context_popularity_score(context_key, candidate_item_id),
                "customer_history_score": self.get_customer_history_score(customer_id, candidate_item_id),
                "embedding_similarity_score": self.get_embedding_similarity_score(context_embedding_vec, candidate_item_id),
                "candidate_category_popularity": self.get_category_popularity_score(candidate_category),
                "allowed_category_rank": int(allowed_category_rank),
                "candidate_category_allowed": 1 if candidate_category in allowed_categories else 0,
                "same_category_as_context": 1 if candidate_category in input_categories else 0,
                "category_rule_weight": float(allowed_category_weight_map.get(candidate_category, 0.0)),
                "candidate_family_in_input": int(candidate_family_in_input),
                "mixed_basket_flag": int(mixed_basket_flag),
                "input_family_count": int(input_family_count),
                "ngcf_score": self.get_ngcf_score(customer_id, candidate_item_id),
                "ngcf_rank": self.get_ngcf_rank(customer_id, candidate_item_id),
                "season": request_context["season"],
                "seasonLabel": int(request_context["seasonLabel"]),
                "isHoliday": int(request_context["isHoliday"]),
                "isFestival": int(request_context["isFestival"]),
                "timeSlot": request_context["timeSlot"],
                "timeSlotLabel": int(request_context["timeSlotLabel"]),
                "monthPartLabel": int(request_context["monthPartLabel"]),
                "weekOfMonth": int(request_context["weekOfMonth"]),
                "rule_mode": rule_mode
            })

        return pd.DataFrame(rows)

    def build_live_feature_matrix(self, score_df):
        numeric_feature_cols = [
            "basket_size",
            "item_cooccurrence_score",
            "category_affinity_score",
            "context_popularity_score",
            "customer_history_score",
            "embedding_similarity_score",
            "candidate_category_popularity",
            "allowed_category_rank",
            "candidate_category_allowed",
            "same_category_as_context",
            "category_rule_weight",
            "candidate_family_in_input",
            "mixed_basket_flag",
            "input_family_count",
            "ngcf_score",
            "ngcf_rank",
            "seasonLabel",
            "isHoliday",
            "isFestival",
            "timeSlotLabel",
            "monthPartLabel",
            "weekOfMonth"
        ]

        categorical_feature_cols = [
            "candidate_category",
            "candidate_family",
            "season",
            "timeSlot",
            "rule_mode"
        ]

        X_numeric = score_df[numeric_feature_cols].copy()
        X_numeric = X_numeric.replace([np.inf, -np.inf], 0).fillna(0)

        X_categorical = pd.get_dummies(
            score_df[categorical_feature_cols],
            prefix=categorical_feature_cols,
            dtype=np.int8
        )

        X_live = pd.concat([X_numeric, X_categorical], axis=1)
        X_live = X_live.reindex(columns=self.trained_feature_columns, fill_value=0)

        return X_live

    def stable_pick_from_category(self, category_df, category, customer_id, input_item_ids, top_k=5):
        category_df = category_df.copy().sort_values(
            ["final_score", "ngcf_score"],
            ascending=[False, False]
        ).reset_index(drop=True)

        if category_df.empty:
            return None

        category_df = category_df.head(top_k).copy()

        seed_value = int(customer_id)

        for item_id in input_item_ids:
            seed_value += int(item_id) * 31

        seed_value += sum(ord(ch) for ch in str(category))

        pick_index = seed_value % len(category_df)

        return category_df.iloc[pick_index]

    def apply_strict_rule_reranking(
        self,
        score_df,
        top_n=10,
        customer_id=None,
        input_item_ids=None,
        family_quota=None
    ):
        score_df = score_df.copy()
        score_df = score_df[score_df["candidate_category_allowed"] == 1].copy()

        if input_item_ids is None:
            input_item_ids = []

        if customer_id is None:
            customer_id = 0

        if family_quota is None:
            family_quota = {}

        if score_df.empty:
            return score_df

        score_df["final_score"] = score_df["score"]
        score_df["final_score"] += score_df["ngcf_score"] * 0.05
        score_df["final_score"] += score_df["category_rule_weight"] * 0.08
        score_df["final_score"] += (1 / score_df["allowed_category_rank"].replace(0, 999)) * 0.02

        score_df = score_df.sort_values(
            ["final_score", "ngcf_score"],
            ascending=[False, False]
        ).reset_index(drop=True)

        selected_rows = []
        selected_item_ids = set()
        selected_categories = set()
        selected_family_counter = Counter()

        category_order = (
            score_df
            .sort_values(["category_rule_weight", "final_score"], ascending=[False, False])
            ["candidate_category"]
            .drop_duplicates()
            .tolist()
        )

        for category in category_order:
            if category in selected_categories:
                continue

            category_df = score_df[score_df["candidate_category"] == category].copy()

            if category_df.empty:
                continue

            category_family = category_df.iloc[0]["candidate_family"]

            if family_quota:
                quota = family_quota.get(category_family, 1)

                if selected_family_counter[category_family] >= quota:
                    continue

            picked_row = self.stable_pick_from_category(
                category_df=category_df,
                category=category,
                customer_id=customer_id,
                input_item_ids=input_item_ids,
                top_k=5
            )

            if picked_row is None:
                continue

            item_id = int(picked_row["candidate_item_id"])

            if item_id in selected_item_ids:
                continue

            selected_rows.append(picked_row)
            selected_item_ids.add(item_id)
            selected_categories.add(category)
            selected_family_counter[category_family] += 1

            if len(selected_rows) >= top_n:
                break

        if len(selected_rows) < top_n:
            for _, row in score_df.iterrows():
                item_id = int(row["candidate_item_id"])
                category = row["candidate_category"]

                if item_id in selected_item_ids:
                    continue

                if category in selected_categories:
                    continue

                selected_rows.append(row)
                selected_item_ids.add(item_id)
                selected_categories.add(category)

                if len(selected_rows) >= top_n:
                    break

        if not selected_rows:
            return score_df.head(top_n).copy()

        return pd.DataFrame(selected_rows).reset_index(drop=True)

    def add_display_score(self, top_df):
        top_df = top_df.copy()

        if top_df.empty:
            top_df["display_score"] = []
            return top_df

        scores = top_df["final_score"].values

        min_score = scores.min()
        max_score = scores.max()

        if max_score > min_score:
            top_df["display_score"] = (scores - min_score) / (max_score - min_score)
        else:
            top_df["display_score"] = 1.0

        return top_df

    def build_input_item_names(self, items):
        input_item_names = []

        for x in items:
            item_id = int(x["itemid"])
            item_name = self.item_to_name.get(item_id, f"Item_{item_id}")
            input_item_names.append(item_name)

        return input_item_names

    def recommend(self, payload, top_n=10):
        customer_id = int(payload["customerid"])
        date_str = payload["date and time"]
        items = payload["items"]
        requested_time_slot = payload.get("timeSlot", None)

        input_item_ids = [int(x["itemid"]) for x in items]
        unique_input_item_ids = list(dict.fromkeys(input_item_ids))

        input_item_names = self.build_input_item_names(items)

        request_context = self.build_request_context(
            customer_id=customer_id,
            date_str=date_str,
            requested_time_slot=requested_time_slot
        )

        context_key = (
            request_context["season"],
            int(request_context["isHoliday"]),
            int(request_context["isFestival"]),
            request_context["timeSlot"],
            int(request_context["weekOfMonth"])
        )

        input_categories = self.get_input_categories(unique_input_item_ids)

        (
            allowed_categories,
            strict_different_category_only,
            rule_mode,
            allowed_category_weight_map,
            family_quota
        ) = self.build_allowed_categories_from_context(
            input_categories,
            time_slot=request_context["timeSlot"]
        )

        if not allowed_categories:
            allowed_categories = self.item_catalog_df["category"].dropna().unique().tolist()
            strict_different_category_only = False
            rule_mode = "fallback_all"
            allowed_category_weight_map = {}
            family_quota = {}

        candidate_pool = self.build_candidate_pool(
            customer_id=customer_id,
            context_item_ids=unique_input_item_ids,
            context_key=context_key,
            allowed_categories=allowed_categories,
            input_categories=input_categories,
            strict_different_category_only=strict_different_category_only
        )

        if not candidate_pool:
            return {
                "input_item_names": input_item_names,
                "recommendations": []
            }

        score_df = self.build_scoring_rows(
            customer_id=customer_id,
            request_context=request_context,
            input_item_ids=unique_input_item_ids,
            candidate_pool=candidate_pool,
            allowed_categories=allowed_categories,
            input_categories=input_categories,
            rule_mode=rule_mode,
            allowed_category_weight_map=allowed_category_weight_map
        )

        if score_df.empty:
            return {
                "input_item_names": input_item_names,
                "recommendations": []
            }

        X_live = self.build_live_feature_matrix(score_df)

        score_df["score"] = self.ranker.predict(X_live)

        top_df = self.apply_strict_rule_reranking(
            score_df=score_df,
            top_n=top_n,
            customer_id=customer_id,
            input_item_ids=unique_input_item_ids,
            family_quota=family_quota
        )

        top_df = self.add_display_score(top_df)

        recommendations = []

        for _, row in top_df.iterrows():
            recommendations.append({
                "category": row["candidate_category"],
                "item_name": row["candidate_item_name"],
                "itemid": int(row["candidate_item_id"]),
                "score": round(float(row["display_score"]), 6)
            })

        return {
            "input_item_names": input_item_names,
            "recommendations": recommendations
        }