from collections import Counter

import numpy as np
import pandas as pd
import xgboost as xgb

from app.utils.helpers import (
    load_pickle,
    load_json,
    infer_season_from_month,
    week_of_month,
    cosine_sim,
    mean_pool_vectors
)


class RecommenderService:
    def __init__(self, config):
        self.config = config

        # model load
        self.ranker = xgb.XGBRanker()
        self.ranker.load_model(str(config["MODEL_FILE"]))

        # trained feature columns
        self.trained_feature_columns = load_json(config["FEATURE_FILE"])

        # stage 1 artifacts load
        self.rules_df = load_pickle(config["ARTIFACT_DIR"] / "association_rules.pkl")
        self.item_pair_counter = load_pickle(config["ARTIFACT_DIR"] / "item_pair_counter.pkl")
        self.context_item_counter = load_pickle(config["ARTIFACT_DIR"] / "context_item_counter.pkl")
        self.user_item_counter = load_pickle(config["ARTIFACT_DIR"] / "user_item_counter.pkl")
        self.user_category_counter = load_pickle(config["ARTIFACT_DIR"] / "user_category_counter.pkl")
        self.item_to_category = load_pickle(config["ARTIFACT_DIR"] / "item_to_category.pkl")
        self.item_to_name = load_pickle(config["ARTIFACT_DIR"] / "item_to_name.pkl")
        self.category_to_vector = load_pickle(config["ARTIFACT_DIR"] / "category_to_vector.pkl")

        # optional lookups
        self.date_context_lookup = {}
        self.customer_default_timeslot = {}

        date_ctx_path = config["ARTIFACT_DIR"] / "date_context_lookup.pkl"
        cust_ts_path = config["ARTIFACT_DIR"] / "customer_default_timeslot.pkl"

        if date_ctx_path.exists():
            self.date_context_lookup = load_pickle(date_ctx_path)

        if cust_ts_path.exists():
            self.customer_default_timeslot = load_pickle(cust_ts_path)

    # =========================================================
    # CONTEXT HELPERS
    # =========================================================
    def build_request_context(self, customer_id, date_str):
        dt = pd.to_datetime(date_str, errors="coerce")
        if pd.isna(dt):
            raise ValueError("date parse করা যায়নি")

        date_key = str(dt.date())

        if date_key in self.date_context_lookup:
            season = self.date_context_lookup[date_key]["season"]
            is_holiday = self.date_context_lookup[date_key]["isHoliday"]
            is_festival = self.date_context_lookup[date_key]["isFestival"]
        else:
            season = infer_season_from_month(dt.month)
            is_holiday = 0
            is_festival = 0

        time_slot = self.customer_default_timeslot.get(int(customer_id), "Afternoon")
        wom = week_of_month(dt)

        return {
            "date": str(dt.date()),
            "season": season,
            "isHoliday": int(is_holiday),
            "isFestival": int(is_festival),
            "timeSlot": time_slot,
            "weekOfMonth": int(wom)
        }

    def get_input_categories(self, input_item_ids):
        categories = []

        for iid in input_item_ids:
            cat = self.item_to_category.get(int(iid), None)
            if cat is not None:
                categories.append(cat)

        categories = list(dict.fromkeys(categories))
        return categories

    def detect_meal_intent(self, input_categories):
        input_set = set(input_categories)

        cooking_triggers = {
            "Meat-Fresh",
            "Fish-Fresh",
            "Protein-Egg",
            "Veg-Cooking",
            "Spices-Cooking"
        }

        breakfast_triggers = {
            "Bakery-Bread",
            "Beverage-Hot",
            "Breakfast-Cereal",
            "Dairy-Milk",
            "Dairy-Other",
            "Spreads"
        }

        snack_triggers = {
            "Snacks-General",
            "Beverage-Carbonated",
            "Beverage-Juice",
            "Chocolates-Sweets",
            "Instant-Food"
        }

        if len(input_set.intersection(cooking_triggers)) > 0:
            return "cooking"

        if len(input_set.intersection(breakfast_triggers)) > 0:
            return "breakfast"

        if len(input_set.intersection(snack_triggers)) > 0:
            return "snack"

        return "generic"

    def get_meal_completion_boosts(self, input_categories):
        meal_intent = self.detect_meal_intent(input_categories)

        if meal_intent == "cooking":
            return {
                "Spices-Cooking": 0.25,
                "Pantry-Oils": 0.22,
                "Veg-Cooking": 0.20,
                "Pantry-Rice": 0.18,
                "Pantry-Pulses": 0.10,
                "Beverage-Carbonated": 0.08,
                "Beverage-Juice": 0.06
            }

        if meal_intent == "breakfast":
            return {
                "Bakery-Bread": 0.20,
                "Beverage-Hot": 0.22,
                "Dairy-Milk": 0.18,
                "Dairy-Other": 0.15,
                "Spreads": 0.20,
                "Protein-Egg": 0.12,
                "Breakfast-Cereal": 0.18
            }

        if meal_intent == "snack":
            return {
                "Snacks-General": 0.22,
                "Beverage-Carbonated": 0.20,
                "Beverage-Juice": 0.18,
                "Chocolates-Sweets": 0.16,
                "Instant-Food": 0.15
            }

        return {
            "Spices-Cooking": 0.08,
            "Pantry-Oils": 0.08,
            "Veg-Cooking": 0.08,
            "Pantry-Rice": 0.08,
            "Beverage-Carbonated": 0.05,
            "Beverage-Juice": 0.05
        }

    def get_single_item_boosts(self, input_categories):
        if len(input_categories) != 1:
            return {}

        only_cat = input_categories[0]

        if only_cat in {"Meat-Fresh", "Fish-Fresh", "Protein-Egg"}:
            return {
                "Spices-Cooking": 0.20,
                "Pantry-Oils": 0.18,
                "Veg-Cooking": 0.16,
                "Pantry-Rice": 0.14,
                "Beverage-Carbonated": 0.05,
                "Beverage-Juice": 0.05
            }

        if only_cat in {"Bakery-Bread", "Spreads", "Beverage-Hot"}:
            return {
                "Beverage-Hot": 0.18,
                "Bakery-Bread": 0.15,
                "Dairy-Milk": 0.12,
                "Spreads": 0.12,
                "Protein-Egg": 0.08
            }

        return {}

    def build_context_embedding_from_items(self, item_ids):
        categories = []
        for iid in item_ids:
            cat = self.item_to_category.get(int(iid), None)
            if cat is not None:
                categories.append(cat)

        categories = list(dict.fromkeys(categories))
        vecs = [self.category_to_vector.get(cat, None) for cat in categories]
        return mean_pool_vectors(vecs)

    # =========================================================
    # STAGE 1 CANDIDATE GENERATORS
    # =========================================================
    def get_rule_candidates(self, context_item_ids, top_n=30):
        context_set = set([str(int(x)) for x in context_item_ids])
        score_counter = Counter()

        for _, row in self.rules_df.iterrows():
            antecedents = set(row["antecedents"])
            consequents = [int(x) for x in row["consequents"]]

            if antecedents.issubset(context_set):
                score = float(row["confidence"]) * float(row["lift"])
                for item in consequents:
                    score_counter[item] += score

        return [item for item, _ in score_counter.most_common(top_n)]

    def get_copurchase_candidates(self, context_item_ids, top_n=30):
        score_counter = Counter()

        for ctx_item in context_item_ids:
            for (a, b), cnt in self.item_pair_counter.items():
                if b == int(ctx_item):
                    score_counter[a] += cnt

        return [item for item, _ in score_counter.most_common(top_n)]

    def get_context_candidates(self, context_key, top_n=30):
        score_counter = Counter()

        for (ctx_key, item_id), cnt in self.context_item_counter.items():
            if ctx_key == context_key:
                score_counter[item_id] += cnt

        return [item for item, _ in score_counter.most_common(top_n)]

    def get_user_history_candidates(self, customer_id, top_n=30):
        score_counter = Counter()

        for (uid, item_id), cnt in self.user_item_counter.items():
            if uid == int(customer_id):
                score_counter[item_id] += cnt

        return [item for item, _ in score_counter.most_common(top_n)]

    # =========================================================
    # FEATURE BUILDERS
    # =========================================================
    def get_item_cooccurrence_score(self, candidate_item_id, context_item_ids):
        if len(context_item_ids) == 0:
            return 0.0

        scores = [self.item_pair_counter.get((int(candidate_item_id), int(ctx)), 0) for ctx in context_item_ids]
        return float(sum(scores) / len(scores)) if len(scores) > 0 else 0.0

    def get_customer_history_score(self, customer_id, candidate_item_id):
        return float(self.user_item_counter.get((int(customer_id), int(candidate_item_id)), 0))

    def get_category_affinity_score(self, customer_id, candidate_item_id):
        cat = self.item_to_category.get(int(candidate_item_id), None)
        if cat is None:
            return 0.0
        return float(self.user_category_counter.get((int(customer_id), cat), 0))

    def get_context_popularity_score(self, context_key, candidate_item_id):
        return float(self.context_item_counter.get((context_key, int(candidate_item_id)), 0))

    def get_embedding_similarity_score(self, context_embedding_vec, candidate_item_id):
        cat = self.item_to_category.get(int(candidate_item_id), None)
        cand_vec = self.category_to_vector.get(cat, None) if cat is not None else None
        return cosine_sim(context_embedding_vec, cand_vec)

    def build_scoring_rows(self, customer_id, request_context, input_items, candidate_pool):
        context_item_ids = [int(x["itemid"]) for x in input_items]
        context_embedding_vec = self.build_context_embedding_from_items(context_item_ids)

        context_key = (
            request_context["season"],
            int(request_context["isHoliday"]),
            int(request_context["isFestival"]),
            request_context["timeSlot"],
            int(request_context["weekOfMonth"])
        )

        input_categories = self.get_input_categories(context_item_ids)
        meal_intent = self.detect_meal_intent(input_categories)

        meal_boost_map = self.get_meal_completion_boosts(input_categories)
        single_item_boost_map = self.get_single_item_boosts(input_categories)

        combined_boost_map = meal_boost_map.copy()
        for cat, boost in single_item_boost_map.items():
            combined_boost_map[cat] = combined_boost_map.get(cat, 0.0) + boost

        rows = []

        for cand in candidate_pool:
            cand_cat = self.item_to_category.get(int(cand), "Unknown")
            business_boost = float(combined_boost_map.get(cand_cat, 0.0))

            rows.append({
                "customerId": int(customer_id),
                "candidate_item_id": int(cand),
                "candidate_category": cand_cat,
                "basket_size": len(context_item_ids),
                "item_cooccurrence_score": self.get_item_cooccurrence_score(cand, context_item_ids),
                "category_affinity_score": self.get_category_affinity_score(customer_id, cand),
                "context_popularity_score": self.get_context_popularity_score(context_key, cand),
                "customer_history_score": self.get_customer_history_score(customer_id, cand),
                "embedding_similarity_score": self.get_embedding_similarity_score(context_embedding_vec, cand),
                "business_boost_score": business_boost,
                "input_category_count": len(input_categories),
                "meal_intent": meal_intent,
                "season": request_context["season"],
                "isHoliday": int(request_context["isHoliday"]),
                "isFestival": int(request_context["isFestival"]),
                "timeSlot": request_context["timeSlot"],
                "weekOfMonth": int(request_context["weekOfMonth"])
            })

        return pd.DataFrame(rows)

    # =========================================================
    # BUSINESS RERANKING
    # =========================================================
    def apply_category_boosts(self, score_df):
        score_df = score_df.copy()
        score_df["final_score"] = score_df["score"] + score_df["business_boost_score"]
        return score_df

    def ensure_category_coverage(self, score_df, input_categories, top_n=10):
        score_df = score_df.copy().sort_values("final_score", ascending=False).reset_index(drop=True)

        meal_intent = self.detect_meal_intent(input_categories)

        if meal_intent == "cooking":
            must_have_categories = ["Pantry-Oils", "Veg-Cooking", "Pantry-Rice"]
        elif meal_intent == "breakfast":
            must_have_categories = ["Beverage-Hot", "Bakery-Bread", "Dairy-Milk"]
        elif meal_intent == "snack":
            must_have_categories = ["Beverage-Carbonated", "Snacks-General"]
        else:
            must_have_categories = []

        selected = []
        selected_item_ids = set()

        for cat in must_have_categories:
            cat_rows = score_df[score_df["candidate_category"] == cat]
            if not cat_rows.empty:
                row = cat_rows.iloc[0]
                selected.append(row)
                selected_item_ids.add(int(row["candidate_item_id"]))

        for _, row in score_df.iterrows():
            iid = int(row["candidate_item_id"])
            if iid not in selected_item_ids:
                selected.append(row)
                selected_item_ids.add(iid)

            if len(selected) >= top_n:
                break

        return pd.DataFrame(selected).reset_index(drop=True)

    def apply_diversity_reranking(self, score_df, top_n=10):
        score_df = score_df.copy().sort_values("final_score", ascending=False).reset_index(drop=True)

        category_caps = {
            "Spices-Cooking": 3,
            "Pantry-Oils": 2,
            "Pantry-Rice": 2,
            "Veg-Cooking": 2,
            "Beverage-Carbonated": 2,
            "Beverage-Juice": 2
        }

        default_cap = 2

        selected_rows = []
        category_counter = Counter()

        for _, row in score_df.iterrows():
            cat = row["candidate_category"]
            cap = category_caps.get(cat, default_cap)

            if category_counter[cat] < cap:
                selected_rows.append(row)
                category_counter[cat] += 1

            if len(selected_rows) >= top_n:
                break

        if len(selected_rows) == 0:
            return score_df.head(top_n).copy()

        return pd.DataFrame(selected_rows).reset_index(drop=True)

    def add_display_score(self, score_df):
        score_df = score_df.copy()
        scores = score_df["final_score"].values

        min_s = scores.min()
        max_s = scores.max()

        if max_s > min_s:
            score_df["display_score"] = (scores - min_s) / (max_s - min_s)
        else:
            score_df["display_score"] = 1.0

        return score_df

    def get_allowed_categories(self, input_categories):
        meal_intent = self.detect_meal_intent(input_categories)

        if meal_intent == "cooking":
            return {
                "Spices-Cooking",
                "Pantry-Oils",
                "Veg-Cooking",
                "Pantry-Rice",
                "Pantry-Pulses",
                "Beverage-Carbonated",
                "Beverage-Juice",
                "Meat-Fresh",
                "Fish-Fresh",
                "Pantry-DryFood"
            }

        if meal_intent == "breakfast":
            return {
                "Bakery-Bread",
                "Beverage-Hot",
                "Dairy-Milk",
                "Dairy-Other",
                "Spreads",
                "Breakfast-Cereal",
                "Protein-Egg"
            }

        if meal_intent == "snack":
            return {
                "Snacks-General",
                "Beverage-Carbonated",
                "Beverage-Juice",
                "Instant-Food",
                "Chocolates-Sweets",
                "Bakery-Bread"
            }

        return None

    def apply_hard_category_filter(self, score_df, input_categories):
        allowed_categories = self.get_allowed_categories(input_categories)

        if allowed_categories is None:
            return score_df.copy()

        filtered_df = score_df[score_df["candidate_category"].isin(allowed_categories)].copy()

        # যদি filter করার পরে খুব কম row থাকে, fallback হিসেবে original return
        if filtered_df.shape[0] < 5:
            return score_df.copy()

        return filtered_df

    def apply_exact_category_cap(self, score_df, top_n=10):
        score_df = score_df.copy().sort_values("final_score", ascending=False).reset_index(drop=True)

        seen_exact_categories = set()
        selected_rows = []

        for _, row in score_df.iterrows():
            exact_cat = row["candidate_category"]

            if exact_cat in seen_exact_categories:
                continue

            selected_rows.append(row)
            seen_exact_categories.add(exact_cat)

            if len(selected_rows) >= top_n:
                break

        if len(selected_rows) == 0:
            return score_df.head(top_n).copy()

        return pd.DataFrame(selected_rows).reset_index(drop=True)

    # =========================================================
    # MAIN RECOMMEND FUNCTION
    # =========================================================
    def recommend(self, payload, top_n=10):
        customer_id = int(payload["customerid"])
        date_str = payload["date"]
        items = payload["items"]

        input_item_ids = [int(x["itemid"]) for x in items]
        request_context = self.build_request_context(customer_id, date_str)

        context_key = (
            request_context["season"],
            int(request_context["isHoliday"]),
            int(request_context["isFestival"]),
            request_context["timeSlot"],
            int(request_context["weekOfMonth"])
        )

        # Stage 1
        rule_candidates = self.get_rule_candidates(input_item_ids, top_n=30)
        copurchase_candidates = self.get_copurchase_candidates(input_item_ids, top_n=30)
        context_candidates = self.get_context_candidates(context_key, top_n=30)
        user_candidates = self.get_user_history_candidates(customer_id, top_n=30)

        candidate_counter = Counter()

        for rank, item in enumerate(rule_candidates):
            candidate_counter[item] += 1.5 / (rank + 1)

        for rank, item in enumerate(copurchase_candidates):
            candidate_counter[item] += 1.3 / (rank + 1)

        for rank, item in enumerate(context_candidates):
            candidate_counter[item] += 0.8 / (rank + 1)

        for rank, item in enumerate(user_candidates):
            candidate_counter[item] += 0.7 / (rank + 1)

        merged_candidates = []
        for item, _ in candidate_counter.most_common(100):
            if item not in input_item_ids:
                merged_candidates.append(item)

        # Stage 2
        score_df = self.build_scoring_rows(customer_id, request_context, items, merged_candidates)

        if score_df.empty:
            return {
                "customerid": customer_id,
                "date": date_str,
                "input_items": [
                    {
                        "item_name": self.item_to_name.get(int(x["itemid"]), f"Item_{x['itemid']}"),
                        "itemid": int(x["itemid"]),
                        "quantity": float(x["quantity"])
                    }
                    for x in items
                ],
                "recommendations": [],
                "success": True
            }

        X_live = score_df[
            [
                "basket_size",
                "item_cooccurrence_score",
                "category_affinity_score",
                "context_popularity_score",
                "customer_history_score",
                "embedding_similarity_score",
                "business_boost_score",
                "input_category_count",
                "isHoliday",
                "isFestival",
                "weekOfMonth",
                "season",
                "timeSlot",
                "candidate_category",
                "meal_intent"
            ]
        ].copy()

        X_live = pd.get_dummies(
            X_live,
            columns=["season", "timeSlot", "candidate_category", "meal_intent"]
        )

        X_live = X_live.reindex(columns=self.trained_feature_columns, fill_value=0)

        pred_scores = self.ranker.predict(X_live)

        score_df["score"] = pred_scores
        score_df["item_name"] = score_df["candidate_item_id"].apply(
            lambda x: self.item_to_name.get(int(x), f"Item_{x}")
        )

        input_categories = self.get_input_categories(input_item_ids)

        score_df = self.apply_hard_category_filter(score_df, input_categories)

        score_df = self.apply_category_boosts(score_df)
        score_df = self.ensure_category_coverage(score_df, input_categories, top_n=top_n * 3)
        score_df = self.apply_exact_category_cap(score_df, top_n=top_n)
        score_df = self.add_display_score(score_df)

        score_df["rank"] = range(1, len(score_df) + 1)

        input_items = []
        for x in items:
            input_items.append({
                "item_name": self.item_to_name.get(int(x["itemid"]), f"Item_{x['itemid']}"),
                "itemid": int(x["itemid"]),
                "quantity": float(x["quantity"])
            })

        recs = []
        for _, row in score_df.iterrows():
            recs.append({
                "category": row["candidate_category"],
                "item_name": row["item_name"],
                "itemid": int(row["candidate_item_id"]),
                "rank": int(row["rank"]),
                "score": round(float(row["display_score"]), 6),
                "raw_score": round(float(row["final_score"]), 6)
            })

        return {
            "customerid": customer_id,
            "date": date_str,
            "input_items": input_items,
            "recommendations": recs,
            "success": True
        }