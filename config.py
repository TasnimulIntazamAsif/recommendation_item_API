from pathlib import Path

BASE_DIR = Path(r"D:\recommendation_api")
MODEL_ASSETS_DIR = BASE_DIR / "model_assets"


class Config:
    DEBUG = True
    JSON_SORT_KEYS = False

    MODEL_DIR = MODEL_ASSETS_DIR / "model_outputs_v2"
    DATA_DIR = MODEL_ASSETS_DIR / "data"
    STAGE1_ARTIFACT_DIR = MODEL_ASSETS_DIR / "stage1_runtime_artifacts"

    MODEL_FILE = MODEL_DIR / "xgboost_ranker_model.json"
    FEATURE_FILE = MODEL_DIR / "ranker_feature_columns.json"
    SUMMARY_FILE = MODEL_DIR / "ranker_training_summary.json"
    FINAL_RESULT_FILE = MODEL_DIR / "final_result_summary.csv"

    MAIN_DATA_FILE = DATA_DIR / "main_data.csv"
    ITEM_CATALOG_FILE = DATA_DIR / "item_catalog.csv"
    CATEGORY_RULE_FILE = DATA_DIR / "category_rule_artifacts.json"
    CATEGORY_EMBEDDING_FILE = DATA_DIR / "category_embedding_lookup.csv"
    NGCF_CANDIDATE_FILE = DATA_DIR / "ngcf_top_candidates.csv"
    STAGE1_ARTIFACT_DIR = MODEL_ASSETS_DIR / "stage1_runtime_artifacts"

    SWAGGER = {
        "title": "Retail Recommendation API",
        "uiversion": 3
    }