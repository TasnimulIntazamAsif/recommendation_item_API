from pathlib import Path

BASE_DIR = Path(r"D:\recommendation_item_API")
MODEL_ASSETS_DIR = BASE_DIR / "model_assets"


class Config:
    DEBUG = True
    JSON_SORT_KEYS = False

    MODEL_DIR = MODEL_ASSETS_DIR / "model_outputs_v2"
    ARTIFACT_DIR = MODEL_ASSETS_DIR / "stage1_artifacts_v2"
    DATA_DIR = MODEL_ASSETS_DIR / "data"

    MODEL_FILE = MODEL_DIR / "xgboost_ranker_model.json"
    FEATURE_FILE = MODEL_DIR / "ranker_feature_columns.json"
    SUMMARY_FILE = MODEL_DIR / "training_summary.json"
    FINAL_RESULT_FILE = MODEL_DIR / "final_result_summary.csv"

    ITEM_CATALOG_FILE = DATA_DIR / "item_catalog.csv"

    SWAGGER = {
        "title": "Retail Recommendation API",
        "uiversion": 3
    }