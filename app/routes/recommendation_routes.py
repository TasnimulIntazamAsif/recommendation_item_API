from flask import Blueprint, request, jsonify, current_app
from flasgger import swag_from

from app.services.recommender_service import RecommenderService

recommendation_bp = Blueprint("recommendation_bp", __name__)

_service = None


def get_service():
    global _service
    if _service is None:
        _service = RecommenderService(current_app.config)
    return _service


@recommendation_bp.route("/health", methods=["GET"])
def health():
    return jsonify({
        "success": True,
        "message": "API is running"
    })


@recommendation_bp.route("/metrics", methods=["GET"])
def metrics():
    import pandas as pd
    summary_file = current_app.config["FINAL_RESULT_FILE"]

    if not summary_file.exists():
        return jsonify({
            "success": False,
            "message": "final_result_summary.csv পাওয়া যায়নি"
        }), 404

    df = pd.read_csv(summary_file)
    return jsonify({
        "success": True,
        "metrics": df.to_dict(orient="records")
    })


@recommendation_bp.route("/recommend", methods=["POST"])
@swag_from({
    "tags": ["Recommendation"],
    "parameters": [
        {
            "name": "body",
            "in": "body",
            "required": True,
            "schema": {
                "type": "object",
                "properties": {
                    "customerid": {"type": "integer", "example": 23412},
                    "date": {"type": "string", "example": "2026-04-20"},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "itemid": {"type": "integer", "example": 13989},
                                "quantity": {"type": "number", "example": 1}
                            },
                            "required": ["itemid", "quantity"]
                        }
                    }
                },
                "required": ["customerid", "date", "items"]
            }
        }
    ],
    "responses": {
        200: {
            "description": "Top recommendations with score"
        },
        400: {
            "description": "Bad request"
        }
    }
})
def recommend():
    try:
        payload = request.get_json()

        if not payload:
            return jsonify({
                "success": False,
                "message": "JSON body পাওয়া যায়নি"
            }), 400

        if "customerid" not in payload or "date" not in payload or "items" not in payload:
            return jsonify({
                "success": False,
                "message": "customerid, date, items লাগবে"
            }), 400

        service = get_service()
        result = service.recommend(payload, top_n=10)

        return jsonify(result), 200

    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500