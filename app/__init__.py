from flask import Flask

from config import Config
from app.extensions import swagger
from app.routes.recommendation_routes import recommendation_bp
from app.services.recommender_service import RecommenderService


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    swagger.init_app(app)

    with app.app_context():
        app.recommender_service = RecommenderService(app.config)

    app.register_blueprint(recommendation_bp, url_prefix="/api")

    return app