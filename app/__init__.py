from flask import Flask
from config import Config
from app.extensions import swagger
from app.routes.recommendation_routes import recommendation_bp


def create_app():
    app = Flask(__name__)
    app.config.from_object(Config)

    swagger.init_app(app)
    app.register_blueprint(recommendation_bp, url_prefix="/api")

    return app