from flask import Flask

def create_app():
    app = Flask(__name__)
    app.config.from_pyfile("app.config")

    from app.api.routes import api_blueprint
    app.register_blueprint(api_blueprint, url_prefix="/api")

    return app
