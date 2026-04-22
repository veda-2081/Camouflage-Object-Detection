from flask import Flask
from .routes import main
def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'your-secret-key'
    app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  
    app.register_blueprint(main)
    return app