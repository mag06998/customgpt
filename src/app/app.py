"""


"""

from flask import Flask
from src.app.routes import chat_blueprint
from flask_cors import CORS

import dotenv
import os
dotenv.load_dotenv("..//..//auth//.env")

def create_app():
    app = Flask(__name__)
    CORS(app, origins=[
        "http://localhost:4200"])  # delete in production; this is just allowing cross origin communication for angular dev server
    app.register_blueprint(chat_blueprint,url_prefix='/chat')
    @app.route('/')
    def index():
        # this will eventually have to return angular application
        return "Hello app is running!"

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)