"""


"""

from flask import Flask
from src.app.routes import chat_blueprint

import dotenv
import os
dotenv.load_dotenv("..//..//auth//.env")

def create_app():
    app = Flask(__name__)
    app.register_blueprint(chat_blueprint,url_prefix='/chat')
    @app.route('/')
    def index():
        # this will eventually have to return angular application
        return "Hello app is running!"

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)