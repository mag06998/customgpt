"""


"""

from flask import Flask, send_from_directory
from pathlib import Path

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

    angular_dist_path=Path(__file__).parent.parent.parent / "frontend" / "dist" / "custom-gpt-frontend" / "browser"
    print(angular_dist_path)
    @app.route('/',methods=['GET'])
    def index():
        # this will eventually have to return angular application
        #'frontend//dist//custom-gpt-frontend//browser//index.html'
        return send_from_directory(str(angular_dist_path),"index.html")

    @app.route('/<path:path>')
    def static_files(path):
        #needs to serve up other files needed
        full_path = angular_dist_path / path
        print(full_path)
        if full_path.exists():
            return send_from_directory(str(angular_dist_path),path)
        else:
            return send_from_directory(str(angular_dist_path),"index.html")

    return app

if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)