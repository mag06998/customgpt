from flask import Blueprint, jsonify, request
from src.service.customGPT import CustomGPT
from src.data.context_database import context_db_connection
from pathlib import Path

chat_blueprint=Blueprint('chat',__name__)

active_gpts:dict[str,CustomGPT]={}
db_path = str(Path(__file__).parent.parent.parent.resolve() / "db" / "gpt-database.db")
@chat_blueprint.route('/get_gpts',methods=['GET'])
def get_gpts():
    db_connection = context_db_connection(db_path)

    gpt_query=db_connection.get_all_gpt_info()
    gpts=[{"name":row[1],"model":row[2]} for row in gpt_query]

    return jsonify(gpts)

@chat_blueprint.route('/query',methods=['POST'])
def query():
    query_data=request.get_json()
    if query_data.get("gpt_name") not in active_gpts:
        print(query_data.get("gpt_name"))
        db_connection = context_db_connection(db_path)
        active_gpts[query_data.get("gpt_name")] = db_connection.read_custom_gpt_by_name(query_data.get("gpt_name"))
    response = {"message":active_gpts[query_data.get("gpt_name")].query(query_data.get("message"))}
    return jsonify(response)