"""
Filename: create_default_gpts.py
Author: Matthew Gardner
Created: 2025-06-20
Description:
    This script is for populating the database of custom gpt configurations and relevant contexts with default values.

Usage:

"""

from src.service.customGPT import CustomGPT
from src.data.context_database import context_db_connection

import dotenv
import os
from pathlib import Path

dotenv.load_dotenv(Path(__file__).parent.parent.resolve() / "auth" / ".env")

#create contexts and write to database
create_gpts=[{"name":"BuddBot",
              "model":"gpt-4-turbo",
              "context_embedding_model":"text-embedding-3-small",
              "initial_role":"""You are a helpful assistant that uses context provided from a FAISS similarity search
              of a technical document to answer questions. Adhere to the text as closely as possible. If the text doesn't have
              The answer then indicate that it does not.""",
              "initial_context":"Answer the question using the text provided context.",
              "docs_for_context":["curtainwall101.docx"]
              }]

#connect to database
db_connection=context_db_connection(Path(__file__).parent.parent.resolve() / "db" / "gpt-database.db")

#uncomment this section if you'd like to reset the database and enter all new gpts

db_connection.initialize_with_entries()
for gpt in create_gpts:
    created_gpt=CustomGPT(name=gpt["name"],
                   model=gpt["model"],
                   context_embedding_model=gpt["context_embedding_model"],
                   initial_role=gpt["initial_role"],
                   initial_context=gpt["initial_context"],
                   api_key=os.getenv("API_KEY")
                   )
    for doc in gpt["docs_for_context"]:
        if db_connection.read_context_by_name(os.path.basename(doc)) is not None:
            created_gpt.add_context(db_connection.read_context_by_name(os.path.basename(doc)))
        else:
            file_path=Path(__file__).parent.parent.resolve() / "documents" / f"{doc}"
            if file_path.exists() and file_path.suffix.lower() == ".docx":
                created_gpt.add_context_from_docx(context_name=os.path.splitext(os.path.basename(doc))[0],
                                                  doc_name=str(file_path),
                                                  chunk_size=1000)
            elif file_path.exists() and file_path.suffix.lower() == ".pdf":
                pass
            else:
                print(f"{file_path} is not a supported document type or does not exist.")
    try:
        db_connection.write_custom_gpt(created_gpt)
    except Exception as e:
        print(e)
        print(f"{created_gpt.name} was not created; issue adding to database")


#test a gpt
test_gpt:CustomGPT=db_connection.read_custom_gpt_by_name("BuddBot")

print(test_gpt.query("Who does the author of the technical document appear to be?"))