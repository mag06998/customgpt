"""
Filename: create_test_context_documents.py
Author: Matthew Gardner
Created: 2025-06-18
Description:
    This script is for generating lorum ipsum documents that can be used for testing context creation. Utilizes the OpenAI api.

Usage:

"""
import dotenv
import os
dotenv.load_dotenv(dotenv_path=os.path.join("..","auth",".env"))
from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
from docx import Document

n_documents = 10 #change this to determine how many documents to create
save_to = "..//documents//testing//"

messages=[{"role":"system","content":"You are a helpful assistant who's primary purpose will be to generate lorum ipsum for test documents. Make it unique every time you are asked."}]

for i in range(n_documents):
    #ask chat gpt
    messages.append({"role":"user","content":"Can you generate about 1000 words of lorum ipsum?"})
    response=client.chat.completions.create(messages=messages,model="gpt-4-turbo")
    messages.append({"role":"system","content":response.choices[0].message.content})

    #write into document
    doc = Document()
    doc.add_paragraph(response.choices[0].message.content)
    doc.save(save_to+f"test_{i}.docx")


