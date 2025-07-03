import unittest
from src.service.context import Context
from docx import Document
import tiktoken
import numpy as np

from dotenv import load_dotenv
import os
load_dotenv(dotenv_path=os.path.join("..", "auth", ".env"))

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class MyTestCase(unittest.TestCase):

    def test_create(self):

        doc_name="..//documents//curtainwall101.docx"
        #doc_name="..//documents//test_doc.docx"
        chunk_size=500
        model="gpt-4-turbo"
        embedding_model="text-embedding-3-small"

        #break document into chunks of text to be used in embedding
        doc = Document(doc_name)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        words = text.split()
        print(words[0:100])
        chunks=[]
        chunk=[]
        embedding=tiktoken.encoding_for_model(model)
        for word in words:
            chunk.append(word)
            if len(embedding.encode(" ".join(chunk))) > chunk_size:
                chunks.append(" ".join(chunk[:-1]))
                chunk = [word]
        if chunk:
            chunks.append(" ".join(chunk))

        #embed chunks using specified embedding model
        embeddings=[]
        for chunk in chunks:
            embedding=np.array(client.embeddings.create(input=chunk,model=embedding_model).data[0].embedding,dtype="float32")
            embeddings.append([chunk,embedding])

        #create context object with embeddings
        context = Context("cw101",doc_name,embeddings,embedding_model)

        #query
        query_vector = np.array(client.embeddings.create(input="What is a view factor?",model=embedding_model).data[0].embedding,dtype="float32")
        chunks = context.query_similar(query_vector)
        print(chunks)

        return

    def test_query(self):



        return

if __name__ == '__main__':
    unittest.main()
