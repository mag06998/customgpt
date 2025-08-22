import chunk
import unittest

from pyexpat import model

from src.service.context import Context
from docx import Document
from pypdf import PdfReader
import tiktoken
import numpy as np
from pathlib import Path

from dotenv import load_dotenv
import os
load_dotenv(dotenv_path=os.path.join("..", "auth", ".env"))

from openai import OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class MyTestCase(unittest.TestCase):

    @unittest.skip("skip")
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

    def test_create_from_pdf(self):

        pdf_path=Path(__file__).parent.parent / "documents" / "ashrae_fundamentals_F01pCh09.pdf"

        pdf_reader = PdfReader(pdf_path)
        print("Num pages: ", len(pdf_reader.pages))
        page = pdf_reader.pages[0]
        text = page.extract_text()
        print(text)

        text=" ".join([page.extract_text() for page in pdf_reader.pages])

        chunk_size=500
        model="gpt-4-turbo"
        embedding_model="text-embedding-3-small"
        embedding=tiktoken.encoding_for_model(model)

        words=text.split()
        chunks=[]
        chunk=[]
        for word in words:
            chunk.append(word)
            if len(embedding.encode(" ".join(chunk))) > chunk_size:
                chunks.append(" ".join(chunk[:-1]))
                chunk = [word]
        if chunk:
            chunks.append(" ".join(chunk))

        print("Maximum chunk length: ",max([len(chunk) for chunk in chunks]))
        print("Total number of chunks: ",len(chunks))

        embeddings=[]
        for chunk in chunks:
            embedding=np.array(client.embeddings.create(input=chunk,model=embedding_model).data[0].embedding,dtype="float32")
            embeddings.append([chunk,embedding])

        #create context object with embeddings
        context = Context("cw101","ASHRAE fundamentals chapter 9",embeddings,embedding_model)


        #query
        query_vector = np.array(client.embeddings.create(input="How do you determine radiant temperature?",model=embedding_model).data[0].embedding,dtype="float32")
        chunks = context.query_similar(query_vector)
        print(chunks)


        return

    @unittest.skip("skip")
    def look_for_consultant_comments(self):




        pass

if __name__ == '__main__':
    unittest.main()
