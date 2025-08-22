import unittest
from src.service.customGPT import CustomGPT
from pathlib import Path
import numpy as np

from dotenv import load_dotenv
import os
load_dotenv(dotenv_path=os.path.join("..", "auth", ".env"))

class MyTestCase(unittest.TestCase):
    @unittest.skip("skip test")
    def test_initialization(self):
        initial_role="""You are a helpful assistant that uses context provided from a FAISS similarity search
                      of a technical document to answer questions. Adhere to the text as closely as possible. If the text doesn't have
                      The answer then indicate that it does not."""
        initial_context="Answer the question using the text provided context."
        client = CustomGPT(name="Testgpt",
                           model="gpt-4-turbo",
                           context_embedding_model="text-embedding-3-small",
                           initial_role=initial_role,
                           initial_context=initial_context,
                           api_key=os.getenv("API_KEY")
                            )

        #try querying
        response = client.query("Just say 'hi' back to me, same case and nothing extra.",False)
        self.assertEqual(response,"hi")

        return

    @unittest.skip("Skipping docx embedding and retrieval for now while we test pdf. We know this works.")
    def test_docx_context_embedding(self):
        #initialize gpt wrapper
        initial_role="""You are a helpful assistant that uses context provided from a FAISS similarity search
                      of a technical document to answer questions. Adhere to the text as closely as possible. If the text doesn't have
                      The answer then indicate that it does not."""
        initial_context="Answer the question using the text provided context."
        client = CustomGPT(name="Testgpt",
                           model="gpt-4-turbo",
                           context_embedding_model="text-embedding-3-small",
                           initial_role=initial_role,
                           initial_context=initial_context,
                           api_key=os.getenv("API_KEY")
                            )


        client.add_context_from_docx("Curtain Wall 101","..//documents//curtainwall101.docx")
        self.assertEqual("Curtain Wall 101" in client.contexts,True)

        response=client.query("Who does the author of the technical document appear to be?")

        print(response)

        return

    def test_pdf_context_embedding(self):
        #initialize gpt wrapper
        initial_role="""You are a helpful assistant that uses context provided from a FAISS similarity search
                      of a technical document to answer questions. Adhere to the text as closely as possible. If the text doesn't have
                      The answer then indicate that it does not."""
        initial_context="Answer the question using the text provided context."
        client = CustomGPT(name="Testgpt",
                           model="gpt-4-turbo",
                           context_embedding_model="text-embedding-3-small",
                           initial_role=initial_role,
                           initial_context=initial_context,
                           api_key=os.getenv("API_KEY")
                            )

        pdf_path=Path(__file__).parent.parent / "documents" / "ashrae_fundamentals_F01pCh08.pdf" #if we want to really make this useful, we may want to find a way to combine contexts or at least get closest match. Right now we'd have to include 30 contexts which will all return their closest matches, which gets expensive quickly
        client.add_context_from_pdf(str(pdf_path.stem),str(pdf_path),1500)
        self.assertEqual(str(pdf_path.stem) in client.contexts,True)

        query="Operative temperature is calculated" #we may want to add an intermediate step in pipeline where chatgpt rephrases query in several ways to retrieve relavant text
        query_vector = np.array(client.embeddings.create(input=query,model="text-embedding-3-small").data[0].embedding,dtype="float32")
        chunks = client.contexts[str(pdf_path.stem)].query_similar(query_vector)
        print("Retrieved context: ",chunks)

        response=client.query(query)

        print("Response: ",response) #will have to add functionality on front end for determining how to display equations

        return

if __name__ == '__main__':
    unittest.main()
