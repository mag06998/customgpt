import unittest
from src.service.customGPT import CustomGPT

from dotenv import load_dotenv
import os
load_dotenv(dotenv_path=os.path.join("..", "auth", ".env"))

class MyTestCase(unittest.TestCase):
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

    def test_context_embedding(self):
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

if __name__ == '__main__':
    unittest.main()
