import unittest
from src.data.context_database import context_db_connection
from src.service.customGPT import CustomGPT
from src.service.context import Context
import sqlite3
import numpy as np

from dotenv import load_dotenv
import os
load_dotenv(dotenv_path=os.path.join("..", "auth", ".env"))

# initialize gpt wrapper and get a context for testing
initial_role = """You are a helpful assistant that uses context provided from a FAISS similarity search
              of a technical document to answer questions. Adhere to the text as closely as possible. If the text doesn't have
              The answer then indicate that it does not."""
initial_context = "Answer the question using the text provided context."
client = CustomGPT(name="Testgpt",
                   model="gpt-4-turbo",
                   context_embedding_model="text-embedding-3-small",
                   initial_role=initial_role,
                   initial_context=initial_context,
                   api_key=os.getenv("API_KEY")
                   )

db_name="..//db//Test_7.db"
class MyTestCase(unittest.TestCase):

    def test_read_write_contexts(self):

        db = context_db_connection(db_name)
        db.initialize_with_entries()

        #context = client.add_context_from_docx("Test_1", "..//documents//curtainwall101.docx")

        #print(context.embeddings)
        context_docs=[f"test_{i}" for i in range(10)]
        client.clear_contexts()
        test_contexts : list[Context]=[client.add_context_from_docx(context_doc, f"..//documents//testing//{context_doc}.docx") for context_doc in context_docs]
        for test_context in test_contexts:
            db.write_context(test_context)
            read_context : Context=db.read_context_by_name(test_context.name)

            #evaluate that context objects before and after write are the same
            self.assertEqual(test_context.name, read_context.name)
            self.assertEqual(len(test_context.embeddings), len(read_context.embeddings))

            #verify that querying index before and after write is the same
            self.assertEqual(read_context.index.d, test_context.index.d)
            self.assertEqual(read_context.index.ntotal, test_context.index.ntotal)

            query_vector = client.embeddings.create(input="What is a view factor?",model="text-embedding-3-small").data[0].embedding
            test_context_similarity = test_context.query_similar(query_vector,k=5)
            read_context_similarity = read_context.query_similar(query_vector,k=5)
            #print(f"test_context_similarity: {test_context_similarity}\nread_context_similarity: {read_context_similarity}")
            self.assertEqual(test_context_similarity, read_context_similarity)

        return

    def test_delete_contexts(self):

        context_docs=[f"test_{i}" for i in range(10)]

        #test deleting by id
        db = context_db_connection(db_name)

        client.clear_contexts()
        test_contexts : list[Context]=[client.add_context_from_docx(context_doc, f"..//documents//testing//{context_doc}.docx") for context_doc in context_docs]
        db.initialize_with_entries(contexts=test_contexts)

        deleted_indeces=[]
        for i in range(0,len(test_contexts)):
            delete_index=np.random.randint(1,len(test_contexts)+1)
            if delete_index not in deleted_indeces:
                deleted_indeces.append(delete_index)
                db.delete_context_by_id(delete_index)
                read_context=db.read_context_by_id(delete_index)
                self.assertEqual(read_context is None, True)

        #test deleting by name
        db.initialize_with_entries(contexts=test_contexts)

        deleted_names=[]
        for i in range(0,len(test_contexts)):
            delete_name=test_contexts[np.random.randint(0,len(test_contexts))].name
            if delete_name not in deleted_names:
                deleted_names.append(delete_name)
                db.delete_context_by_name(delete_name)
                read_context=db.read_context_by_name(delete_name)
                self.assertEqual(read_context is None, True)

        return

    def test_read_write_custom_gpts(self):
        # test writing and reading of custom gpt objects
        db = context_db_connection(db_name)

        context_docs=[f"test_{i}" for i in range(10)]
        test_contexts : list[Context]=[client.add_context_from_docx(context_doc, f"..//documents//testing//{context_doc}.docx") for context_doc in context_docs]

        n_trials=10

        np.random.randint(0,len(test_contexts))
        for trial_num in range(n_trials):
            #re initialize database
            db.initialize_with_entries()

            #initialize a new custom gpt object
            initial_role = """You are a helpful assistant that uses context provided from a FAISS similarity search
                          of a technical document to answer questions. Adhere to the text as closely as possible. If the text doesn't have
                          The answer then indicate that it does not."""
            initial_context = "Answer the question using the text provided context."
            new_client = CustomGPT(name="Testgpt",
                               model="gpt-4-turbo",
                               context_embedding_model="text-embedding-3-small",
                               initial_role=initial_role,
                               initial_context=initial_context,
                               api_key=os.getenv("API_KEY")
                               )

            #create connection for additional test queries
            conn = sqlite3.connect(db_name)
            cursor=conn.cursor()

            try:
                #add contexts randomly to created gpt
                contexts_added_to_gpt=set()
                for i in range(np.random.randint(0,len(test_contexts))):
                    index=np.random.randint(0,len(test_contexts))
                    if index not in contexts_added_to_gpt:
                        contexts_added_to_gpt.add(index)
                        new_client.add_context(test_contexts[index])

                #add contexts randomly to database
                contexts_added_to_db=set()
                for i in range(np.random.randint(0,len(test_contexts))):
                    index=np.random.randint(0,len(test_contexts))
                    if index not in contexts_added_to_db:
                        contexts_added_to_db.add(index)
                        db.write_context(test_contexts[index])

                contexts_in_both=contexts_added_to_gpt & contexts_added_to_db

                #query to get current state of db
                cursor.execute('''SELECT * FROM context''')
                results = cursor.fetchall()
                num_initial_contexts_added=len(results)
                self.assertEqual(num_initial_contexts_added, len(contexts_added_to_db))
                print(results)

                #write custom_gpt object to datebase
                db.write_custom_gpt(new_client)

                #verify correct number of contexts have been added
                cursor.execute('''SELECT * FROM context''')
                results = cursor.fetchall()
                self.assertEqual(len(contexts_added_to_gpt)-len(contexts_in_both), len(results) - num_initial_contexts_added)

                #verify similarity between written gpt and retreived from db
                retrieved_gpt=db.read_custom_gpt_by_name(new_client.name)

                self.assertEqual(new_client.name, retrieved_gpt.name)
                self.assertEqual(len(new_client.contexts),len(retrieved_gpt.contexts))

                #verify same contexts are retrieved as were written in
                for context_name in new_client.contexts:
                    self.assertEqual(new_client.contexts[context_name]==retrieved_gpt.contexts[context_name],True)

                # close database connection
                conn.commit()
            except:
                conn.rollback()
            conn.close()

        return

    def test_delete_custom_gpts(self):


        return


if __name__ == '__main__':
    unittest.main()
