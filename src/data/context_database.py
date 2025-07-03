import os
import sqlite3
from src.service.context import Context
from src.service.customGPT import CustomGPT
import faiss
import numpy as np
from pathlib import Path

class context_db_connection:
    def __init__(self,db_name):
        self.db_name = db_name
        return

    def delete_context_by_id(self, id : int) -> None:

        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        try:
            cursor.execute('''PRAGMA foreign_keys = ON;''') #pragma stands for program directive; foreign key constraints (and therefore cascading aren't enabled by default so must specify)
            cursor.execute('''DELETE FROM context WHERE id = ?''', (id,))
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

        return

    def delete_context_by_name(self, name : str) -> None:
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        try:
            cursor.execute(
                '''PRAGMA foreign_keys = ON;''')  # pragma stands for program directive; foreign key constraints (and therefore cascading aren't enabled by default so must specify)
            cursor.execute('''DELETE
                              FROM context
                              WHERE name = ?''', (name,))
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

        return

    def read_context_by_id(self,id : int) -> Context:

        #variables
        context=None

        #initialize connection
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        try:
            #initial query
            cursor.execute('''SELECT * FROM context WHERE id = ?''',(id,))
            query_context = cursor.fetchone()
            print(f"query_context: {query_context}")

            if query_context is not None:
                cursor.execute('''SELECT * FROM context_embeddings WHERE context_id = ? ORDER BY chunk_index ASC''',(id,))
                query_embeddings = cursor.fetchall()
                print("length of query_text: ", len(query_embeddings))

                #create context object to return and load data into if from query
                context = Context(name=query_context[1],associated_doc_name=query_context[2])
                context.load_faiss_index(query_context[3])

                context.set_embeddings([[e[3],np.array(e[4].split(","),dtype="float32")] for e in query_embeddings])
            #close database connection
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

        return context

    def read_context_by_name(self, name : str) -> Context:

        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        cursor.execute('''SELECT * FROM context WHERE name = ?''',(name,))
        query_context = cursor.fetchone()

        if query_context is not None:
            print(f"query_context: {query_context}")

            cursor.execute('''SELECT * FROM context_embeddings WHERE context_id = ? ORDER BY chunk_index ASC''',(query_context[0],))
            query_embeddings = cursor.fetchall()
            print("length of query_text: ", len(query_embeddings))

            #create context object to return and load data into if from query
            context = Context(name=query_context[1],associated_doc_name=query_context[2])
            context.load_faiss_index(query_context[3])

            context.set_embeddings([[e[3],np.array(e[4].split(","),dtype="float32")] for e in query_embeddings])
        else:
            context = None

        # close database connection
        conn.commit()
        conn.close()

        return context

    def write_context(self,context : Context) -> int:

        #input validation
        if any([context.embeddings is None, context.index is None, context.associated_doc_name is None]):
            raise ValueError(f"Context {context.name} does not have all initialized attributes to be written to database. One of context.text, context.embeddings, context.index, context.associated_doc_name are None")

        #variables
        id=None

        #initialize database connection
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        try:
            # save faiss index to file
            faiss_file=str(Path(__file__).parent.parent.parent / "faiss" / f"{context.name}.faiss")
            print("Faiss file: ",faiss_file)
            faiss.write_index(context.index, faiss_file)

            # insert into context table
            cursor.execute('''INSERT INTO context (name, origin_filename, faiss_index_filename) VALUES (?, ?, ?)''', (context.name, context.associated_doc_name, faiss_file))

            # insert embeddings into context_embeddings database
            id = cursor.lastrowid
            for i in range(len(context.embeddings)):
                chunk_text = context.embeddings[i][0]
                embedding_as_string = ','.join(map(str,context.embeddings[i][1]))
                cursor.execute('''INSERT INTO context_embeddings (context_id, chunk_index, chunk_text, embedding_vector) VALUES (?, ?, ?, ?)''',
                               (id,i,chunk_text,embedding_as_string))
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

        return id

    def delete_custom_gpt_by_id(self, id : int) -> None:



        return

    def delete_custom_gpt_by_name(self, name : str) -> None:

        return

    def read_custom_gpt_by_id(self, id : int) -> CustomGPT:

        return

    def get_all_gpt_info(self) -> list:

        #begin connection
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        try:
            cursor.execute('''SELECT * FROM custom_gpt''')
            gpts = cursor.fetchall()
            return gpts
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()
        return

    def read_custom_gpt_by_name(self, name : str) -> CustomGPT:

        #variables
        custom_gpt=None

        #initialize database connection
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        try:
            cursor.execute('''SELECT * FROM custom_gpt WHERE name = ?''',(name,))
            result = cursor.fetchone()

            if result is not None:
                gpt_id = result[0]
                #create custom gpt object to return
                custom_gpt = CustomGPT(name=result[1],
                                       model=result[2],
                                       context_embedding_model=result[3],
                                       initial_role=result[4],
                                       initial_context=result[5],
                                       api_key=os.getenv("API_KEY")
                                       )
                #populate object with contexts
                cursor.execute('''SELECT context_id FROM gpt_context WHERE gpt_id = ?''',(gpt_id,))
                context_ids = cursor.fetchall()
                if context_ids is not None:
                    for context_id in context_ids:
                        custom_gpt.add_context(self.read_context_by_id(context_id[0]))
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

        return custom_gpt

    def write_custom_gpt(self, custom_gpt : CustomGPT) -> int:

        #variables
        id=None

        #initialize connection
        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        try:
            #insert contexts into context database
            context_ids=[]
            for context in custom_gpt.contexts:
                #should check if context is already in database, have some logic for determining if the contexts are actually equivalent if it is
                #if they have same name but are not equivalent, write in the new context with a modified name
                cursor.execute('''SELECT *
                                  FROM context
                                  WHERE name = ?''', (custom_gpt.contexts[context].name,))
                context_query=cursor.fetchone()
                if context_query is not None:
                    print(context_query)
                    context_with_same_name=self.read_context_by_id(context_query[0])
                    if context_with_same_name == custom_gpt.contexts[context]:
                        context_ids.append(context_query[0])
                    else:
                        context.name=context.name+"_"+custom_gpt.name
                        context_ids.append(self.write_context(custom_gpt.contexts[context]))
                else:
                    context_ids.append(self.write_context(custom_gpt.contexts[context]))

            #insert into custom_gpt
            cursor.execute('''INSERT INTO custom_gpt (name, model, context_embedding_model, initial_role, initial_context) VALUES (?, ?, ?, ?, ?)''',
                           (custom_gpt.name,
                            custom_gpt.model,
                            custom_gpt.context_embedding_model,
                            custom_gpt.initial_role,
                            custom_gpt.initial_context
                            ))
            id=cursor.lastrowid

            #associate contexts with gpt
            for context_id in context_ids:
                cursor.execute('''INSERT INTO gpt_context (gpt_id, context_id) VALUES (?, ?)''',
                               (id,context_id))

            # close database connection
            conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            conn.close()

        return id

    def initialize_with_entries(self, contexts : list[Context] = None, custom_gpts : list[CustomGPT] = None) -> None:
        """
        Initializes the database. Any GPT objects and context provided will be used to populate the database. Context objects associated with a particular gpt will also be added.
        This will completely recreate the database; so don't call this function if there is necessary information in the existing database with the same name.
        :param contexts: list of Context objects
        :param custom_gpts: list of CustomGPT objects
        :return: None
        """

        conn = sqlite3.connect(self.db_name)
        cursor = conn.cursor()

        #create tables
        cursor.execute('''DROP TABLE IF EXISTS custom_gpt''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS custom_gpt (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        model TEXT NOT NULL,
        context_embedding_model TEXT NOT NULL,
        initial_role TEXT,
        initial_context TEXT
        )
        ''')
        cursor.execute('''DROP TABLE IF EXISTS context''')
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS context (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        origin_filename TEXT NOT NULL,
        faiss_index_filename TEXT NOT NULL
        )
        ''')
        cursor.execute('''DROP TABLE IF EXISTS context_embeddings''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS context_embeddings (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        context_id INTEGER NOT NULL,
        chunk_index INTEGER NOT NULL,
        chunk_text TEXT NOT NULL,
        embedding_vector TEXT NOT NULL,
        CONSTRAINT fk_context
            FOREIGN KEY(context_id) REFERENCES context(id)
            ON DELETE CASCADE
        )
        ''')
        cursor.execute('''DROP TABLE IF EXISTS gpt_context''')
        cursor.execute('''CREATE TABLE IF NOT EXISTS gpt_context (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        gpt_id INTEGER NOT NULL,
        context_id INTEGER NOT NULL,
        CONSTRAINT fk_context
            FOREIGN KEY(context_id) REFERENCES context (id)
            ON DELETE CASCADE,
        CONSTRAINT fk_gpt_context
            FOREIGN KEY(gpt_id) REFERENCES custom_gpt (id)
            ON DELETE CASCADE
        )
        ''')

        #insert data into tables
        if contexts:
            for context in contexts:
                self.write_context(context)

        if custom_gpts:
            for custom_gpt in custom_gpts:
                self.write_custom_gpt(custom_gpt)

        #close database connection
        conn.commit()
        conn.close()

        return