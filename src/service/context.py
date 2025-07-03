import numpy as np
import faiss
import os

class Context:
    def __init__(self, name : str, associated_doc_name : str = None, embeddings : list = None, embedding_model : str = "text-embedding-3-small"):
        self.associated_doc_name = associated_doc_name
        self.embeddings = embeddings
        self.embeddings_model = embedding_model
        self.index = None
        self.name = name

        if embeddings is not None: self.generate_faiss_index()

        return

    def query_similar(self, query_vector : np.array, k : int = 10) -> str:
        if all([self.index is not None, self.embeddings is not None]):
            D, I = self.index.search(np.array([query_vector],dtype="float32"),k)
            top_chunks=""
            for i in range(k):
                top_chunks+=("Chunk from FAISS:"+self.embeddings[I[0][i]][0]+"\n")
        else:
            raise RuntimeError("FAISS index has not been initialized or there is no associated text.")
        return top_chunks

    def generate_faiss_index(self) -> None:
        #create and populate faiss index to evaluate similarity against
        if len(self.embeddings)>0:
            self.index=faiss.IndexFlatL2(len(self.embeddings[0][1]))
            vectors = np.array([e[1] / np.linalg.norm(e[1]) for e in self.embeddings])
            self.index.add(vectors)
        else:
            self.index = None
            raise ValueError("Embeddings attribute is empty")
        return

    def load_faiss_index(self, fiass_index_filename : str) -> None:
        self.index=faiss.read_index(fiass_index_filename)
        return

    def set_embeddings(self, embeddings : list) -> None:
        if len(embeddings) > 0:
            self.embeddings = embeddings
        else:
            raise ValueError("Embeddings cannot be empty list.")
        return

    def set_embedding_model(self, embedding_model : str = "text-embedding-3-small") -> None:
        self.embedding_model = embedding_model
        return

    def set_associated_doc_name(self, associated_doc_name : str) -> None:
        self.associated_doc_name = associated_doc_name
        return

    #overrides
    def __eq__(self, other) -> bool:
        if any([self.name != other.name,
                self.associated_doc_name != other.associated_doc_name,
                self.embeddings_model != other.embeddings_model,
                #self.embeddings != other.embeddings
                ]):
            return False
        if any([self.index.d != other.index.d,
                self.index.ntotal != other.index.ntotal]):
            return False
        query_vector=np.random.rand(1,len(self.embeddings[0][1]))[0]
        if self.query_similar(query_vector) != other.query_similar(query_vector):
            return False
        return True

    def __str__(self):
        outstring=f'''Context name: {self.name}\n
        Associated doc name: {self.associated_doc_name}\n
        No. Embeddings: {len(self.embeddings)}\n'''
        return outstring



