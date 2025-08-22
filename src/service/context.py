import numpy as np
import faiss
import os

class Context:
    """
    Context object that wraps a FAISS index for similarity searching.

    Purpose:
    --------
    - Stores text chunks alongside their embeddings.
    - Provides utilities to build, load, and query a FAISS similarity index.
    - Keeps track of metadata such as a human-readable context name,
      the associated document, and the embedding model used.
    - Implements equality checks and string representation for easier debugging.

    Typical usage:
    --------------
    1. Initialize with a name and optional embeddings.
    2. If embeddings are provided, a FAISS index is built automatically.
    3. Call `query_similar(query_vector, k)` to retrieve the top-k
       most similar chunks of text to the query vector.
    """
    def __init__(self, name: str, associated_doc_name: str = None,
                 embeddings: list = None,
                 embedding_model: str = "text-embedding-3-small"):
        # Name of this context object (like an identifier)
        self.associated_doc_name = associated_doc_name  # Optional link to a document
        self.embeddings = embeddings                    # List of (chunk, vector) pairs
        self.embeddings_model = embedding_model         # Model used to create embeddings
        self.index = None                               # Will hold the FAISS index
        self.name = name                                # Human-readable context name

        # If embeddings were provided, immediately build a FAISS index for similarity search
        if embeddings is not None:
            self.generate_faiss_index()

        return

    def query_similar(self, query_vector: np.array, k: int = 10) -> str:
        """
        Query the FAISS index for the k most similar chunks to the given vector.
        Returns a string containing the top matching chunks.
        """
        if all([self.index is not None, self.embeddings is not None]):
            # Perform similarity search in FAISS
            D, I = self.index.search(np.array([query_vector], dtype="float32"), k)

            # Collect the top k text chunks that match
            top_chunks = ""
            for i in range(k):
                top_chunks += ("Chunk from FAISS: " + self.embeddings[I[0][i]][0] + "\n")
        else:
            raise RuntimeError("FAISS index has not been initialized or there is no associated text.")
        return top_chunks

    def generate_faiss_index(self) -> None:
        """
        Create and populate a FAISS index from the stored embeddings.
        Each embedding vector is normalized before insertion.
        """
        if len(self.embeddings) > 0:
            # Create a flat L2 index with dimension equal to embedding length
            self.index = faiss.IndexFlatL2(len(self.embeddings[0][1]))

            # Normalize each embedding vector before adding to index
            vectors = np.array([e[1] / np.linalg.norm(e[1]) for e in self.embeddings])

            # Add the vectors to FAISS index
            self.index.add(vectors)
        else:
            self.index = None
            raise ValueError("Embeddings attribute is empty")
        return

    def load_faiss_index(self, fiass_index_filename: str) -> None:
        """
        Load a FAISS index from file and assign to this context.
        """
        self.index = faiss.read_index(fiass_index_filename)
        return

    def set_embeddings(self, embeddings: list) -> None:
        """
        Replace embeddings list with a new one.
        Does not automatically rebuild FAISS index.
        """
        if len(embeddings) > 0:
            self.embeddings = embeddings
        else:
            raise ValueError("Embeddings cannot be empty list.")
        return

    def set_embedding_model(self, embedding_model: str = "text-embedding-3-small") -> None:
        """
        Update which embedding model this context references.
        """
        self.embedding_model = embedding_model
        return

    def set_associated_doc_name(self, associated_doc_name: str) -> None:
        """
        Update the associated document name metadata.
        """
        self.associated_doc_name = associated_doc_name
        return

    # Overrides
    def __eq__(self, other) -> bool:
        """
        Equality check between two Context objects.
        Compares name, doc name, embedding model, FAISS index size,
        and ensures they return the same similarity results on a random query.
        """
        # Compare metadata fields
        if any([self.name != other.name,
                self.associated_doc_name != other.associated_doc_name,
                self.embeddings_model != other.embeddings_model,
                # self.embeddings != other.embeddings   # skipped, since embeddings can be large
                ]):
            return False

        # Compare FAISS index dimensions and number of vectors
        if any([self.index.d != other.index.d,
                self.index.ntotal != other.index.ntotal]):
            return False

        # Run a test query with a random vector and compare results
        query_vector = np.random.rand(1, len(self.embeddings[0][1]))[0]
        if self.query_similar(query_vector) != other.query_similar(query_vector):
            return False

        return True

    def __str__(self):
        """
        String representation of the Context object.
        Summarizes its name, associated doc, and number of embeddings.
        """
        outstring = f'''Context name: {self.name}\n
        Associated doc name: {self.associated_doc_name}\n
        No. Embeddings: {len(self.embeddings)}\n'''
        return outstring



