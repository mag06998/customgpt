from openai import OpenAI
from src.service.context import Context
import tiktoken
from docx import Document
import numpy as np
import copy

class CustomGPT(OpenAI):

    def __init__(self, name: str, model: str, context_embedding_model : str, initial_role : str, initial_context : str, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.context_embedding_model = context_embedding_model.lower()
        self.contexts={}
        self.chat_history=[{"role":"system","content":initial_role}]
        self.initial_role=initial_role
        self.initial_context=initial_context
        self.model = model

    def add_context_from_docx(self, context_name : str, doc_name : str, chunk_size : int = 500) -> Context:

        #break document into chunks of text to be used in embedding
        doc = Document(doc_name)
        text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
        words = text.split()
        chunks=[]
        chunk=[]
        embedding=tiktoken.encoding_for_model(self.model)
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
            embedding=np.array(super().embeddings.create(input=chunk,model=self.context_embedding_model).data[0].embedding,dtype="float32")
            embeddings.append([chunk,embedding])

        #create context object with embedding
        context = Context(context_name,doc_name,embeddings,self.context_embedding_model)
        self.contexts[context_name] = context

        return context

    def add_context(self, context : Context):
        if (context.name not in self.contexts):
            self.contexts[context.name] = context
        else:
            raise ValueError(f"context {context.name} already exists within this custom GPT")
        return

    def remove_context(self, name : str):
        if (name in self.contexts):
            del self.contexts[name]
        else:
            raise ValueError(f"context {name} does not exist within this custom GPT")
        return

    def clear_contexts(self):
        self.contexts={}
        return

    def clear_chat_history(self):
        self.chat_history=[self.chat_history[0]]
        return

    def query(self, query : str, retrieve_relevant_context=True):
        if retrieve_relevant_context:
            context_text=""
            chunk_idx=0
            for context in self.contexts:
                context_text+=f"Chunk {chunk_idx} from FAISS: "+self.contexts[context].query_similar(super().embeddings.create(input=query,model=self.context_embedding_model).data[0].embedding)+"\n"
                chunk_idx+=1
            query_content = f"Using this initial context:{self.initial_context}\nAnd the following additional context:{context_text}\n Answer the following:{query}"

            #append chat history; distinction must be made between the context fed into the model and what we append to our chat history. Don't want to feed every bit of context in every time because we'll hit the token limit
            print(len(self.chat_history))
            if len(self.chat_history) > 0:
                chat_history_for_query : list = copy.deepcopy(self.chat_history) + [{"role":"user","content":query_content}]
            else:
                chat_history_for_query = [{"role":"user","content":query_content}]
            self.chat_history.append({"role": "user", "content": query})
            response = super().chat.completions.create(model=self.model, messages=chat_history_for_query)
        else:
            self.chat_history.append({"role":"user","content":self.initial_context+"\n"+query})
            response = super().chat.completions.create(model=self.model, messages=self.chat_history)

        self.chat_history.append({"role":response.choices[0].message.role, "content":response.choices[0].message.content})

        return response.choices[0].message.content

    def attributes_as_dict(self):
        return {"name":self.name,
                "model":self.model,}
#overrides



