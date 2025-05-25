from langchain_chroma import Chroma

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from embeddings.embedder import create_embeddings
from ingest.pdf_loader import load_all_pdfs

def get_vectorestore():
    # create embeddings
    hf_embeddings = create_embeddings()

    # check if vectorestore exists
    if os.path.exists("src/db/chroma.sqlite3"):
        vector_store = Chroma(
            collection_name = "documents",
            embedding_function = hf_embeddings,
            persist_directory = "src/db/"
        )
        print(f"Vectorestore is loaded!")

    else:
        # load all documents
        documents = load_all_pdfs()

        vector_store = Chroma(
            collection_name = "documents",
            embedding_function = hf_embeddings,
            persist_directory = "src/db/"
        )

        # add documents into vectorestore
        vector_store.add_documents(documents)

        print(f"Vectorestore is created!")
    
    return vector_store
    

