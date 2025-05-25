from langchain.text_splitter import RecursiveCharacterTextSplitter

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from ingest.pdf_loader import load_all_pdfs

def chunk_all_documents():
    try:
        #load documents
        documents = load_all_pdfs()

        #creating document splitter
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=50
        )

        chunks = text_splitter.split_documents(documents)

        print(f"Total chunks: {len(chunks)}")
        print(f"Example chunk:\n{chunks[1]}")
        return chunks
    except Exception as e:
        print(f"Error during document chunking: {e}")
        return []
