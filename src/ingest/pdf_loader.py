from langchain_community.document_loaders import PyPDFLoader
import os

directory_path = "data/raw"

def load_all_pdfs(directory_path="data/raw"):
    list_of_all_docs = []
    list_of_files = os.listdir(directory_path)
    for file in list_of_files:
        file_path = os.path.join(directory_path, file)
        try: 
            loader = PyPDFLoader(file_path)
            docs = loader.load()
            for doc in docs:
                doc.metadata['source_file'] = file
            list_of_all_docs.extend(docs)
        except Exception as e:
            print(f"Failed to process the file {file}: {e}")

    return list_of_all_docs