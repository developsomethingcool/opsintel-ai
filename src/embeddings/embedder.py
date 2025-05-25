#from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
import torch

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from preprocess.chunker import chunk_all_documents


def create_embeddings():
    chunks = chunk_all_documents()

    #mps acceleration
    if torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # loading bge-small-en embedding model
    model_name = "BAAI/bge-small-en"
    model_kwargs = {"device": device}
    encode_kwargs = {"normalize_embeddings": True}

    # creating HugginFaceEmbeddings
    hf = HuggingFaceEmbeddings(
        model_name=model_name, 
        model_kwargs=model_kwargs, 
        encode_kwargs=encode_kwargs
    )

    # return embedding
    return hf