from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate 
from langgraph.graph import START, StateGraph
from typing_extensions import List, TypedDict
from langchain_core.documents import Document

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from vectorestore.chroma_client import get_vectorestore

vectorestore = get_vectorestore()

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=(
        "Answer the question using only the information provided in the context below.\n"
        "If the answer cannot be found in the context, say you don't know.\n\n"
        "Context:\n{context}\n\n"
        "Question:\n{question}\n\n"
        "Answer:"
    )
)

llm = ChatOllama(model="llama3.1")

# define state for application
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

# define applicatio steps
def retrieve(state: State):
    retrieved_docs = vectorestore.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()

questions = [{"question": "What was mention about economics?"}, 
             {"question": "Who is the author of the document?"},
             {"question": "What were the propoused solutions?"}
             ]

for question in questions:
    response = graph.invoke(question)
    print(response["answer"])
    print("========================")