import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from vectorestore.chroma_client import get_vectorestore

from langchain_ollama import ChatOllama
from langchain.tools.retriever import create_retriever_tool
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.messages import convert_to_messages, ToolMessage

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt import tools_condition


def build_agentic_rag_graph(vectorestore):
    
    # Loading vectorestore and retriever
    #vectorestore = get_vectorestore()
    retriever = vectorestore.as_retriever(search_kwargs={"k": 3})

        
    # definition of llms
    chat_llm = ChatOllama(model="qwen2.5:14b")
    grader_llm = ChatOllama(model="qwen2.5:14b")


    # creation and testing of the tool
    retriever_tool = create_retriever_tool(
        retriever,
        "retriever of the report",
        "Search and retrieve what the Dragi Report includes regarding the geopolitical, economic, technological, and security-related challenges facing the European Union, offering strategic recommendations to strengthen the EU's resilience, competitiveness, and global leadership.",
    )

    def generate_query_or_respond(state: MessagesState):
        """Call the model to generate a response based on the current state. Given
        the question, it will decide to retrieve using the retriever tool, or simply respond to the user.
        """
        response = (
            chat_llm.bind_tools([retriever_tool]).invoke(state["messages"])
        )
        return  {"messages": state["messages"] + [response]}

    GRADE_PROMPT = (
        "You are a grader assessing relevance of a retrieved document to a user question. \n "
        "Here is the retrieved document: \n\n {context} \n\n"
        "Here is the user question: {question} \n"
        "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
        "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
    )

    class GradeDocuments(BaseModel):
        """Grade documents using a binary score for relevance check."""

        binary_score: str = Field(
            description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
        )

    def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question"]:
        """Determine whether the retrieved documents are relevant to the question."""
        question = state["messages"][0].content
        #context = state["messages"][-1].content

        # Robust retrieval of the last ToolMessage
        context = next(
            (msg.content for msg in reversed(state["messages"]) if isinstance(msg, ToolMessage)),
            ""
        )

        prompt = GRADE_PROMPT.format(question=question, context=context)
        response = (
            grader_llm.with_structured_output(GradeDocuments).invoke(
                [{"role": "user", "content": prompt}]
            )
        )

        score = response.binary_score

        if score == "yes":
            return "generate_answer"
        else:
            return "rewrite_question"

    # Rewrite question

    REWRITE_PROMPT = (
        "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
        "Here is the initial question:"
        "\n ------- \n"
        "{question}"
        "\n ------- \n"
        "Formulate an improved question:"
    )

    def rewrite_question(state: MessagesState):
        """Rewrite the original user question."""
        rewrite_count = state.get("rewrite_count", 0)

        if rewrite_count >= 3:
                # Give up after 3 rewrites
                final_message = {
                    "role": "assistant",
                    "content": "Sorry, I couldn't find a relevant answer after several attempts.",
                }
                
                return ({
                    "messages": state["messages"] + [final_message],
                    "rewrite_count": rewrite_count
                }, END)

        messages = state["messages"]
        question = messages[0].content
        prompt = REWRITE_PROMPT.format(question=question)
        response = chat_llm.invoke([{"role": "user", "content": prompt}])
        
        new_message = {
            "role": "user",
            "content": response.content,
        }
        return {
            "messages": state["messages"] + [new_message],
            "rewrite_count": rewrite_count + 1
        }
        #return {"messages": [{"role": "user", "content": response.content}]}


    # Generate the answer

    GENERATE_PROMPT = (
        "You are an assistant for question-answering tasks. "
        "Use the following pieces of retrieved context to answer the question. "
        "If you don't know the answer, just say that you don't know. "
        "Use three sentences maximum and keep the answer concise.\n"
        "Question: {question} \n"
        "Context: {context}"
    )

    def generate_answer(state: MessagesState):
        """Generate an answer."""
        question = state["messages"][0].content
        
        #context = state["messages"][-1].content
        context = next(
            (msg.content for msg in reversed(state["messages"]) if isinstance(msg, ToolMessage)),
            ""
        )

        prompt = GENERATE_PROMPT.format(question=question, context=context)
        response = chat_llm.invoke([{"role": "user", "content": prompt}])
        #return {"messages": [response]}
        return {"messages": state["messages"] + [response]}


    # Assemble the graph
    workflow = StateGraph(MessagesState)
    workflow.add_node(generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node(rewrite_question)
    workflow.add_node(generate_answer)

    workflow.add_edge(START, "generate_query_or_respond")
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {
            "tools": "retrieve",
            END: END,
        },
    )

    # Edges taken after the 'action' node is called

    workflow.add_conditional_edges(
        "retrieve",
        grade_documents,
        {
            "generate_answer": "generate_answer",
            "rewrite_question": "rewrite_question",
        }
    )

    workflow.add_edge("generate_answer", END)
    workflow.add_edge("rewrite_question", "generate_query_or_respond")

    #Compile graph
    graph = workflow.compile()

    return graph


def run_agentic_rag(graph, state):
    
    result = graph.invoke(state)
    # Parsing result messages
    answer = result["messages"][-1].content if result["messages"] else "No answer found."
    return answer

# Visualize the graph
#=================================
#Create the graph structure and store it in .png
# image_data = graph.get_graph(xray=True).draw_mermaid_png()
# file_path = "agentic_rag_graph.png"
# with open(file_path, "wb") as file:
#     file.write(image_data)
#=================================