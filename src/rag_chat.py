import os
from typing import TypedDict, List

from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

from langgraph.graph import StateGraph, END

load_dotenv()

# =========================================================
# 1. QUERY CLASSIFIER PROMPT + FUNCTION
# =========================================================

CLASSIFIER_PROMPT = PromptTemplate(
    template="""
You are a classifier for company policy questions.

Classify the user question into ONE of the following categories:
HR, IT, Legal, Travel, Compensation, General

Return ONLY the category name.

Question:
{question}

Category:
""",
    input_variables=["question"]
)


def classify_query(llm, question: str) -> str:
    response = llm.invoke(
        CLASSIFIER_PROMPT.format(question=question)
    )

    category = response.content.strip()

    allowed = {"HR", "IT", "Legal", "Travel", "Compensation", "General"}
    if category not in allowed:
        return "General"

    return category


# =========================================================
# 2. LOAD SHARED RAG COMPONENTS
# =========================================================

def load_rag_components():
    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # Vector DB
    db = Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )

    # LLM
    llm = ChatGroq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.1-8b-instant"
    )

    # RAG Prompt
    prompt_template = """
You are a company policy assistant.

Answer ONLY using the provided context.
If the answer is not in the context, say:
"I don't know based on the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""

    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )

    return llm, db, PROMPT


# Initialize shared components (IMPORTANT)
llm, db, PROMPT = load_rag_components()

# =========================================================
# 3. LANGGRAPH STATE
# =========================================================

class RAGState(TypedDict):
    question: str
    policy_type: str
    answer: str
    sources: List[str]


# =========================================================
# 4. LANGGRAPH NODES (AGENTS)
# =========================================================

# ---- Classifier Agent ----
def classifier_node(state: RAGState):
    policy_type = classify_query(llm, state["question"])
    return {
        "policy_type": policy_type
    }

def is_policy_question(policy_type: str) -> bool:
    return policy_type != "General"

# ---- Answer + Retrieval Agent ----
def answer_node(state: RAGState):
    policy_type = state["policy_type"]
    question = state["question"]

    # 🚫 Guardrail: non-policy questions
    if policy_type == "General":
        return {
            "answer": "I don't know based on the provided documents.",
            "sources": []
        }

    retriever = db.as_retriever(
        search_kwargs={
            "k": 4,
            "filter": {"policy_type": policy_type}
        }
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    result = qa.invoke({"query": question})

    sources = [
        doc.metadata.get("source", "unknown")
        for doc in result["source_documents"]
    ]

    return {
        "answer": result["result"],
        "sources": sources
    }



# =========================================================
# 5. BUILD & COMPILE LANGGRAPH
# =========================================================

graph = StateGraph(RAGState)

graph.add_node("classifier", classifier_node)
graph.add_node("answer", answer_node)

graph.set_entry_point("classifier")

graph.add_edge("classifier", "answer")
graph.add_edge("answer", END)

rag_graph = graph.compile()

# Make graph importable in Streamlit
__all__ = ["rag_graph"]


# =========================================================
# 6. RUN LOCALLY (CLI TEST)
# =========================================================

if __name__ == "__main__":
    print("Agentic Company Policy Assistant (LangGraph)")
    print("Type 'exit' to stop")

    while True:
        question = input("\nAsk question: ")

        if question.lower() == "exit":
            break

        result = rag_graph.invoke({
            "question": question
        })

        print("\n[Policy Type]:", result["policy_type"])
        print("\nAnswer:\n", result["answer"])

        print("\nSources:")
        for src in result["sources"]:
            print("-", src)
