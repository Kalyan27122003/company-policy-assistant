import os
from dotenv import load_dotenv

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate

load_dotenv()

# ---------------------------
# CLASSIFIER PROMPT
# ---------------------------

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


# ---------------------------
# LOAD RAG COMPONENTS
# ---------------------------

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

    # Prompt
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


# ---------------------------
# RUN LOCALLY (CLI)
# ---------------------------
if __name__ == "__main__":

    llm, db, PROMPT = load_rag_components()

    print("Company Policy Assistant Ready!")
    print("Type 'exit' to stop")

    while True:
        query = input("\nAsk question: ")

        if query.lower() == "exit":
            break

        # Step 2: Classification
        policy_type = classify_query(llm, query)
        print(f"[Classifier] Policy type → {policy_type}")

        retriever = db.as_retriever(search_kwargs={"k": 3})

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": PROMPT}
        )

        result = qa.invoke({"query": query})

        print("\nAnswer:\n", result["result"])

        print("\nSources:")
        for doc in result["source_documents"]:
            print("-", doc.metadata.get("source", "unknown"))
