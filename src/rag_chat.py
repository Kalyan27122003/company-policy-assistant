from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
import os

from langchain_classic.chains.retrieval_qa.base import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.embeddings import HuggingFaceEmbeddings

from dotenv import load_dotenv
load_dotenv()


def load_rag():

    # Embeddings
    
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # Vector DB

    db = Chroma(
        persist_directory="db",
        embedding_function=embeddings
    )

    retriever = db.as_retriever(search_kwargs={"k": 3})

    # LLM (Groq)

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

    # QA Chain

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )

    return qa


# Run locally

if __name__ == "__main__":

    qa = load_rag()

    print("Company Policy Assistant Ready!")
    print("Type 'exit' to stop")

    while True:
        query = input("\nAsk question: ")

        if query.lower() == "exit":
            break

        result = qa.invoke({"query": query})

        print("\nAnswer:\n", result["result"])

        print("\nSources:")
        for doc in result["source_documents"]:
            print("-", doc.metadata["source"])
        
