from chunk import chunk_documents
# from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def build_vector_db():
    chunks = chunk_documents()

    print("Creating embeddings...")

    # embedding = OllamaEmbeddings(model="nomic-embed-text")

    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    db = Chroma.from_documents(
        chunks,
        embedding,
        persist_directory="db"
    )


    print("Vector DB created and saved in /db")


if __name__ == "__main__":
    build_vector_db()
