from chunk import chunk_documents
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

def build_vector_db():
    chunks = chunk_documents()

    print("Creating embeddings...")

    embedding = OllamaEmbeddings(model="nomic-embed-text")

    db = Chroma.from_documents(
        chunks,
        embedding,
        persist_directory="db"
    )

    db.persist()

    print("Vector DB created and saved in /db")


if __name__ == "__main__":
    build_vector_db()
