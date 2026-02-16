from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

from chunk import chunk_documents


# -------------------------------------------------
# Helper: Infer policy type from filename
# -------------------------------------------------
def infer_policy_type(filename: str) -> str:
    name = filename.lower()

    if "hr" in name or "human" in name:
        return "HR"
    if "it" in name or "security" in name:
        return "IT"
    if "legal" in name or "compliance" in name:
        return "Legal"
    if "travel" in name or "expense" in name:
        return "Travel"
    if "compensation" in name or "salary" in name or "pay" in name:
        return "Compensation"

    return "General"


# -------------------------------------------------
# Build Vector Database
# -------------------------------------------------
def build_vector_db():

    print("Loading and chunking documents...")
    chunks = chunk_documents()

    # Add policy_type metadata to each chunk
    for chunk in chunks:
        source_file = chunk.metadata.get("source", "")
        chunk.metadata["policy_type"] = infer_policy_type(source_file)

    print("Creating embeddings...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    print("Building Chroma vector store...")
    db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="db"
    )

    print("Vector DB created successfully!")
    print("Stored in ./db")


# -------------------------------------------------
# Run standalone
# -------------------------------------------------
if __name__ == "__main__":
    build_vector_db()
