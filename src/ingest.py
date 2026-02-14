import os
from langchain_community.document_loaders import PyPDFLoader

DATA_PATH = "data"


def load_pdfs():
    documents = []

    for file in os.listdir(DATA_PATH):
        if file.endswith(".pdf"):
            path = os.path.join(DATA_PATH, file)

            print(f"Loading {file}...")

            loader = PyPDFLoader(path)
            docs = loader.load()

            for doc in docs:
                doc.metadata["source"] = file
                doc.metadata["policy_type"] = infer_policy_type(file)
            documents.extend(docs)

    return documents
def infer_policy_type(filename: str):
    name = filename.lower()
    if "hr" in name:
        return "HR"
    if "it" in name or "security" in name:
        return "IT"
    if "legal" in name or "compliance" in name:
        return "Legal"
    if "travel" in name or "expense" in name:
        return "Travel"
    if "compensation" in name or "pay" in name:
        return "Compensation"
    return "General"


if __name__ == "__main__":
    docs = load_pdfs()

    print("\nTotal pages loaded:", len(docs))

    print("\nSample content:\n")
    print(docs[0].page_content[:400])
