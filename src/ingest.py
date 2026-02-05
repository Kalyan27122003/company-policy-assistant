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

            documents.extend(docs)

    return documents


if __name__ == "__main__":
    docs = load_pdfs()

    print("\nTotal pages loaded:", len(docs))

    print("\nSample content:\n")
    print(docs[0].page_content[:400])
