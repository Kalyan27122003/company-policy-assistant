import os
from langchain_core.documents import Document

# Folder where markdown files are stored
DATA_PATH = "data_markdown"


def load_markdown_files():
    """
    Load all Markdown (.md) files from data_markdown directory
    and convert them into LangChain Document objects.
    """
    documents = []

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Markdown directory not found: {DATA_PATH}")

    for file in os.listdir(DATA_PATH):
        if file.lower().endswith(".md"):
            file_path = os.path.join(DATA_PATH, file)

            print(f"Loading Markdown file: {file}")

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            doc = Document(
                page_content=content,
                metadata={
                    "source": file,
                    "file_type": "markdown"
                }
            )

            documents.append(doc)

    if not documents:
        raise ValueError("No Markdown files found in data_markdown folder.")

    return documents


# -------------------------------------------------
# Run standalone (for testing)
# -------------------------------------------------
if __name__ == "__main__":
    docs = load_markdown_files()

    print(f"\nTotal markdown documents loaded: {len(docs)}")
    print("\nSample content:\n")
    print(docs[0].page_content[:500])
