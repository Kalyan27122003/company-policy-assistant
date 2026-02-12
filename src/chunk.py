from ingest import load_pdfs
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_documents():
    docs = load_pdfs()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,     
        chunk_overlap=200   
    )

    chunks = splitter.split_documents(docs)

    return chunks

if __name__ == "__main__":
    chunks = chunk_documents()

    print(f"\nTotal chunks created: {len(chunks)}")

    print("\nSample chunk:\n")
    print(chunks[0].page_content[:400])
