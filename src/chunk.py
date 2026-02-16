from langchain_text_splitters import MarkdownHeaderTextSplitter
from ingest import load_markdown_files


def chunk_documents():
    """
    Perform semantic chunking based on Markdown headers.
    """
    documents = load_markdown_files()

    headers_to_split_on = [
        ("#", "Title"),
        ("##", "Section"),
        ("###", "Subsection"),
    ]

    splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on
    )

    all_chunks = []

    for doc in documents:
        md_chunks = splitter.split_text(doc.page_content)

        for chunk in md_chunks:
            # Preserve original metadata
            chunk.metadata["source"] = doc.metadata["source"]
            chunk.metadata["file_type"] = doc.metadata["file_type"]

            all_chunks.append(chunk)

    print(f"Created {len(all_chunks)} semantic chunks")

    return all_chunks


# -------------------------------------------------
# Run standalone (for testing)
# -------------------------------------------------
if __name__ == "__main__":
    chunks = chunk_documents()

    print("\nSample semantic chunk:\n")
    print(chunks[0].page_content[:500])
