import os
from docx import Document
from markdownify import markdownify as md

# ==============================
# CHANGE THIS TO YOUR SYSTEM PATH
# ==============================
SOURCE_DIR = r"C:\\Users\\kalya\\OneDrive\\Desktop\\company policies\\Company Policies"

# Where markdown files will be saved (inside project)
OUTPUT_DIR = "data_markdown"


def docx_to_markdown(docx_path: str) -> str:
    """
    Convert a Word document to Markdown text.
    """
    document = Document(docx_path)
    paragraphs = []

    for para in document.paragraphs:
        text = para.text.strip()
        if text:
            paragraphs.append(text)

    raw_text = "\n\n".join(paragraphs)
    markdown_text = md(raw_text)

    return markdown_text


def load_docs_from_system():
    if not os.path.exists(SOURCE_DIR):
        raise FileNotFoundError(f"Source folder not found: {SOURCE_DIR}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for file in os.listdir(SOURCE_DIR):
        if file.lower().endswith(".docx"):
            input_path = os.path.join(SOURCE_DIR, file)
            output_file = file.replace(".docx", ".md")
            output_path = os.path.join(OUTPUT_DIR, output_file)

            print(f"Processing: {file}")

            markdown_content = docx_to_markdown(input_path)

            with open(output_path, "w", encoding="utf-8") as f:
                f.write(f"# {file.replace('.docx','')}\n\n")
                f.write(markdown_content)

    print("\n✅ All system documents loaded and converted to Markdown.")


if __name__ == "__main__":
    load_docs_from_system()
