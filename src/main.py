from pathlib import Path

from src.pdf_loader import extract_pages
from src.chunking import chunk_pages
from src.vector_store import index_chunks, search_chunks

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main():
    pdf_path = PROJECT_ROOT / "data" / "sample.pdf"

    # Step 1: Extract the PDF.
    pages = extract_pages(pdf_path)
    print(f"Extracted {len(pages)} pages.")

    # Step 2: Split extracted text into chunks.
    chunks = chunk_pages(pages)
    print(f"Created {len(chunks)} chunks.")

    # Step 3: Generate embeddings and index the chunks.
    count = index_chunks(chunks)
    print(f"Indexed {count} chunks.")

    # Step 4: Ask a research question.
    question = input("\nEnter your research question: ").strip()

    # Step 5: Retrieve relevant evidence.
    matches = search_chunks(question, top_k=5)

    if not matches:
        print("No matching passages found.")
        return

    print("\n--- Retrieved Evidence ---")

    for rank, match in enumerate(matches, start=1):
        print(f"\nResult {rank}")
        print(f"Source: {match['source']}")
        print(f"Page: {match['page_number']}")
        print(f"Distance: {match['distance']:.4f}")
        print(match["text"])


if __name__ == "__main__":
    main()
