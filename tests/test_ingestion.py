import fitz

from src.pdf_loader import extract_pages
from src.chunking import chunk_pages


def test_pdf_extraction_and_chunking(tmp_path):
    # Create a temporary test PDF.
    pdf_path = tmp_path / "sample.pdf"

    with fitz.open() as document:
        page = document.new_page()
        page.insert_text(
            (72, 72),
            "Artificial intelligence research. " * 40,
        )
        document.save(pdf_path)

    # Test PDF extraction.
    pages = extract_pages(pdf_path)

    assert len(pages) == 1
    assert pages[0]["source"] == "sample.pdf"
    assert pages[0]["page_number"] == 1
    assert "Artificial intelligence" in pages[0]["text"]

    # Test chunking and metadata preservation.
    chunks = chunk_pages(
        pages,
        chunk_size=100,
        overlap=20,
    )

    assert len(chunks) > 1
    assert chunks[0]["chunk_index"] == 0
    assert all(c["page_number"] == 1 for c in chunks)
    assert all(c["source"] == "sample.pdf" for c in chunks)
    assert all(c["text"] for c in chunks)


def test_invalid_chunk_settings():
    pages = [
        {
            "source": "sample.pdf",
            "page_number": 1,
            "text": "Some sample text.",
        }
    ]

    try:
        chunk_pages(pages, chunk_size=100, overlap=100)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected ValueError for invalid overlap")
