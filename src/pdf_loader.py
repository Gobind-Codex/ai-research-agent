from pathlib import Path

import pymupdf


def extract_pages(pdf_path: str | Path) -> list[dict]:
    """
    Extract text from a PDF, keeping its filename
    and original page number.
    """
    path = Path(pdf_path).expanduser().resolve()

    if not path.is_file():
        raise FileNotFoundError(f"PDF not found: {path}")

    if path.suffix.lower() != ".pdf":
        raise ValueError("The input file must be a PDF.")

    extracted_pages = []

    with pymupdf.open(path) as document:
        for page_index in range(document.page_count):
            page = document.load_page(page_index)
            raw_text = page.get_text("text")
            if not isinstance(raw_text, str):
                raise TypeError("PyMuPDF returned a non-text page value.")
            text = raw_text.strip()

            # Ignore blank pages, but preserve real page numbers.
            if not text:
                continue

            extracted_pages.append(
                {
                    "source": path.name,
                    "page_number": page_index + 1,
                    "text": text,
                }
            )

    return extracted_pages
