
from pathlib import Path
import json
import re

import pymupdf


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PAPERS_DIR = PROJECT_ROOT / "data" / "papers"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_FILE = OUTPUT_DIR / "pages.json"


# --------------------------------------------------
# 2. VERIFIED PAPER TITLES
# --------------------------------------------------

PAPER_TITLES = {
    "attention_is_all_you_need.pdf": (
        "Attention Is All You Need"
    ),
    "dynamic_chunking.pdf": (
        "Dynamic Chunking for End-to-End Hierarchical Sequence Modeling"
    ),
    "moshi.pdf": (
        "Moshi: a speech-text foundation model for real-time dialogue"
    ),
}


# --------------------------------------------------
# 3. TEXT CLEANING
# --------------------------------------------------

def clean_text(text: str) -> str:
    """
    Clean extracted PDF text while preserving paragraph breaks.

    This function deliberately avoids aggressive text cleaning
    so that technical content and equations are not damaged.
    """

    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


# --------------------------------------------------
# 4. EXTRACT ONE PDF
# --------------------------------------------------

def extract_pdf(pdf_path: Path) -> dict:
    """
    Extract metadata and page-level text from one PDF.

    Every page retains its original PDF page number.
    """

    pages = []

    with pymupdf.open(pdf_path) as document:

        metadata = document.metadata or {}

        # Prefer verified titles over unreliable PDF metadata.
        title = (
            PAPER_TITLES.get(pdf_path.name)
            or metadata.get("title")
            or pdf_path.stem
        )

        author = metadata.get("author") or "Unknown"

        for page_index, page in enumerate(document):

            # Extract text from the current page.
            raw_text = page.get_text("text", sort=True)

            # Apply conservative text cleaning.
            text = clean_text(raw_text)

            page_record = {
                "page_number": page_index + 1,
                "text": text,
                "character_count": len(text),
                "is_empty": not bool(text),
            }

            pages.append(page_record)

        paper = {
            "source_file": pdf_path.name,
            "title": title,
            "author": author,
            "subject": metadata.get("subject") or "",
            "page_count": len(document),
            "pages": pages,
        }

    return paper


# --------------------------------------------------
# 5. INGEST ALL PAPERS
# --------------------------------------------------

def ingest_all_papers() -> list[dict]:
    """
    Find all PDFs, extract their content,
    and save the results as structured JSON.
    """

    if not PAPERS_DIR.exists():
        raise FileNotFoundError(
            f"Paper directory does not exist: {PAPERS_DIR}"
        )

    pdf_files = sorted(PAPERS_DIR.glob("*.pdf"))

    if not pdf_files:
        raise FileNotFoundError(
            f"No PDF files found in: {PAPERS_DIR}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    papers = []
    errors = []

    print("=" * 60)
    print("THE RELAY - PDF INGESTION")
    print("=" * 60)

    for pdf_path in pdf_files:

        print(f"\nExtracting: {pdf_path.name}")

        try:
            paper = extract_pdf(pdf_path)
            papers.append(paper)

            empty_pages = sum(
                page["is_empty"]
                for page in paper["pages"]
            )

            total_characters = sum(
                page["character_count"]
                for page in paper["pages"]
            )

            print(f"  Title: {paper['title']}")
            print(f"  Pages: {paper['page_count']}")
            print(f"  Characters: {total_characters}")
            print(f"  Empty pages: {empty_pages}")

        except Exception as error:

            errors.append({
                "file": pdf_path.name,
                "error": str(error),
            })

            print(f"  ERROR: {error}")

    # Avoid saving an incomplete dataset if any PDF failed.
    if errors:
        raise RuntimeError(
            f"Failed to extract {len(errors)} PDF(s): {errors}"
        )

    # Save extracted records.
    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        json.dump(
            papers,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------
    # 6. FINAL INGESTION SUMMARY
    # --------------------------------------------------

    total_pages = sum(
        paper["page_count"]
        for paper in papers
    )

    total_characters = sum(
        page["character_count"]
        for paper in papers
        for page in paper["pages"]
    )

    total_empty_pages = sum(
        page["is_empty"]
        for paper in papers
        for page in paper["pages"]
    )

    print("\n" + "=" * 60)
    print("INGESTION COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print(f"Papers processed: {len(papers)}")
    print(f"Total pages: {total_pages}")
    print(f"Total characters: {total_characters}")
    print(f"Total empty pages: {total_empty_pages}")

    print(
        f"Output file: {OUTPUT_FILE.relative_to(PROJECT_ROOT)}"
    )

    return papers


# --------------------------------------------------
# 7. ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":
    ingest_all_papers()
