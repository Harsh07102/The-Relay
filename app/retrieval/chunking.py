
from pathlib import Path
import argparse
import json
import re


# --------------------------------------------------
# 1. PROJECT PATHS
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "processed" / "pages.json"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"


# --------------------------------------------------
# 2. DEFAULT CHUNKING CONFIGURATION
# --------------------------------------------------

DEFAULT_CHUNK_SIZE = 350
DEFAULT_OVERLAP = 60


# --------------------------------------------------
# 3. CREATE A SAFE PAPER IDENTIFIER
# --------------------------------------------------

def create_slug(filename: str) -> str:
    """Create a readable identifier from a PDF filename."""

    name = Path(filename).stem.lower()
    return re.sub(r"[^a-z0-9]+", "-", name).strip("-")


# --------------------------------------------------
# 4. CHUNK A SINGLE PAGE
# --------------------------------------------------

def chunk_text(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict]:
    """
    Split text into overlapping word-based chunks.

    The end_word index is exclusive.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0:
        raise ValueError("overlap cannot be negative")

    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    words = text.split()

    chunks = []
    start = 0
    chunk_index = 1

    while start < len(words):

        end = min(start + chunk_size, len(words))

        chunk_words = words[start:end]

        chunk = {
            "chunk_index": chunk_index,
            "start_word": start,
            "end_word": end,
            "word_count": len(chunk_words),
            "text": " ".join(chunk_words),
        }

        chunks.append(chunk)

        # Move forward while retaining the configured overlap.
        if end == len(words):
            break

        start = end - overlap
        chunk_index += 1

    return chunks


# --------------------------------------------------
# 5. CHUNK ALL PAPER PAGES
# --------------------------------------------------

def create_chunks(
    papers: list[dict],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict]:
    """
    Chunk every page independently.

    Each resulting chunk retains its paper and page metadata.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and smaller than chunk_size"
        )

    all_chunks = []

    for paper in papers:

        source_file = paper["source_file"]
        title = paper["title"]
        paper_slug = create_slug(source_file)

        for page in paper["pages"]:

            page_number = page["page_number"]
            page_text = page["text"]

            if not page_text.strip():
                continue

            page_chunks = chunk_text(
                text=page_text,
                chunk_size=chunk_size,
                overlap=overlap,
            )

            for chunk in page_chunks:

                chunk_id = (
                    f"{paper_slug}-"
                    f"p{page_number:04d}-"
                    f"c{chunk['chunk_index']:04d}"
                )

                record = {
                    "chunk_id": chunk_id,
                    "title": title,
                    "source_file": source_file,
                    "page_number": page_number,
                    "citation": f"[{title}, p. {page_number}]",
                    "chunk_index": chunk["chunk_index"],
                    "start_word": chunk["start_word"],
                    "end_word": chunk["end_word"],
                    "word_count": chunk["word_count"],
                    "text": chunk["text"],
                }

                all_chunks.append(record)

    return all_chunks


# --------------------------------------------------
# 6. LOAD, CHUNK AND SAVE
# --------------------------------------------------

def run_chunking(
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> list[dict]:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Ingestion output not found: {INPUT_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with INPUT_FILE.open("r", encoding="utf-8") as file:
        papers = json.load(file)

    chunks = create_chunks(
        papers=papers,
        chunk_size=chunk_size,
        overlap=overlap,
    )

    output_file = (
        OUTPUT_DIR
        / f"chunks_{chunk_size}_{overlap}.json"
    )

    output_data = {
        "configuration": {
            "chunk_size_words": chunk_size,
            "overlap_words": overlap,
            "unit": "words",
        },
        "total_chunks": len(chunks),
        "chunks": chunks,
    }

    with output_file.open("w", encoding="utf-8") as file:
        json.dump(
            output_data,
            file,
            indent=2,
            ensure_ascii=False,
        )

    # --------------------------------------------------
    # 7. GENERATE SUMMARY
    # --------------------------------------------------

    total_words = sum(
        chunk["word_count"] for chunk in chunks
    )

    paper_summary = {}

    for chunk in chunks:
        filename = chunk["source_file"]

        paper_summary[filename] = (
            paper_summary.get(filename, 0) + 1
        )

    print("\n" + "=" * 60)
    print("THE RELAY - TEXT CHUNKING")
    print("=" * 60)

    print(f"Chunk size: {chunk_size} words")
    print(f"Overlap: {overlap} words")
    print(f"Total papers: {len(papers)}")
    print(f"Total chunks: {len(chunks)}")
    print(f"Total words across chunks: {total_words}")

    print("\nChunks per paper:")

    for filename, count in paper_summary.items():
        print(f"  {filename}: {count}")

    if chunks:
        average_words = total_words / len(chunks)
        print(f"\nAverage chunk size: {average_words:.1f} words")

    print(f"\nSaved to: {output_file.relative_to(PROJECT_ROOT)}")
    print("=" * 60)

    return chunks


# --------------------------------------------------
# 8. COMMAND-LINE ENTRY POINT
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Chunk the ingested research papers."
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=DEFAULT_CHUNK_SIZE,
        help="Maximum words per chunk (default: 350)",
    )

    parser.add_argument(
        "--overlap",
        type=int,
        default=DEFAULT_OVERLAP,
        help="Overlapping words between chunks (default: 60)",
    )

    args = parser.parse_args()

    run_chunking(
        chunk_size=args.chunk_size,
        overlap=args.overlap,
    )
