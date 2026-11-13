
import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHUNKS_FILE = PROJECT_ROOT / "data" / "processed" / "chunks_350_60.json"
INDEX_DIR = PROJECT_ROOT / "data" / "processed"

INDEX_FILE = INDEX_DIR / "faiss_index.bin"
METADATA_FILE = INDEX_DIR / "vector_metadata.json"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


class VectorStore:
    def __init__(self, model_name=MODEL_NAME):
        print(f"Loading embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build(self, chunks):
        """Generate embeddings and build the FAISS similarity index."""

        if not chunks:
            raise ValueError("No chunks provided.")

        self.chunks = chunks

        texts = [chunk["text"] for chunk in chunks]

        print(f"Generating embeddings for {len(texts)} chunks...")

        embeddings = self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        embeddings = np.asarray(embeddings, dtype=np.float32)

        dimension = embeddings.shape[1]

        # Inner product on normalized vectors = cosine similarity.
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)

        print("\nFAISS index created successfully.")
        print(f"Number of vectors: {self.index.ntotal}")
        print(f"Vector dimensions: {dimension}")

    def save(self):
        """Save the FAISS index and corresponding chunk metadata."""

        if self.index is None:
            raise ValueError("Build the index before saving.")

        INDEX_DIR.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(INDEX_FILE))

        with open(METADATA_FILE, "w", encoding="utf-8") as file:
            json.dump(
                {
                    "model_name": MODEL_NAME,
                    "total_vectors": len(self.chunks),
                    "chunks": self.chunks,
                },
                file,
                ensure_ascii=False,
                indent=2,
            )

        print(f"\nIndex saved: {INDEX_FILE}")
        print(f"Metadata saved: {METADATA_FILE}")

    def load(self):
        """Load an existing FAISS index and its chunk metadata."""

        if not INDEX_FILE.exists() or not METADATA_FILE.exists():
            raise FileNotFoundError(
                "Vector index not found. Build the index first."
            )

        self.index = faiss.read_index(str(INDEX_FILE))

        with open(METADATA_FILE, encoding="utf-8") as file:
            metadata = json.load(file)

        if metadata["model_name"] != MODEL_NAME:
            raise ValueError(
                "Embedding model mismatch between saved index and code."
            )

        self.chunks = metadata["chunks"]

        if self.index.ntotal != len(self.chunks):
            raise ValueError("Index and metadata sizes do not match.")

        print(f"Loaded {self.index.ntotal} vectors.")

    def search(self, query, top_k=5):
        """Retrieve the most semantically similar chunks."""

        if self.index is None:
            raise ValueError("Load or build the vector index first.")

        if not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k < 1:
            raise ValueError("top_k must be at least 1.")

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        query_embedding = np.asarray(
            query_embedding,
            dtype=np.float32,
        )

        k = min(top_k, self.index.ntotal)

        scores, indices = self.index.search(query_embedding, k)

        results = []

        for score, index_position in zip(scores[0], indices[0]):
            if index_position < 0:
                continue

            chunk = self.chunks[int(index_position)]

            results.append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "title": chunk["title"],
                    "page_number": chunk["page_number"],
                    "citation": chunk["citation"],
                    "score": float(score),
                    "text": chunk["text"],
                }
            )

        return results


def main():
    if not CHUNKS_FILE.exists():
        raise FileNotFoundError(
            f"Chunk file not found: {CHUNKS_FILE}"
        )

    with open(CHUNKS_FILE, encoding="utf-8") as file:
        data = json.load(file)

    chunks = data["chunks"]

    store = VectorStore()
    store.build(chunks)
    store.save()

    print("\nTesting semantic search...")

    test_queries = [
        "Why does the Transformer use self-attention?",
        "How does Moshi support real-time spoken dialogue?",
        "What is dynamic chunking and why is it useful?",
    ]

    for query in test_queries:
        print(f"\nQuestion: {query}")

        results = store.search(query, top_k=3)

        for rank, result in enumerate(results, start=1):
            print(
                f"\n{rank}. {result['citation']} "
                f"(similarity: {result['score']:.4f})"
            )
            print(result["text"][:300])


if __name__ == "__main__":
    main()
