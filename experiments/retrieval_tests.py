
from app.retrieval.vector_store import VectorStore


def main():
    store = VectorStore()
    store.load()

    questions = [
        "Why does the Transformer use self-attention instead of recurrence?",
        "What are the advantages of multi-head attention?",
        "How does the Transformer handle positional information?",
        "What is Moshi's approach to real-time spoken dialogue?",
        "How does Moshi process incoming audio?",
        "What are the main components of Moshi's architecture?",
        "What is dynamic chunking?",
        "How does H-Net learn where to divide sequences?",
        "How does dynamic chunking compare with fixed chunking?",
    ]

    for question in questions:
        print("\n" + "=" * 80)
        print("QUESTION:", question)
        print("=" * 80)

        results = store.search(question, top_k=5)

        for rank, result in enumerate(results, start=1):
            print(
                f"\n{rank}. {result['citation']} "
                f"| Similarity: {result['score']:.4f}"
            )
            print(result["text"][:500])


if __name__ == "__main__":
    main()
