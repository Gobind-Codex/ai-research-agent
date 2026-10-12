from src.research_agent import answer_question
from src.verification import validate_citations


def main():
    print("AI Research Agent — Local Mode")
    print("Type 'exit' to quit.\n")

    while True:
        question = input("Research question: ").strip()

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if not question:
            print("Please enter a question.\n")
            continue

        try:
            result = answer_question(question)

            answer = result["answer"]
            sources = result["sources"]

            validation = validate_citations(answer, sources)

            print("\n--- Research Answer ---\n")
            print(answer)

            if not validation["has_citations"]:
                print("\nWarning: The answer contains no source citations.")

            if not validation["valid"]:
                print(
                    "\nWarning: Invalid source IDs:",
                    validation["invalid_ids"],
                )

            print("\n--- Retrieved Sources ---")

            for source in sources:
                print(
                    f"[{source['id']}] "
                    f"{source['source']} | "
                    f"Page {source['page_number']} | "
                    f"Chunk {source['chunk_index']}"
                )

            print("\n" + "=" * 50 + "\n")

        except Exception as error:
            print(f"\nRequest failed: {error}\n")


if __name__ == "__main__":
    main()
