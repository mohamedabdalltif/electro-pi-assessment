from rag_pipeline import build_vectorstore, build_rag_chain

QUESTIONS = [
    # In-domain — answer exists in docs
    "What is the refund policy for missing items?",
    "How much is the base delivery fee and how does surge pricing work?",
    # Out-of-domain — triggers guardrail
    "What is the capital of France?",
]

if __name__ == "__main__":
    print("Building vectorstore...")
    vs = build_vectorstore()
    chain = build_rag_chain(vs)

    print("\n" + "=" * 60)
    print("RAG PIPELINE EVALUATION — 3 QUESTIONS")
    print("=" * 60)

    for i, q in enumerate(QUESTIONS, 1):
        print(f"\nQ{i}: {q}")
        print("-" * 40)
        answer = chain.invoke(q)
        print(f"A{i}: {answer}")
        print()
