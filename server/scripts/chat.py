from rag.pipeline import RAGPipelineGM

def main():
    rag = RAGPipelineGM()
    print("GroupMe RAG. Type 'exit' to quit.")
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit","quit"}: break
        if not q: continue
        ans = rag.generate(q, k=6)
        print("\nBot:", ans)

if __name__ == "__main__":
    main()
