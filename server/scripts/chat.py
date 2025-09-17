from rag.pipeline import RAGPipelineGM

def main():
    rag = RAGPipelineGM()
    print("GroupMe RAG. Type 'exit' to quit.")
    print("tenant/db:", rag.client.tenant, rag.client.database)
    print("collection name:", rag.collection.name)
    print("collection id:", rag.collection.id)
    print("count:", rag.collection.count())
    print("peek:", rag.collection.peek())
    while True:
        q = input("\nYou: ").strip()
        if q.lower() in {"exit","quit"}: break
        if not q: continue
        ans = rag.generate(q, k=6)
        print("\nBot:", ans)

if __name__ == "__main__":
    main()
