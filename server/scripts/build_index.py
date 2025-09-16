import argparse, pathlib, glob
from ingestion.loader import load_jsonl, make_chunks_from_records
from rag.pipeline import RAGPipelineGM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_glob", default="data/raw/*.jsonl")
    ap.add_argument("--window", type=int, default=1)
    args = ap.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        print("No input files. Put .jsonl dumps under data/raw/")
        return

    all_chunks = []
    for fp in files:
        recs = load_jsonl(pathlib.Path(fp))
        chunks = make_chunks_from_records(recs, window=args.window)
        all_chunks.extend(chunks)
        print(f"{fp}: {len(chunks)} chunks")

    rag = RAGPipelineGM()
    rag.add_messages(all_chunks)

if __name__ == "__main__":
    main()
