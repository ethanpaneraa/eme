import argparse, glob, pathlib, sys
import pathlib as _p; sys.path.append(str(_p.Path(__file__).resolve().parents[1]))

from ingestion.loader import load_jsonl, make_chunks_from_records
from rag.pipeline import RAGPipelineGM

def batched(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i:i+n]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_glob", default="data/raw/*.jsonl")
    ap.add_argument("--window", type=int, default=1)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    files = sorted(glob.glob(args.input_glob))
    if not files:
        print("No input files. Put .jsonl dumps under data/raw/")
        return

    rag = RAGPipelineGM()
    total = 0
    for fp in files:
        recs   = load_jsonl(pathlib.Path(fp))
        chunks = make_chunks_from_records(recs, window=args.window)
        print(f"{fp}: {len(chunks)} chunks")
        for part in batched(chunks, args.batch):
            rag.add_messages(part)
            total += len(part)
            print(f"  indexed +{len(part)} (total {total})")

    print("Done.")
    print("Count:", rag.collection.count())
    print("Peek:", rag.collection.peek())

if __name__ == "__main__":
    main()
