import sys 
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import ingestion.models
from ingestion.papernu_loader import make_context_from_papernu_data
from ingestion.models import FullCourseRecord
from rag.pipeline_papernu import RAGPipelinePaperNU
from pprint import pprint

def main():
    contexts: list[FullCourseRecord] = make_context_from_papernu_data()

    print(f"==================Preview of context================")
    for context in contexts[:2]:
        print(context)
    print(f"====================================================\n")
    
    print("Building paper index...")
    pipeline = RAGPipelinePaperNU()
    # pipeline.add_all_records(contexts)
    print("Done building paper index.")

    query = "What are the prerequisites for CS 336: Design & Analysis of Algorithms?"
    context = pipeline.build_context(query)
    print(f"Context for query '{query}':\n")
    print(context)




if __name__ == '__main__':
    main()