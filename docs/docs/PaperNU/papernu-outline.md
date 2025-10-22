---
sidebar_position: 1
---

# PaperNU Documentation!

This contains cursory documentation for how the RAG embedding and retrival of PaperNU works.

## Loading

- Loading the data from `server/data/raw/papernu.json` is handled by `papernu_loader.py`. 
- It is called internally by `build_index.py` when you build the vector index

## Embedding & Retrieval
- Both GroupMe and PaperNU retrieval and embeddings are handled in the large `pipeline.py` file with the `RAGPipeline` class. 
- The `RAGPipeline` class contains both the logic for building context on papernu data and groupme data

## Special Notes
- Keigo Healy who worked on the PaperNU addition is familiar with ChromaDB and NOT familiar with Pinecone and hasn't tested integration for Pinecone. 
  - [Reach out to Keigo](mailto:keigo@u.northwestern.edu) if there are questions 