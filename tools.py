"""
tools.py
--------
Retrieval layer for the Modular Salesforce RAG system.
Import `retrieve_context` into orchestrator.py to get relevant
context from ChromaDB before calling the LLM.
"""

from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

# ── Configuration (must match ingest_data.py) ─────────────────────────────────
CHROMA_DIR  = "./chroma_db"
COLLECTION  = "salesforce_rag"
EMBED_MODEL = "all-MiniLM-L6-v2"
TOP_K       = 4  # number of chunks to retrieve

# ── Initialise once at import time ────────────────────────────────────────────
_embedder   = SentenceTransformer(EMBED_MODEL)
_client     = PersistentClient(path=CHROMA_DIR)
_collection = _client.get_or_create_collection(name=COLLECTION)


# ── Public API ─────────────────────────────────────────────────────────────────

def retrieve_context(user_query: str, module_name: str) -> str:
    """
    Retrieve relevant text chunks from ChromaDB for a given query and module.

    Args:
        user_query  : The raw question from the user.
        module_name : One of the 6 router outputs. If "general_queries",
                      returns "" immediately (no DB lookup needed).

    Returns:
        A single concatenated string of the top-K matching chunks,
        or "" if nothing is found or an error occurs.
    """

    # General queries do not need vector DB — return empty
    if module_name == "general_queries":
        return ""

    try:
        # Embed the query using the same model used during ingestion
        query_embedding = _embedder.encode(user_query).tolist()

        # Query ChromaDB, filtering to only the relevant module's chunks
        results = _collection.query(
            query_embeddings=[query_embedding],
            n_results=TOP_K,
            where={"module": module_name},  # Critical: isolates search by module
        )

        documents = results.get("documents", [[]])[0]

        if not documents:
            return ""

        # Concatenate the retrieved chunks into a single context block
        context = "\n\n---\n\n".join(documents)
        return context

    except Exception as e:
        return ""


