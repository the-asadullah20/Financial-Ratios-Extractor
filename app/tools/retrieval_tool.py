"""
Semantic Retrieval Tool - Vector search over Qdrant indexed document chunks with document_id filtering.
"""
from typing import List, Dict, Any, Optional
from app.storage.vector_store import vector_store


def semantic_retrieval_search(query: str, document_id: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Performs semantic vector search across the Qdrant vector store.
    Strictly filters results by document_id to guarantee zero chunk mixups between PDFs.
    """
    results = vector_store.search(query=query, document_id=document_id, top_k=top_k)
    return results
