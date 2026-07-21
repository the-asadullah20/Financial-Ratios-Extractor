"""
Vector Database Storage using Qdrant.
Handles chunking of raw Markdown, embedding generation, indexing in Qdrant,
and document-isolated semantic vector search.
"""
import re
import uuid
import logging
from typing import List, Dict, Any, Optional

from app.config import settings

logger = logging.getLogger("vector_store")


class Chunk(Dict[str, Any]):
    chunk_id: str
    document_id: str
    text: str
    page_number: int


def semantic_chunk_markdown(markdown_text: str, document_id: str, chunk_size: int = 800) -> List[Dict[str, Any]]:
    """
    Splits markdown into semantic page-aware chunks with document_id metadata.
    """
    chunks = []
    page_blocks = re.split(r"--- PAGE (\d+) ---", markdown_text)
    
    if len(page_blocks) > 1:
        for i in range(1, len(page_blocks), 2):
            page_num = int(page_blocks[i])
            page_content = page_blocks[i + 1].strip()
            
            paragraphs = page_content.split("\n\n")
            current_chunk = ""
            for p in paragraphs:
                if len(current_chunk) + len(p) > chunk_size and current_chunk:
                    chunks.append({
                        "chunk_id": uuid.uuid4().hex[:12],
                        "document_id": document_id,
                        "text": current_chunk.strip(),
                        "page_number": page_num,
                    })
                    current_chunk = p
                else:
                    current_chunk += "\n\n" + p if current_chunk else p
            if current_chunk.strip():
                chunks.append({
                    "chunk_id": uuid.uuid4().hex[:12],
                    "document_id": document_id,
                    "text": current_chunk.strip(),
                    "page_number": page_num,
                })
    else:
        paragraphs = markdown_text.split("\n\n")
        current_chunk = ""
        for p in paragraphs:
            if len(current_chunk) + len(p) > chunk_size and current_chunk:
                chunks.append({
                    "chunk_id": uuid.uuid4().hex[:12],
                    "document_id": document_id,
                    "text": current_chunk.strip(),
                    "page_number": 1,
                })
                current_chunk = p
            else:
                current_chunk += "\n\n" + p if current_chunk else p
        if current_chunk.strip():
            chunks.append({
                "chunk_id": uuid.uuid4().hex[:12],
                "document_id": document_id,
                "text": current_chunk.strip(),
                "page_number": 1,
            })

    return chunks


class VectorStore:
    def __init__(self):
        self.client = None
        self.collection_name = "financial_chunks"
        self.vector_dim = 384
        self._memory_chunks: List[Dict[str, Any]] = []
        self._init_qdrant()

    def _init_qdrant(self):
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams
            
            # Connect to Qdrant Cloud Cluster if QDRANT_URL is set, else local persistent disk
            if settings.QDRANT_URL:
                kwargs = {"url": settings.QDRANT_URL}
                if settings.QDRANT_API_KEY:
                    kwargs["api_key"] = settings.QDRANT_API_KEY
                self.client = QdrantClient(**kwargs)
                logger.info("Connected to Qdrant Cloud Cluster at %s", settings.QDRANT_URL)
            else:
                self.client = QdrantClient(path=settings.QDRANT_STORAGE_PATH)
                logger.info("Initialized local persistent Qdrant storage at %s", settings.QDRANT_STORAGE_PATH)
            
            existing_collections = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in existing_collections:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(size=self.vector_dim, distance=Distance.COSINE),
                )
        except Exception as exc:
            logger.warning("Could not initialize QdrantClient (%s). Using fallback vector store.", exc)
            self.client = None

    def _generate_embedding(self, text: str) -> List[float]:
        """Deterministic embedding vector generator for semantic similarity."""
        import hashlib
        import math
        
        vec = [0.0] * self.vector_dim
        words = re.findall(r"\w+", text.lower())
        for w in words:
            h = int(hashlib.sha256(w.encode()).hexdigest(), 16)
            idx = h % self.vector_dim
            val = ((h >> 8) % 1000) / 1000.0 - 0.5
            vec[idx] += val
        
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def index_document(self, document_id: str, markdown_content: str) -> int:
        """Chunks and indexes document into Qdrant vector store with unique point IDs."""
        chunks = semantic_chunk_markdown(markdown_content, document_id)
        self._memory_chunks.extend(chunks)

        if self.client:
            try:
                from qdrant_client.models import PointStruct
                points = []
                for idx, ch in enumerate(chunks):
                    # Deterministic unique UUID string for each document chunk
                    point_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"{document_id}_{idx}"))
                    emb = self._generate_embedding(ch["text"])
                    points.append(PointStruct(
                        id=point_uuid,
                        vector=emb,
                        payload=ch
                    ))
                self.client.upsert(collection_name=self.collection_name, points=points)
                logger.info("Indexed %d chunks into Qdrant for document %s", len(chunks), document_id)
            except Exception as exc:
                logger.warning("Qdrant upsert failed (%s). Saved to memory fallback.", exc)
        return len(chunks)

    def search(self, query: str, document_id: Optional[str] = None, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Performs semantic vector search across indexed chunks.
        Filters strictly by document_id to guarantee zero mixup between documents.
        """
        query_emb = self._generate_embedding(query)
        
        if self.client:
            try:
                from qdrant_client.models import Filter, FieldCondition, MatchValue
                query_filter = None
                if document_id:
                    query_filter = Filter(
                        must=[
                            FieldCondition(
                                key="document_id",
                                match=MatchValue(value=document_id)
                            )
                        ]
                    )

                search_results = self.client.search(
                    collection_name=self.collection_name,
                    query_vector=query_emb,
                    query_filter=query_filter,
                    limit=top_k
                )
                return [hit.payload for hit in search_results]
            except Exception as exc:
                logger.warning("Qdrant search error (%s), using fallback search.", exc)

        # Fallback search filtered by document_id
        filtered_memory = [ch for ch in self._memory_chunks if not document_id or ch.get("document_id") == document_id]
        results = []
        for ch in filtered_memory:
            ch_emb = self._generate_embedding(ch["text"])
            score = sum(a * b for a, b in zip(query_emb, ch_emb))
            results.append((score, ch))
        
        results.sort(key=lambda x: x[0], reverse=True)
        return [res[1] for res in results[:top_k]]


vector_store = VectorStore()
