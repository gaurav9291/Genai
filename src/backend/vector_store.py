# vector_store.py
"""Vector store implementation supporting Pinecone (primary) and FAISS (fallback) via LangChain"""

import os
import json
import pickle
import logging
import hashlib
from typing import List, Dict, Any, Optional, Tuple

# FAISS stack
import faiss
import numpy as np
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore import InMemoryDocstore
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

# App settings
from .config import settings

# Pinecone (optional)
try:
    from langchain_pinecone import PineconeVectorStore
    from pinecone import Pinecone, ServerlessSpec
except ImportError:
    PineconeVectorStore = None
    Pinecone = None
    ServerlessSpec = None

import asyncio

logger = logging.getLogger(__name__)

_embeddings_cache = {}
_vector_store_instance = None


class DeterministicFallbackEmbeddings(Embeddings):
    """Small local embedding fallback used when HF models are unavailable.

    This is not a semantic embedding model. It exists so local demos and tests
    can keep ingestion/indexing paths alive without network access.
    """

    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.is_fallback = True

    def _embed(self, text: str) -> List[float]:
        vector = np.zeros(self.dimension, dtype=np.float32)
        tokens = [token for token in (text or "").lower().split() if token]
        if not tokens:
            return vector.tolist()
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self.dimension
            sign = 1.0 if digest[4] % 2 == 0 else -1.0
            vector[index] += sign
        norm = np.linalg.norm(vector)
        if norm:
            vector = vector / norm
        return vector.tolist()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)


def get_embeddings(model_name: str = None) -> HuggingFaceEmbeddings:
    if model_name is None:
        model_name = settings.embedding_model
    
    # Normalize model names to prevent reloading the same model with different aliases
    normalized_name = model_name
    if normalized_name == "all-MiniLM-L6-v2":
        normalized_name = "sentence-transformers/all-MiniLM-L6-v2"
        
    if normalized_name not in _embeddings_cache:
        logger.info(f"Loading HuggingFaceEmbeddings for model: {normalized_name}...")
        try:
            _embeddings_cache[normalized_name] = HuggingFaceEmbeddings(
                model_name=normalized_name,
                model_kwargs={"device": "cpu"},
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as exc:
            logger.warning(
                "Hugging Face embedding model unavailable (%s). "
                "Using deterministic local fallback embeddings.",
                exc,
            )
            _embeddings_cache[normalized_name] = DeterministicFallbackEmbeddings(
                dimension=settings.vector_dimension
            )
    return _embeddings_cache[normalized_name]

def get_vector_store() -> 'VectorStoreWrapper':
    global _vector_store_instance
    if _vector_store_instance is None:
        logger.info("Initializing VectorStoreWrapper singleton...")
        _vector_store_instance = VectorStoreWrapper()
    return _vector_store_instance

class VectorStoreWrapper:
    def __init__(self):
        self.embedding_model = get_embeddings(settings.embedding_model)
        self.vs = None
        self.type = None
        self.namespace = settings.pinecone_namespace
        self._initialize()

    def _initialize(self):
        try:
            if settings.pinecone_api_key and Pinecone:
                pc = Pinecone(api_key=settings.pinecone_api_key)
                index_name = settings.pinecone_index_name
                if index_name not in pc.list_indexes().names():
                    pc.create_index(
                        name=index_name,
                        dimension=settings.vector_dimension,
                        metric=settings.pinecone_metric,
                        spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region)
                    )
                self.vs = PineconeVectorStore.from_existing_index(
                    index_name=index_name,
                    embedding=self.embedding_model,
                    namespace=self.namespace
                )
                self.type = "pinecone"
            else:
                raise ImportError("Pinecone not available")
        except Exception as e:
            logger.warning(f"Pinecone init failed: {e}, falling back to FAISS")
            dimension = len(self.embedding_model.embed_query("test"))
            index = faiss.IndexFlatL2(dimension)
            self.vs = FAISS(
                embedding_function=self.embedding_model,
                index=index,
                docstore=InMemoryDocstore({}),
                index_to_docstore_id={}
            )
            self.type = "faiss"

    async def async_add_documents(self, documents: List[Document]) -> List[str]:
        loop = asyncio.get_running_loop()
        if self.type == "pinecone":
            return await loop.run_in_executor(None, lambda: self.vs.add_documents(documents))
        else:
            return await loop.run_in_executor(None, lambda: self.vs.add_documents(documents))

    async def async_update_metadata(self, id: str, metadata: Dict[str, Any]):
        loop = asyncio.get_running_loop()
        if self.type == "pinecone":
            await loop.run_in_executor(None, lambda: self.vs._index.update(id=id, set_metadata=metadata, namespace=self.namespace))
            return True
        else:
            for doc_id in list(self.vs.docstore._dict.keys()):
                if doc_id == id:
                    self.vs.docstore._dict[doc_id].metadata.update(metadata)
                    return True
            return False
