from typing import List, Optional
import json

from pyserini.search.lucene import LuceneSearcher, LuceneImpactSearcher
from pyserini.search.faiss import FaissSearcher, AutoQueryEncoder
from pyserini.search.hybrid import HybridSearcher

from lrage.api.registry import register_retriever
from lrage.api.retriever import Retriever, QueryContext, RetrievedDocument


@register_retriever("pyserini", "pyserini_retriever")
class PyseriniRetriever(Retriever):
    def __init__(
        self,
        retriever_type: str,
        bm25_index_path: str,
        faiss_index_path: Optional[str] = None,
        encoder_path: Optional[str] = None,
    ) -> None:
        self.retriever_type = retriever_type
        self.doc_searcher = LuceneSearcher(bm25_index_path)
        self.searcher = self._initialize_searcher(
            retriever_type, bm25_index_path, faiss_index_path, encoder_path
        )

    def _initialize_searcher(
        self,
        retriever_type: str,
        bm25_index_path: str,
        faiss_index_path: Optional[str],
        encoder_path: Optional[str],
    ):
        if retriever_type == 'bm25':
            return self.doc_searcher
        elif retriever_type == 'sparse':
            if encoder_path is None:
                raise ValueError("SparseRetriever requires an encoder_path")
            return LuceneImpactSearcher(bm25_index_path, encoder_path)
        elif retriever_type == 'dense':
            if faiss_index_path is None or encoder_path is None:
                raise ValueError("DenseRetriever requires faiss_index_path and encoder_path")
            return FaissSearcher(faiss_index_path, AutoQueryEncoder(encoder_path))
        elif retriever_type == 'hybrid':
            if faiss_index_path is None or encoder_path is None:
                raise ValueError("HybridRetriever requires faiss_index_path and encoder_path")
            dense_searcher = FaissSearcher(faiss_index_path, AutoQueryEncoder(encoder_path))
            sparse_searcher = self.doc_searcher
            return HybridSearcher(dense_searcher, sparse_searcher)
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")

    def _get_docs(self, doc_ids: List[str]) -> List[dict]:
        docs = [self.doc_searcher.doc(doc_id) for doc_id in doc_ids]
        return [
            json.loads(doc.raw()) 
            for doc in docs 
            if doc is not None
        ]

    def retrieve(self, query: str, top_k: int = 3) -> QueryContext:
        """
        Retrieve documents for a single query
        
        Args:
            query: Search query
            top_k: Number of top documents to retrieve
            
        Returns:
            QueryContext containing the query and retrieved documents
        """
        hits = self.searcher.search(query, k=top_k)
        docs = self._get_docs([hit.docid for hit in hits])
        
        return QueryContext(
            query=query,
            docs=[RetrievedDocument(id=doc['id'], contents=doc['contents']) 
                  for doc in docs],
            doc_ids=[doc['id'] for doc in docs]
        )

    def batch_retrieve(
        self,
        queries: List[str],
        q_ids: List[str],
        top_k: int = 3,
    ) -> List[QueryContext]:
        """
        Retrieve documents for multiple queries
        
        Args:
            queries: List of search queries
            q_ids: List of query identifiers
            top_k: Number of top documents to retrieve per query
            
        Returns:
            List of QueryContext, one for each query
        """
        hits = self.searcher.batch_search(queries, q_ids, k=top_k, threads=100)
        results = []
        
        for query_id, query in zip(q_ids, queries):
            if query_id in hits:
                docs = self._get_docs([hit.docid for hit in hits[query_id]])
                results.append(QueryContext(
                    query=query,
                    docs=[RetrievedDocument(id=doc['id'], contents=doc['contents']) 
                          for doc in docs],
                    doc_ids=[doc['id'] for doc in docs]
                ))
            else:
                results.append(QueryContext(
                    query=query,
                    docs=[],
                    doc_ids=[]
                ))
                
        return results