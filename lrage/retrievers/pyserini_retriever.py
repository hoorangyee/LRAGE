from typing import List, Optional
import json

from pyserini.search.lucene import LuceneSearcher, LuceneImpactSearcher
from pyserini.search.faiss import FaissSearcher, AutoQueryEncoder
from pyserini.search.hybrid import HybridSearcher
from pyserini.prebuilt_index_info import TF_INDEX_INFO, IMPACT_INDEX_INFO, FAISS_INDEX_INFO

from lrage.api.registry import register_retriever
from lrage.api.retriever import Retriever, QueryContext, RetrievedDocument


@register_retriever("pyserini", "pyserini_retriever")
class PyseriniRetriever(Retriever):
    def __init__(
        self,
        retriever_type: str,
        bm25_index_path: str,
        sparse_index_path: Optional[str] = None,
        faiss_index_path: Optional[str] = None,
        encoder_path: Optional[str] = None,
    ) -> None:
        self.retriever_type = retriever_type

        if bm25_index_path is None:
            raise ValueError("All retrievers require a bm25_index_path")
        self.bm25_searcher = LuceneSearcher.from_prebuilt_index(bm25_index_path) if bm25_index_path in TF_INDEX_INFO else LuceneSearcher(bm25_index_path)

        self.searcher = self._initialize_searcher(
            retriever_type, sparse_index_path, faiss_index_path, encoder_path
        )

    def _initialize_searcher(
        self,
        retriever_type: str,
        sparse_index_path: Optional[str],
        faiss_index_path: Optional[str],
        encoder_path: Optional[str],
    ):

        if retriever_type == 'bm25':
            return self.bm25_searcher
        elif retriever_type == 'sparse':
            if sparse_index_path is None or encoder_path is None:
                raise ValueError("SparseRetriever requires an sparse_index_path and encoder_path")
            return LuceneImpactSearcher.from_prebuilt_index(sparse_index_path) if sparse_index_path in IMPACT_INDEX_INFO else LuceneImpactSearcher(sparse_index_path)
        elif retriever_type == 'dense':
            if faiss_index_path is None or encoder_path is None:
                raise ValueError("DenseRetriever requires faiss_index_path and encoder_path")
            return (FaissSearcher.from_prebuilt_index(faiss_index_path, AutoQueryEncoder(encoder_path)) if faiss_index_path in FAISS_INDEX_INFO.keys() 
                        else FaissSearcher(faiss_index_path, AutoQueryEncoder(encoder_path)))
        elif retriever_type == 'hybrid':
            if faiss_index_path is None or encoder_path is None:
                raise ValueError("HybridRetriever requires faiss_index_path and encoder_path")
            dense_searcher = (FaissSearcher.from_prebuilt_index(faiss_index_path, AutoQueryEncoder(encoder_path)) if faiss_index_path in FAISS_INDEX_INFO.keys() 
                                else FaissSearcher(faiss_index_path, AutoQueryEncoder(encoder_path)))
            return HybridSearcher(dense_searcher, self.bm25_searcher)
        else:
            raise ValueError(f"Unknown retriever type: {retriever_type}")

    def _get_docs(self, doc_ids: List[str]) -> List[dict]:
        docs = [self.bm25_searcher.doc(doc_id) for doc_id in doc_ids]
        try:
            docs = [json.loads(doc.raw()) for doc in docs]
        except json.JSONDecodeError:
            docs = [{'id': doc_id, 'contents': doc.raw()} for doc_id, doc in zip(doc_ids, docs)]
        return docs

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