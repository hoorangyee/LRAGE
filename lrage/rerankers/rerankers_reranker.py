from typing import List, Optional, Union, Any

import rerankers

from lrage.api.registry import register_reranker
from lrage.api.retriever import QueryContext, RetrievedDocument
from lrage.api.reranker import Reranker


@register_reranker("rerankers", "rerankers_reranker")
class RerankersReranker(Reranker):
    """Reranker implementation using the rerankers library"""
    
    def __init__(
        self,
        reranker_type: str,
        reranker_path: Optional[str] = None,
    ) -> None:
        """
        Initialize the RerankersReranker.
        
        Args:
            reranker_type: Type of reranker to use
            reranker_path: Optional path to reranker model
        """
        self.reranker_type = reranker_type
        if reranker_path is not None:
            self.reranker = rerankers.Reranker(reranker_type, reranker_path)
        else:
            self.reranker = rerankers.Reranker(reranker_type)

    def _postprocess_results(self, results: Any) -> QueryContext:
        """
        Convert rerankers library results to QueryContext.
        
        Args:
            results: Results from rerankers library
            
        Returns:
            QueryContext containing processed results
        """
        return QueryContext(
            query=results.query,
            docs=[
                RetrievedDocument(
                    id=result.document.doc_id,
                    contents=result.document.text
                )
                for result in results.results
            ],
            doc_ids=[result.document.doc_id for result in results.results]
        )

    def rank(
        self,
        query: str,
        docs: List[str],
        doc_ids: Union[List[int], List[str]]
    ) -> QueryContext:
        """
        Rank documents for a query.
        
        Args:
            query: Query string
            docs: List of document contents
            doc_ids: List of document IDs
            
        Returns:
            QueryContext containing ranked documents
        """
        # Convert all doc_ids to strings for consistency
        doc_ids = [str(doc_id) for doc_id in doc_ids]
        
        # Run reranking
        results = self.reranker.rank(query, docs, doc_ids)
        
        # Process results
        return self._postprocess_results(results)