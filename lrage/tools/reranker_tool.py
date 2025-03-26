from typing import List, Union
from smolagents import Tool
from lrage.api.reranker import Reranker

class RerankerTool(Tool):
    name = "reranker"
    description = "Reranks a list of documents based on relevance to your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to rerank documents for.",
        },
        "docs": {
            "type": "array",
            "description": "List of document contents to rerank.",
        },
        "doc_ids": {
            "type": "array", 
            "description": "List of document IDs corresponding to the documents."
        }
    }
    output_type = "string"

    def __init__(self, reranker: Reranker, **kwargs):
        super().__init__(**kwargs)
        self.reranker = reranker

    def forward(self, query: str, docs: List[str], doc_ids: Union[List[int], List[str]]) -> str:
        assert isinstance(query, str), "Your query must be a string"
        assert isinstance(docs, list), "Documents must be provided as a list"
        assert isinstance(doc_ids, list), "Document IDs must be provided as a list"
        assert len(docs) == len(doc_ids), "Number of documents and document IDs must match"
        
        query_context = self.reranker.rank(query, docs, doc_ids)
        
        result = "\nReranked documents:\n"
        for i, doc in enumerate(query_context.docs, 1):
            result += f"\n\n===== Document {i} (ID: {doc.id}) =====\n{doc.contents}"
            
        return result