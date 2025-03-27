from lrage.api.retriever import Retriever
from smolagents import Tool

class RetrieverTool(Tool):
    name = "retriever"
    description = "Uses semantic search to retrieve the parts of documentation that could be most relevant to answer your query."
    inputs = {
        "query": {
            "type": "string",
            "description": "The query to perform. This should be semantically close to your target documents. Use the affirmative form rather than a question.",
        },
        "top_k": {
            "type": "integer",
            "description": "Number of top documents to retrieve",
            "nullable": False,
        }
    }
    output_type = "string"

    def __init__(self, retriever: Retriever, **kwargs):
        super().__init__(**kwargs)
        self.retriever = retriever

    def forward(self, query: str, top_k: int = 3) -> str:
        assert isinstance(query, str), "Your search query must be a string"
        
        query_context = self.retriever.retrieve(query, top_k=top_k)
        
        result = "\nRetrieved documents:\n"
        for i, doc in enumerate(query_context.docs, 1):
            result += f"\n\n===== Document {i} (ID: {doc.id}) =====\n{doc.contents}"
            
        return result