from dataclasses import dataclass
import abc
from typing import List, Optional, ClassVar, Type, TypeVar

from lrage import utils

T = TypeVar("T", bound="Retriever")

@dataclass
class RetrievedDocument:
    """Represents a single retrieved document"""
    id: str
    contents: str
    
    # Class level constants for context formatting
    CONTEXT_START: ClassVar[str] = "DOCUMENTS START\n"
    CONTEXT_END: ClassVar[str] = "\nDOCUMENTS END\n"
    DOCUMENT_PREFIX: ClassVar[str] = "DOCUMENT"
    
    @classmethod
    def build_context(cls, docs: List['RetrievedDocument']) -> str:
        """
        Build a context string from a list of retrieved documents.
        
        Args:
            docs: List of RetrievedDocument instances
            
        Returns:
            Formatted context string with document markers
        """
        context_parts = [cls.CONTEXT_START]
        
        for doc_num, doc in enumerate(docs, 1):
            doc_text = f"{cls.DOCUMENT_PREFIX} {doc_num}. {doc.contents}\n"
            context_parts.append(doc_text)
            
        context_parts.append(cls.CONTEXT_END)
        return ''.join(context_parts)

@dataclass
class QueryContext:
    """Represents a query and its associated context"""
    query: str
    docs: List[RetrievedDocument]
    doc_ids: List[str]
    
    def build_context(self) -> str:
        """
        Build a context string from the documents in this query context.
        
        Returns:
            Formatted context string
        """
        return RetrievedDocument.build_context(self.docs)

class Retriever(abc.ABC):
    """Abstract base class for retrievers"""
    
    @abc.abstractmethod
    def retrieve(self, query: str, top_k: int = 3) -> QueryContext:
        """
        Retrieve documents for a single query.
        
        Args:
            query: Search query string
            top_k: Number of top documents to retrieve
            
        Returns:
            QueryContext containing the query and its retrieved documents
        """
        pass
    
    @abc.abstractmethod
    def batch_retrieve(
        self,
        queries: List[str],
        q_ids: List[str],
        top_k: int = 3
    ) -> List[QueryContext]:
        """
        Retrieve documents for multiple queries.
        
        Args:
            queries: List of search queries
            q_ids: List of query identifiers
            top_k: Number of top documents to retrieve per query
            
        Returns:
            List of QueryContext, one for each query
        """
        pass

    @classmethod
    def create_from_arg_string(
        cls: Type[T], 
        arg_string: str, 
        additional_config: Optional[dict] = None
    ) -> T:
        """
        Creates an instance of the Retriever class using the given argument string and additional config.

        Parameters:
        - arg_string: A string containing arguments in the format key1=value1,key2=value2.
        - additional_config: Optional dictionary containing additional configuration parameters.

        Returns:
        - Instance of the Retriever class.
        """
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)