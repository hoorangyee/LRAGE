import abc
from typing import List, Optional, Union, Type, TypeVar

from lrage import utils
from lrage.api.retriever import QueryContext

T = TypeVar("T", bound="Reranker")

class Reranker(abc.ABC):
    """Abstract base class for rerankers"""
    
    @abc.abstractmethod
    def rank(
        self,
        query: str,
        docs: List[str],
        doc_ids: Union[List[int], List[str]]
    ) -> QueryContext:
        """
        Rank documents for a query.
        
        Args:
            query: The query string
            docs: List of document contents
            doc_ids: List of document IDs
            
        Returns:
            QueryContext containing ranked documents
        """
        pass

    @classmethod
    def create_from_arg_string(
        cls: Type[T], 
        arg_string: str, 
        additional_config: Optional[dict] = None
    ) -> T:
        """
        Creates an instance of the Reranker class using the given argument string and additional config.

        Parameters:
        - arg_string: A string containing arguments in the format key1=value1,key2=value2.
        - additional_config: Optional dictionary containing additional configuration parameters.

        Returns:
        - Instance of the Reranker class.
        """
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)