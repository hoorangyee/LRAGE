from lrage.api.model import TemplateLM, LM
from lrage.api.registry import register_model
from typing import List, Tuple, Optional, Dict, Any, Union
import logging

eval_logger = logging.getLogger(__name__)

@register_model("codeagent")
class CodeAgentLM(TemplateLM):
    """
    An LRAGE compatible wrapper for smolagents.CodeAgent.
    
    This wrapper allows CodeAgent to be used with LRAGE by implementing
    the LM interface. It primarily supports the generate_until method using the
    CodeAgent's run capabilities and attempts to delegate the loglikelihood methods
    to the underlying model when possible.
    """
    
    def __init__(
        self, 
        tools=None,
        model=None,
        max_steps=20,
        **agent_kwargs
    ):
        """
        Initialize the CodeAgentLM wrapper with a CodeAgent instance.
        
        Args:
            agent: An existing CodeAgent instance, or None to create a new one
            tools: Tools to provide to the CodeAgent if creating a new one
            model: Model to use with the CodeAgent if creating a new one
            max_steps: Maximum number of steps for the agent to take
            **agent_kwargs: Additional arguments to pass to CodeAgent constructor
        """
        super().__init__()
        from smolagents import CodeAgent
        
        if tools is None:
            tools = []
        
        if model is None:
            raise ValueError("model must be provided")
        
        self.agent = CodeAgent(
            tools=tools,
            model=model,
            max_steps=max_steps,
            **agent_kwargs
        )
        
        # The underlying model that CodeAgent uses
        self.model = self.agent.model
    
    @property
    def eot_token_id(self):
        """Return the end-of-text token ID."""
        if hasattr(self.model, "eot_token_id"):
            return self.model.eot_token_id
            
        if hasattr(self.model, "tokenizer") and hasattr(self.model.tokenizer, "eos_token_id"):
            return self.model.tokenizer.eos_token_id
            
        return None
    
    @property
    def tokenizer_name(self) -> str:
        """Return the tokenizer name for caching purposes."""
        if hasattr(self.model, "tokenizer_name"):
            return self.model.tokenizer_name
            
        if hasattr(self.model, "tokenizer") and hasattr(self.model.tokenizer, "name_or_path"):
            return self.model.tokenizer.name_or_path.replace("/", "__")
            
        return "codeagent_tokenizer"
    
    def tok_encode(self, string: str, **kwargs):
        """
        Tokenize a string using the model's tokenizer.
        """
        if hasattr(self.model, "tok_encode"):
            return self.model.tok_encode(string, **kwargs)
            
        if hasattr(self.model, "tokenizer"):
            return self.model.tokenizer.encode(string, **kwargs)
            
        raise NotImplementedError("Tokenization not supported by this model")
    
    def _loglikelihood_tokens(self, requests, **kwargs):
        raise NotImplementedError("CodeAgentLM._loglikelihood_tokens() is not supported")

    def loglikelihood(self, requests) -> List[Tuple[float, bool]]:
        raise NotImplementedError("CodeAgentLM.loglikelihood() is not supported")
    
    def loglikelihood_rolling(self, requests) -> List[float]:
        raise NotImplementedError("CodeAgentLM.loglikelihood_rolling() is not supported")
    
    def generate_until(self, requests) -> List[str]:
        """
        Generate text continuations using the CodeAgent.
        
        This method delegates to the underlying model if possible,
        otherwise uses the CodeAgent's run capability to generate responses.
        
        Args:
            requests: list[Instance] - Each instance contains (context, gen_kwargs)
            
        Returns:
            list[str] - Generated continuations
        """
        results = []
        for request in requests:
            context, gen_kwargs = request.args
            generation_kwargs = dict(gen_kwargs)  # Create a copy to modify
            
            until = generation_kwargs.pop("until", [])
            
            try:
                result = self.agent.run(context)
                
                if not isinstance(result, str):
                    result = str(result)
                
                for stop_seq in until:
                    if stop_seq in result:
                        result = result[:result.index(stop_seq)]
                
                results.append(result)
                
            except Exception as e:
                eval_logger.error(f"Error during generation: {e}")
                results.append("")
                
        return results
    
    @classmethod
    def create_from_arg_string(cls, arg_string, additional_config=None):
        """
        Create an instance from a string of arguments.
        
        Args:
            arg_string: str - Arguments in the format key1=value1,key2=value2
            additional_config: dict, optional - Additional configuration parameters
            
        Returns:
            CodeAgentLM instance
        """
        from lrage import utils
        
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        
        combined_args = {**args, **args2}
        
        if "model" in combined_args and isinstance(combined_args["model"], str):
            model_str = combined_args.pop("model")
            model_args = combined_args.pop("model_args", {})
            batch_size = combined_args.pop("batch_size", 1)
            device = combined_args.pop("device", "cpu")
            
            if isinstance(model_args, str):
                model_args = utils.simple_parse_args_string(model_args)

                if not("batch_size" in model_args.keys()):
                    eval_logger.warning(f"Batch size not specified in model_args. Using batch size: {batch_size}")
                    model_args["batch_size"] = batch_size
                
                if not("device" in model_args.keys()):
                    eval_logger.warning(f"Device not specified in model_args. Using device: {device}")
                    model_args["device"] = device
                    
            if model_str.startswith("hf:"):
                from smolagents.models import TransformersModel
                model_id = model_str[3:]  # Remove 'hf:' prefix
                combined_args["model"] = TransformersModel(model_id=model_id, **model_args)
            elif model_str.startswith("hfapi:"):
                try:
                    from smolagents.models import HfApiModel
                    model_id = model_str[6:]  # Remove 'hfapi:' prefix
                    combined_args["model"] = HfApiModel(model_id=model_id, **model_args)
                except ImportError:
                    raise ImportError("HfApiModel not found in smolagents.models. Make sure you have the right version installed.")
            # Note: VLLMModel is not supported in latest smolagents version but in latest dev version. See: https://github.com/huggingface/smolagents/pull/337
            elif model_str.startswith("vllm:"):
                try:
                    from smolagents.models import VLLMModel
                    model_id = model_str[5:]  # Remove 'vllm:' prefix
                    combined_args["model"] = VLLMModel(model_id=model_id, **model_args)
                except ImportError:
                    raise ImportError("VLLMModel not found in smolagents.models. Make sure you have the right version installed.")
            elif model_str.startswith("mlx:"):
                try:
                    from smolagents.models import MLXModel
                    model_id = model_str[4:]  # Remove 'mlx:' prefix
                    combined_args["model"] = MLXModel(model_id=model_id, **model_args)
                except ImportError:
                    raise ImportError("MLXModel not found in smolagents.models. Make sure you have the right version installed.")
            elif model_str.startswith("transformers:"):
                try:
                    from smolagents.models import TransformersModel
                    model_id = model_str[13:]  # Remove 'transformers:' prefix
                    combined_args["model"] = TransformersModel(model_id=model_id, **model_args)
                except ImportError:
                    raise ImportError("TransformersModel not found in smolagents.models. Make sure you have the right version installed.")
            elif model_str.startswith("litellm:"):
                try:
                    from smolagents.models import LiteLLMModel
                    model_id = model_str[8:]  # Remove 'litellm:' prefix
                    combined_args["model"] = LiteLLMModel(model_id=model_id, **model_args)
                except ImportError:
                    raise ImportError("LitelLMModel not found in smolagents.models. Make sure you have the right version installed.")
            elif model_str.startswith("openai:"):
                try:
                    from smolagents.models import OpenAIServerModel
                    model_id = model_str[7:]  # Remove 'openai:' prefix
                    combined_args["model"] = OpenAIServerModel(model_id=model_id, **model_args)
                except ImportError:
                    raise ImportError("OpenAIModel not found in smolagents.models. Make sure you have the right version installed.")
            else:
                raise ValueError(f"Unsupported model type: {model_str}")
        
        return cls(**combined_args)