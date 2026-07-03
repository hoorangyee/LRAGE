from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator, model_validator


class RunStatus(str, Enum):
    queued = "queued"
    running = "running"
    cancelling = "cancelling"
    completed = "completed"
    failed = "failed"
    cancelled = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in (RunStatus.completed, RunStatus.failed, RunStatus.cancelled)


class GenKwargs(BaseModel):
    """Serialized to a `k=v,...` string for simple_evaluate, so values must
    survive simple_parse_args_string (no commas or equals signs)."""

    max_gen_toks: Optional[int] = Field(default=None, ge=1)
    temperature: Optional[float] = Field(default=None, ge=0)
    top_p: Optional[float] = Field(default=None, ge=0, le=1)
    do_sample: Optional[bool] = None

    def to_args_string(self) -> Optional[str]:
        parts = [
            f"{key}={value}"
            for key, value in self.model_dump(exclude_none=True).items()
        ]
        return ",".join(parts) if parts else None


class RetrievalConfig(BaseModel):
    retrieve_docs: bool = False
    top_k: int = Field(default=3, ge=1)
    retriever: Optional[str] = None
    retriever_args: Optional[str] = None
    rerank: bool = False
    reranker: Optional[str] = None
    reranker_args: Optional[str] = None

    @model_validator(mode="after")
    def check_consistency(self) -> "RetrievalConfig":
        if self.retrieve_docs:
            if not self.retriever:
                raise ValueError("retriever is required when retrieve_docs is on")
            if not self.retriever_args:
                raise ValueError("retriever_args is required when retrieve_docs is on")
        if self.rerank:
            if not self.retrieve_docs:
                raise ValueError("rerank requires retrieve_docs")
            if not self.reranker:
                raise ValueError("reranker is required when rerank is on")
            if not self.reranker_args:
                raise ValueError("reranker_args is required when rerank is on")
        return self


class RunConfig(BaseModel):
    """Mirrors the lrage.evaluator.simple_evaluate parameter surface."""

    model: str
    model_args: str
    tasks: List[str] = Field(min_length=1)

    judge_model: Optional[str] = None
    judge_model_args: Optional[str] = None
    judge_device: Optional[str] = None
    judge_gen_kwargs: Optional[GenKwargs] = None

    num_fewshot: Optional[int] = Field(default=None, ge=0)
    fewshot_as_multiturn: bool = False
    batch_size: Union[Literal["auto"], int] = "auto"
    max_batch_size: Optional[int] = Field(default=None, ge=1)
    device: Optional[str] = None
    limit: Optional[float] = Field(default=None, gt=0)
    use_cache: Optional[str] = None
    cache_requests: bool = False
    gen_kwargs: Optional[GenKwargs] = None
    system_instruction: Optional[str] = None
    apply_chat_template: bool = False

    retrieval: RetrievalConfig = Field(default_factory=RetrievalConfig)

    log_samples: bool = True
    predict_only: bool = False
    random_seed: int = 0
    numpy_random_seed: int = 1234
    torch_random_seed: int = 1234
    fewshot_random_seed: int = 1234

    @field_validator("model", "model_args")
    @classmethod
    def non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("must not be empty")
        return v

    @model_validator(mode="after")
    def check_judge(self) -> "RunConfig":
        if self.judge_model and not self.judge_model_args:
            raise ValueError("judge_model_args is required when judge_model is set")
        return self


class ApiKeys(BaseModel):
    """Held in memory for the run's duration only; never persisted."""

    openai_api_key: Optional[str] = None
    hf_token: Optional[str] = None


class RunSubmission(BaseModel):
    name: Optional[str] = Field(default=None, max_length=200)
    config: RunConfig
    api_keys: Optional[ApiKeys] = None


class ProgressSnapshot(BaseModel):
    phase: Optional[str] = None
    desc: Optional[str] = None
    n: Optional[int] = None
    total: Optional[int] = None
    pct: Optional[float] = None


class HeadlineMetric(BaseModel):
    metric: str
    value: float


class RunListItem(BaseModel):
    run_id: str
    name: Optional[str]
    status: RunStatus
    created_at: str
    started_at: Optional[str]
    finished_at: Optional[str]
    model: str
    tasks: List[str]
    error: Optional[str]
    headline_metric: Optional[HeadlineMetric] = None


class RunDetail(RunListItem):
    config: Dict[str, Any]
    progress: Optional[ProgressSnapshot] = None
    queue_position: Optional[int] = None
