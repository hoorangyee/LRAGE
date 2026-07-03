"""The real evaluation worker: a port of the Gradio bridge
(lrage/gui/gui_utils/evaluation.py:eval_tasks) with the hardcoded parameters
lifted into RunConfig, per-run output directories, scoped API keys, and
progress capture.

`run_evaluation(ctx)` takes only what a future process-based runner could
also provide (config + paths + publisher/token), so hard-cancellation via a
subprocess can be added behind this same seam later.
"""
import gc
import logging
import os
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

from lrage.webapp.jobs.manager import JobContext
from lrage.webapp.jobs.progress import capture_lrage_logs, tqdm_progress_hook

# Test seam: monkeypatch this with a fake to avoid importing the evaluation
# stack (torch, pyserini, ...). None means "import the real one on first use".
simple_evaluate = None


def _get_simple_evaluate():
    if simple_evaluate is not None:
        return simple_evaluate
    from lrage.evaluator import simple_evaluate as real

    return real


@contextmanager
def log_to_file(path: Path):
    """Persist the run's "lrage" log lines so the Logs tab survives server
    restarts (the in-memory event buffer does not)."""
    handler = logging.FileHandler(path, encoding="utf-8")
    handler.setLevel(logging.INFO)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    )
    logger = logging.getLogger("lrage")
    logger.addHandler(handler)
    try:
        yield
    finally:
        logger.removeHandler(handler)
        handler.close()


@contextmanager
def scoped_env(values: Dict[str, Optional[str]]):
    """Set env vars for the run's duration and restore afterwards. Only the
    single worker thread mutates the environment, one run at a time."""
    saved: Dict[str, Optional[str]] = {}
    try:
        for key, value in values.items():
            if value:
                saved[key] = os.environ.get(key)
                os.environ[key] = value
        yield
    finally:
        for key, old in saved.items():
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old


def build_summary(result_dict: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Flatten the results dict into table rows — a port of the Gradio
    build_table (lrage/gui/gui_utils/evaluation.py:14) without pandas."""
    rows: List[Dict[str, Any]] = []
    for task_key in sorted(result_dict.get("results", {}).keys()):
        metrics = dict(result_dict["results"][task_key])
        n_shot = str(result_dict.get("n-shot", {}).get(task_key, ""))
        alias = metrics.pop("alias", task_key)
        for metric_key, value in sorted(metrics.items()):
            metric, _, metric_filter = metric_key.partition(",")
            if metric.endswith("_stderr"):
                continue
            if not isinstance(value, (int, float)):
                continue
            stderr = metrics.get(f"{metric}_stderr,{metric_filter}")
            rows.append(
                {
                    "task": alias,
                    "n_shot": n_shot,
                    "metric": metric,
                    "value": float(value),
                    "stderr": float(stderr)
                    if isinstance(stderr, (int, float))
                    else None,
                }
            )
    return rows


def run_evaluation(ctx: JobContext) -> None:
    import datasets

    from lrage.loggers import EvaluationTracker

    job = ctx.job
    cfg = job.submission.config
    api_keys = job.submission.api_keys

    output_dir = ctx.settings.output_root / job.run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    datasets.config.HF_DATASETS_TRUST_REMOTE_CODE = True
    evaluation_tracker = EvaluationTracker(output_path=str(output_dir))

    env = {
        "OPENAI_API_KEY": api_keys.openai_api_key if api_keys else None,
        "HF_TOKEN": api_keys.hf_token if api_keys else None,
    }

    retrieval = cfg.retrieval
    evaluate_fn = _get_simple_evaluate()

    try:
        with scoped_env(env), log_to_file(
            output_dir / "run.log"
        ), capture_lrage_logs(ctx.publish), tqdm_progress_hook(
            ctx.publish, ctx.token
        ):
            results = evaluate_fn(
                model=cfg.model,
                model_args=cfg.model_args,
                judge_model=cfg.judge_model,
                judge_model_args=cfg.judge_model_args,
                judge_device=cfg.judge_device,
                tasks=list(cfg.tasks),
                num_fewshot=cfg.num_fewshot,
                fewshot_as_multiturn=cfg.fewshot_as_multiturn,
                batch_size=cfg.batch_size,
                max_batch_size=cfg.max_batch_size,
                device=cfg.device,
                limit=cfg.limit,
                use_cache=cfg.use_cache or None,
                cache_requests=cfg.cache_requests,
                gen_kwargs=cfg.gen_kwargs.to_args_string() if cfg.gen_kwargs else None,
                judge_gen_kwargs=cfg.judge_gen_kwargs.to_args_string()
                if cfg.judge_gen_kwargs
                else None,
                system_instruction=cfg.system_instruction,
                apply_chat_template=cfg.apply_chat_template,
                retrieve_docs=retrieval.retrieve_docs,
                top_k=retrieval.top_k,
                retriever=retrieval.retriever,
                retriever_args=retrieval.retriever_args,
                rerank=retrieval.rerank,
                reranker=retrieval.reranker,
                reranker_args=retrieval.reranker_args,
                log_samples=cfg.log_samples,
                predict_only=cfg.predict_only,
                evaluation_tracker=evaluation_tracker,
                random_seed=cfg.random_seed,
                numpy_random_seed=cfg.numpy_random_seed,
                torch_random_seed=cfg.torch_random_seed,
                fewshot_random_seed=cfg.fewshot_random_seed,
            )

        if results is None:
            raise RuntimeError("Evaluation returned no results")

        ctx.publish({"type": "phase", "phase": "saving"})
        samples = results.pop("samples", None) if cfg.log_samples else None
        evaluation_tracker.save_results_aggregated(
            results=results, samples=samples
        )
        if cfg.log_samples and samples:
            for task_name in results.get("configs", {}):
                evaluation_tracker.save_results_samples(
                    task_name=task_name, samples=samples[task_name]
                )

        results_files = sorted(output_dir.glob("**/results_*.json"))
        samples_files = sorted(str(p) for p in output_dir.glob("**/samples_*.jsonl"))
        ctx.repo.set_outputs(
            job.run_id,
            output_dir=str(output_dir),
            results_path=str(results_files[-1]) if results_files else None,
            samples_paths=samples_files,
            summary=build_summary(results),
        )
    finally:
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
