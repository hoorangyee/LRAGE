# Lazy re-exports (PEP 562). Importing `lrage.evaluator` pulls in the full
# evaluation stack (torch, pyserini, ...), which subpackages like `lrage.webapp`
# must be importable without.
_EVALUATOR_EXPORTS = ("evaluate", "simple_evaluate")

__all__ = list(_EVALUATOR_EXPORTS)


def __getattr__(name):
    if name in _EVALUATOR_EXPORTS:
        from lrage import evaluator

        return getattr(evaluator, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(list(globals().keys()) + list(_EVALUATOR_EXPORTS))
