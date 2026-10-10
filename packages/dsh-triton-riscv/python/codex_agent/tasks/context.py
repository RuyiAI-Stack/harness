"""Internal execution context, not a model-controlled or ambient environment flag."""
from contextvars import ContextVar

defer_memory = ContextVar("worker_defers_memory", default=False)
