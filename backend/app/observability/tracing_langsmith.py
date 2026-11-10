"""Privacy-conscious shared tracing helpers for LangChain, LangGraph, and FastAPI.

Usage:
    request_id = new_request_id()
    config = tracing_config(
        request_id,
        run_name="site_activation_analysis",
        operation="site_activation_analysis",
    )

Pass ``config`` to LangChain Runnable.invoke/ainvoke calls. For LangGraph,
pass it to graph.invoke/ainvoke. Keep raw clinical/site payloads out of
trace metadata and avoid passing them to traceable wrapper functions.
"""
from __future__ import annotations


import os
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Callable, Iterator

from langsmith import traceable, tracing_context

from app.config.settings import settings
logger = settings.get_logger()


_request_id_var: ContextVar[str | None] = ContextVar(
    "site_activation_request_id", default=None
)
_trace_extra_var: ContextVar[dict[str, Any]] = ContextVar(
    "site_activation_trace_extra", default={}
)
_workflow_callable_var: ContextVar[Callable[[], Any] | None] = ContextVar(
    "site_activation_workflow_callable", default=None
)


def new_request_id() -> str:
    """Return a correlation ID for one incoming API request."""
    return str(uuid.uuid4())


def _safe_environment() -> str:
    """Return a normalized environment label without assuming production."""
    return os.getenv("APP_ENV", "development").strip().lower() or "development"


def langsmith_metadata(request_id: str, **extra: Any) -> dict[str, Any]:
    """Build allowlisted operational metadata; never pass PHI or secrets here."""
    metadata: dict[str, Any] = {
        "request_id": request_id,
        "service": os.getenv("SERVICE_NAME", "clinical-trial-site-activation"),
        "environment": _safe_environment(),
    }

    # Only accept explicitly approved scalar metadata. Callers must not pass
    # prompts, response text, document contents, credentials, or identifiers
    # that could identify a patient or contain confidential site information.
    allowed_keys = {
        "operation",
        "workflow",
        "agent",
        "model_provider",
        "model_name",
        "deployment_version",
    }
    for key, value in extra.items():
        if key in allowed_keys and isinstance(value, (str, int, float, bool)):
            metadata[key] = value
    return metadata


def tracing_config(request_id: str, **extra: Any) -> dict[str, Any]:
    """Return a LangChain/LangGraph RunnableConfig with request correlation.

    Pass this configuration to the actual ``invoke``/``ainvoke`` call.
    ``thread_id`` is useful for LangGraph's configurable state; it does not,
    by itself, enable LangSmith tracing.
    """
    run_name = extra.pop("run_name", "site_activation_request")
    operation = extra.get("operation")
    environment = _safe_environment()

    tags = [
        "clinical-trial-site-activation",
        f"environment:{environment}",
    ]
    if isinstance(operation, str) and operation.strip():
        tags.append(f"operation:{operation.strip()}")

    return {
        "run_name": run_name,
        "tags": tags,
        "metadata": langsmith_metadata(request_id, **extra),
        "configurable": {"thread_id": request_id},
        "recursion_limit": int(os.getenv("LANGGRAPH_RECURSION_LIMIT", "25")),
    }


@traceable(name="clinical_ai_operation", run_type="chain")
def traced_operation(
    operation: str,
    request_id: str,
    payload_present: bool,
) -> dict[str, Any]:
    """Trace a small operational event without capturing the raw payload.

    Do not change this function to accept a request body or document text:
    @traceable may capture function inputs in LangSmith.
    """
    logger.info(
        "traced_operation operation=%s request_id=%s payload_present=%s",
        operation,
        request_id,
        payload_present,
    )
    return {
        "operation": operation,
        "request_id": request_id,
        "payload_present": payload_present,
    }


def tracing_is_configured() -> bool:
    """Return whether the essential LangSmith environment settings are present."""
    return (
        os.getenv("LANGSMITH_TRACING", "").strip().lower() == "true"
        and bool(os.getenv("LANGSMITH_API_KEY"))
    )


@contextmanager
def request_tracing_context(
    request_id: str,
    **extra: Any,
) -> Iterator[None]:
    """Set request-scoped LangSmith context for all nested LangChain calls.

    Only pass safe operational metadata. Do not include raw request bodies,
    site response text, document contents, PHI, credentials, or secrets.
    """
    token_id = _request_id_var.set(request_id)
    token_extra = _trace_extra_var.set(dict(extra))
    environment = _safe_environment()
    tags = [
        "clinical-trial-site-activation",
        f"environment:{environment}",
    ]
    operation = extra.get("operation")
    if isinstance(operation, str) and operation.strip():
        tags.append(f"operation:{operation.strip()}")

    try:
        with tracing_context(
            enabled=tracing_is_configured(),
            project_name=os.getenv(
                "LANGSMITH_PROJECT", "clinical-trial-site-activation"
            ),
            tags=tags,
            metadata=langsmith_metadata(request_id, **extra),
        ):
            yield
    finally:
        _trace_extra_var.reset(token_extra)
        _request_id_var.reset(token_id)


def current_request_id() -> str | None:
    """Return the request ID for the current execution context, if set."""
    return _request_id_var.get()


def current_tracing_config(**extra: Any) -> dict[str, Any]:
    """Create a RunnableConfig for an actual LangChain model/chain invocation."""
    request_id = current_request_id()
    if not request_id:
        # Allow agents to run in isolated unit tests without request middleware.
        return {}

    merged = dict(_trace_extra_var.get())
    merged.update(extra)
    merged.setdefault("operation", "site_activation_llm")
    return tracing_config(
        request_id,
        run_name=merged.pop("run_name", "site_activation_llm"),
        **merged,
    )


def _sanitize_workflow_output(result: Any) -> dict[str, Any]:
    """Return a minimal workflow summary for LangSmith; exclude raw results."""
    if hasattr(result, "model_dump"):
        summary = result.model_dump(mode="python")
    elif isinstance(result, dict):
        summary = result
    else:
        summary = {}

    document_analysis = summary.get("document_analysis") or {}
    response_analysis = summary.get("response_analysis") or []
    action_analysis = summary.get("action_analysis") or []
    if hasattr(document_analysis, "model_dump"):
        document_analysis = document_analysis.model_dump(mode="python")

    return {
        "workflow_status": str(summary.get("workflow_status", "unknown")),
        "documents_examined": int(
            document_analysis.get("documents_examined", 0)
            if isinstance(document_analysis, dict) else 0
        ),
        "responses_analyzed": (
            len(response_analysis) if isinstance(response_analysis, list) else 0
        ),
        "actions_processed": (
            len(action_analysis) if isinstance(action_analysis, list) else 0
        ),
    }


def _sanitize_workflow_inputs(inputs: dict[str, Any]) -> dict[str, Any]:
    """Allow only non-sensitive root-trace inputs."""
    allowed = {"request_id", "documents_count", "responses_count", "actions_count"}
    return {key: value for key, value in inputs.items() if key in allowed}


@traceable(
    name="site_activation_workflow",
    run_type="chain",
    process_inputs=_sanitize_workflow_inputs,
    process_outputs=_sanitize_workflow_output,
)
def traced_site_activation_workflow(
    request_id: str,
    documents_count: int,
    responses_count: int,
    actions_count: int,
    workflow_callable: Callable[[], Any],
) -> Any:
    """Trace workflow execution while returning its real result to the API.

    LangSmith records only sanitized inputs and output summary. The callback is
    deliberately removed from captured inputs by ``process_inputs``.
    """
    return workflow_callable()
