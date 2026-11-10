"""
Site_Activation_Router.py

FastAPI router for the Clinical Trial Site Activation & Response Intelligence
Agent POC.

Expected project layout:
    src/
      schemas.py
      Site_Activation_Orchestrator.py
      Site_Activation_Router.py

This router:
- Validates request and response payloads using shared Pydantic schemas.
- Invokes the site activation orchestrator.
- Returns structured HTTP errors.
- Does not send emails or make site activation decisions.

The application must include this router in its FastAPI app.
"""

from __future__ import annotations


from typing import Any

from fastapi import APIRouter, HTTPException, Request, Response, status
from pydantic import ValidationError

from app.schemas.schemas import (
    SiteActivationRequest,
    SiteActivationResponse,
    WorkflowStatus,
)

from app.services.Site_Activation_Service import SiteActivationService
from app.observability.tracing_langsmith import (
    new_request_id,
    request_tracing_context,
    tracing_is_configured,
    traced_site_activation_workflow,
)

from app.config.settings import settings
logger = settings.get_logger()

router = APIRouter(
    prefix="/api/v1/site-activation",
    tags=["Clinical Trial Site Activation"],
)

# One orchestrator instance per process for this simple POC.
# If the orchestrator later holds request-specific state, use dependency
# injection or a factory instead.
service = SiteActivationService()




@router.post(
    "/analyze",
    response_model=SiteActivationResponse,
    status_code=status.HTTP_200_OK,
    summary="Analyze site activation documents, responses, and actions",
)
def analyze_site_activation(
    request: SiteActivationRequest,
    http_request: Request,
    http_response: Response,
) -> SiteActivationResponse:
    """
    Run the site activation POC workflow for one study site.

    Request data should contain synthetic or approved operational metadata.
    The endpoint returns recommendations and drafts for human review only.
    """
    

    # Accept a valid caller-supplied correlation ID, otherwise generate one.
    # Keep it short and conservative so it is safe for logs and metadata.
    supplied_request_id = http_request.headers.get("X-Request-ID", "").strip()
    if (
        supplied_request_id
        and len(supplied_request_id) <= 64
        and all(ch.isalnum() or ch in "-_." for ch in supplied_request_id)
    ):
        request_id = supplied_request_id
    else:
        request_id = new_request_id()

    http_response.headers["X-Request-ID"] = request_id

    try:
        # The request body is deliberately not copied into LangSmith metadata.
        # This context is inherited by LangChain calls made by the agents.
        with request_tracing_context(
            request_id,
            operation="site_activation_analyze",
            workflow="clinical_trial_site_activation",
        ):
            # The traced wrapper returns the real service result to the API,
            # while its process_outputs hook sends only a sanitized summary
            # to LangSmith. Never pass the raw request body as trace input.
            result: Any = traced_site_activation_workflow(
                request_id=request_id,
                documents_count=len(request.documents),
                responses_count=len(request.responses),
                actions_count=len(request.actions),
                workflow_callable=lambda: service.analyze_site(request=request),
            )

        # The service may return a Pydantic model or a plain dictionary.
        # Normalize known model objects before adapting and validating.
        if isinstance(result, SiteActivationResponse):
            response = result
        else:
            if hasattr(result, "model_dump"):
                result = result.model_dump(mode="python")

        # Adapt the current orchestrator's action wrapper:
        # {"action_id": ..., "matched_response_id": ..., "result": {...}}
        # to the shared SiteActivationResponse contract, whose action_analysis
        # contains ActionPrioritizationResponse objects directly.
        if not isinstance(result, SiteActivationResponse) and isinstance(result, dict) and isinstance(
            result.get("action_analysis"), list
        ):
            normalized_actions = []
            for item in result["action_analysis"]:
                if isinstance(item, dict) and isinstance(item.get("result"), dict):
                    normalized_actions.append(item["result"])
                else:
                    normalized_actions.append(item)
            result = {**result, "action_analysis": normalized_actions}

        # Validate dictionary results; retain an already validated response.
        if not isinstance(result, SiteActivationResponse):
            if not isinstance(result, dict):
                raise TypeError(
                    "SiteActivationService.analyze_site must return a "
                    "dictionary or Pydantic model; got "
                    f"{type(result).__name__}"
                )
            response = SiteActivationResponse.model_validate(result)

        if response.workflow_status == WorkflowStatus.FAILED:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "SITE_ACTIVATION_WORKFLOW_FAILED",
                    "message": (
                        "The site activation workflow could not complete. "
                        "Review the sanitized error codes in the response "
                        "or application logs."
                    ),
                    "workflow_id": response.workflow_id,
                },
            )

        return response

    except HTTPException:
        raise
    except ValidationError as exc:
        # Log schema locations and error types only. Do not log input values,
        # prompts, site response text, document contents, or other sensitive data.
        safe_errors = [
            {
                "loc": list(error.get("loc", ())),
                "type": error.get("type", "validation_error"),
                "msg": error.get("msg", "Schema validation failed"),
            }
            for error in exc.errors(include_input=False)
        ]
        logger.warning(
            "Site activation schema validation failed | errors=%s",
            safe_errors,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "INVALID_SITE_ACTIVATION_DATA",
                "message": (
                    "The request or workflow output did not match "
                    "the expected schema. Review validation_errors below."
                ),
                # Safe diagnostics only: deliberately exclude input values.
                "validation_errors": safe_errors,
            },
        ) from exc
    except ValueError as exc:
        # Request validation errors from the orchestrator should be surfaced
        # as client errors without returning arbitrary exception details.
        logger.warning(
            "Site activation request rejected: %s",
            exc.__class__.__name__,
        )
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail={
                "code": "INVALID_SITE_ACTIVATION_REQUEST",
                "message": str(exc),
            },
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected site activation workflow failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "code": "SITE_ACTIVATION_INTERNAL_ERROR",
                "message": "An unexpected workflow error occurred.",
            },
        ) from exc


@router.get(
    "/health",
    status_code=status.HTTP_200_OK,
    summary="Check router availability",
)
def site_activation_health() -> dict[str, str]:
    """Basic process-level health check; does not verify external services."""
    return {
        "status": "ok",
        "service": "site-activation",
        "scope": "router-only",
    }
