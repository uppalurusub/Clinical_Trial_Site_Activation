
"""
Site_Activation_Orchestrator.py

Clinical Trial Site Activation & Response Intelligence Agent — POC

Coordinates:
1. Document Intelligence Agent
2. Site Response Intelligence Agent
3. Action Prioritization & Follow-up Agent

Design principles:
- Explicit workflow orchestration
- Structured results
- Source-reference preservation
- Fail-visible component errors
- Human review for consequential decisions
- No autonomous site activation or external email sending

Synthetic data only for the initial POC.
"""

from __future__ import annotations

import logging
import time
import uuid

from datetime import date, datetime, timezone
from typing import Any, Optional

from app.agents.Document_Intelligence_Agent import DocumentIntelligenceAgent
from app.agents.Action_Prioritization_Followup_Agent import ActionPrioritizationFollowupAgent
from app.agents.Site_Response_Intelligence_Agent import (
    SiteResponseIntelligenceAgent,
)
from app.observability.tracing_langsmith import current_request_id




from app.config.settings import settings
logger = settings.get_logger()



class SiteActivationOrchestrator:
    """Coordinate site activation intelligence components."""

    def __init__(
        self,
        document_agent: Optional[DocumentIntelligenceAgent] = None,
        response_agent: Optional[SiteResponseIntelligenceAgent] = None,
        action_agent: Optional[
            ActionPrioritizationFollowupAgent
        ] = None,
    ) -> None:
        self.document_agent = (
            document_agent or DocumentIntelligenceAgent()
        )
        self.response_agent = (
            response_agent or SiteResponseIntelligenceAgent()
        )
        self.action_agent = (
            action_agent or ActionPrioritizationFollowupAgent()
        )

    @staticmethod
    def _validate_request(request: dict[str, Any]) -> None:
        """Validate minimum required POC input."""
        if not isinstance(request, dict):
            raise ValueError("Request must be a dictionary.")

        for field in ("study_id", "site_id"):
            if not str(request.get(field, "")).strip():
                raise ValueError(f"Missing required field: {field}")

        for field in ("documents", "responses", "actions"):
            value = request.get(field, [])

            if value is None:
                continue

            if not isinstance(value, list):
                raise ValueError(f"'{field}' must be a list.")

            if not all(isinstance(item, dict) for item in value):
                raise ValueError(
                    f"Every item in '{field}' must be a dictionary."
                )

    @staticmethod
    def _to_dict(value: Any) -> dict[str, Any]:
        """Normalize dictionaries and Pydantic v2 models for downstream logic."""
        if isinstance(value, dict):
            return value
        if hasattr(value, "model_dump"):
            dumped = value.model_dump(mode="python")
            if isinstance(dumped, dict):
                return dumped
        raise TypeError(
            f"Expected a dictionary or Pydantic model, got {type(value).__name__}"
        )

    @staticmethod
    def _build_response_context(
        response_result: Optional[dict[str, Any]],
    ) -> dict[str, Any]:
        if not response_result:
            return {
                "response_status": "NO_RESPONSE_FOUND",
                "response_summary": (
                    "No matching response was supplied. "
                    "Verify the source records before follow-up."
                ),
            }

        return {
            "response_status": response_result.get(
                "status", "UNCLASSIFIED"
            ),
            "response_summary": response_result.get(
                "suggested_next_action", ""
            ),
        }

    @staticmethod
    def _match_response(
        action: dict[str, Any],
        response_results: list[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        """
        Match a response to an action using exact action IDs.

        Deliberately avoids fuzzy matching in this initial POC.
        Responses without a matching action ID remain unmatched.
        """
        action_id = action.get("action_id")

        if not action_id:
            return None

        matches = [
            response
            for response in response_results
            if response.get("action_id") == action_id
        ]

        if not matches:
            return None

        # Multiple responses may exist for one action. Return the last
        # supplied response; production code should use timestamps and
        # authoritative source ordering instead.
        return matches[-1]

    def run(
        self,
        request: dict[str, Any],
        as_of: Optional[date] = None,
    ) -> dict[str, Any]:
        """
        Execute the site activation intelligence workflow.

        Request shape:
        {
            "study_id": "STUDY-DEMO-001",
            "site_id": "SITE-DEMO-001",
            "documents": [...],
            "responses": [...],
            "actions": [...]
        }
        """
        self._validate_request(request)

        workflow_id = str(uuid.uuid4())
        started = time.perf_counter()
        as_of = as_of or date.today()

        study_id = str(request["study_id"])
        site_id = str(request["site_id"])

        request_id = current_request_id()
        logger.info(
            "Site activation workflow started | workflow_id=%s request_id=%s",
            workflow_id,
            request_id or "not_set",
        )

        # Phase 1: Document intelligence
        document_result: Optional[dict[str, Any]] = None
        response_results: list[dict[str, Any]] = []
        action_results: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []

        try:
            document_output = self.document_agent.analyze(
                study_id=study_id,
                site_id=site_id,
                documents=request.get("documents", []),
                as_of=as_of,
            )
            # Downstream summary logic uses dictionary access (.get()).
            document_result = self._to_dict(document_output)
        except Exception:
            logger.exception(
                "Document analysis failed | workflow_id=%s",
                workflow_id,
            )
            errors.append({
                "component": "Document_Intelligence_Agent",
                "error_code": "DOCUMENT_ANALYSIS_FAILED",
            })

        # Phase 2: Site response intelligence
        for response in request.get("responses", []):
            try:
                response_output = self.response_agent.analyze(
                    response_id=str(response.get("response_id", "")),
                    response_text=response.get("response_text", ""),
                    action_id=response.get("action_id"),
                    action_description=response.get("action_description", ""),
                    site_id=site_id,
                    study_id=study_id,
                    source_ref=response.get("source_ref"),
                    received_at=response.get("received_at"),
                )
                response_result = self._to_dict(response_output)
                response_results.append(response_result)

            except Exception:
                logger.exception(
                    "Response analysis failed | workflow_id=%s",
                    workflow_id,
                )
                errors.append({
                    "component": "Site_Response_Intelligence_Agent",
                    "error_code": "RESPONSE_ANALYSIS_FAILED",
                })

        # Phase 3: Action prioritization and follow-up drafting
        for action in request.get("actions", []):
            try:
                matched_response = self._match_response(
                    action=action,
                    response_results=response_results,
                )

                response_context = self._build_response_context(
                    matched_response
                )

                # The action agent accepts the action/request as its first
                # positional argument; normalize its Pydantic result.
                action_output = self.action_agent.run(
                    action,
                    response_context=response_context,
                    as_of=as_of,
                )
                action_result = self._to_dict(action_output)

                action_results.append({
                    "action_id": action.get("action_id"),
                    "matched_response_id": (
                        matched_response.get("response_id")
                        if matched_response
                        else None
                    ),
                    "result": action_result,
                })

            except Exception:
                logger.exception(
                    "Action processing failed | workflow_id=%s",
                    workflow_id,
                )
                errors.append({
                    "component": "Action_Prioritization_Followup_Agent",
                    "error_code": "ACTION_PROCESSING_FAILED",
                })

        duration_seconds = round(
            time.perf_counter() - started, 4
        )

        # A failed component must not be silently represented as success.
        if errors:
            workflow_status = (
                "PARTIAL_SUCCESS"
                if (
                    document_result is not None
                    or response_results
                    or action_results
                )
                else "FAILED"
            )
        else:
            workflow_status = "SUCCESS"

        # No readiness or activation decision is made here.
        result = {
            "workflow_id": workflow_id,
            "workflow_name": "Clinical Trial Site Activation Intelligence",
            "workflow_status": workflow_status,
            "study_id": study_id,
            "site_id": site_id,
            "evaluated_at": datetime.now(
                timezone.utc
            ).isoformat(),
            "evaluation_date": as_of.isoformat(),
            "document_analysis": document_result,
            "response_analysis": response_results,
            "action_analysis": action_results,
            "errors": errors,
            "summary": {
                "documents_examined": (
                    document_result.get(
                        "documents_examined",
                        document_result.get("documents_reviewed", 0),
                    )
                    if document_result
                    else 0
                ),
                "document_findings": (
                    document_result.get(
                        "findings_count",
                        len(document_result.get("findings", [])),
                    )
                    if document_result
                    else 0
                ),
                "responses_analyzed": len(response_results),
                "actions_processed": len(action_results),
                "actions_requiring_review": sum(
                    1
                    for item in action_results
                    if item["result"].get("human_review_required")
                ),
            },
            "execution_time_seconds": duration_seconds,
            "human_review_required": True,
            "activation_decision": "NOT_EVALUATED",
            "external_emails_sent": False,
        }

        logger.info(
            "Site activation workflow completed | "
            "workflow_id=%s request_id=%s status=%s duration_seconds=%.4f",
            workflow_id,
            request_id or "not_set",
            workflow_status,
            duration_seconds,
        )

        return result