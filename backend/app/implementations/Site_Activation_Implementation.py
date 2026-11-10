"""
Site_Activation_Implementation.py

Application implementation layer for the Clinical Trial Site Activation POC.
Coordinates with SiteActivationOrchestrator and validates its output against
SiteActivationResponse.

Adjust the schemas import below if your project stores schemas in another module.
"""

from __future__ import annotations

from datetime import date
from typing import Any

from pydantic import ValidationError

# These imports follow the user's current backend/app package layout.
# If schemas.py is in app/models/ or another package, update only this import.
from app.schemas.schemas import SiteActivationRequest, SiteActivationResponse, WorkflowStatus
from app.orchestration.Site_Activation_Orchestrator import SiteActivationOrchestrator


class SiteActivationImplementation:
    """Call the orchestrator and normalize its result for the API response schema."""

    def __init__(
        self,
        orchestrator: SiteActivationOrchestrator | None = None,
    ) -> None:
        self.orchestrator = orchestrator or SiteActivationOrchestrator()

    @classmethod
    def _to_plain(cls, value: Any) -> Any:
        """Recursively convert Pydantic v2 models and containers to plain values."""
        if hasattr(value, "model_dump"):
            return cls._to_plain(value.model_dump(mode="python"))
        if isinstance(value, dict):
            return {key: cls._to_plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [cls._to_plain(item) for item in value]
        return value

    @staticmethod
    def _enum_value(value: Any) -> Any:
        return value.value if hasattr(value, "value") else value

    @classmethod
    def _normalize_action_analysis(cls, raw_actions: Any) -> list[dict[str, Any]]:
        """
        Flatten the orchestrator wrapper:
          {"action_id": ..., "matched_response_id": ..., "result": {...}}

        The strict ActionPrioritizationResponse schema rejects wrapper-only
        action_id and matched_response_id fields, so they must not be injected
        into the inner result.
        """
        normalized: list[dict[str, Any]] = []

        for raw_item in raw_actions or []:
            item = cls._to_plain(raw_item)
            if not isinstance(item, dict):
                continue

            inner_result = item.get("result")
            if isinstance(inner_result, dict):
                normalized.append(inner_result)
                continue

            # If already in the direct response shape, discard only wrapper-only
            # fields that are forbidden by the response model.
            direct_item = dict(item)
            direct_item.pop("matched_response_id", None)
            if "priority_result" in direct_item and "followup_draft" in direct_item:
                # action_id is present inside priority_result in this POC.
                direct_item.pop("action_id", None)
            normalized.append(direct_item)

        return normalized

    @staticmethod
    def _derive_workflow_status(
        document_analysis: dict[str, Any] | None,
        response_analysis: list[dict[str, Any]],
        action_analysis: list[dict[str, Any]],
    ) -> WorkflowStatus:
        findings = (document_analysis or {}).get("findings", [])
        has_critical_document_finding = any(
            str(SiteActivationImplementation._enum_value(
                finding.get("severity", "")
            )).upper() == "CRITICAL"
            for finding in findings
            if isinstance(finding, dict)
        )
        has_escalation = any(
            str(SiteActivationImplementation._enum_value(
                item.get("status", "")
            )).upper() in {"ESCALATION", "NEEDS_ESCALATION"}
            for item in response_analysis
            if isinstance(item, dict)
        )
        has_critical_action = any(
            str(SiteActivationImplementation._enum_value(
                (item.get("priority_result") or {}).get("priority", "")
            )).upper() == "CRITICAL"
            for item in action_analysis
            if isinstance(item, dict)
        )

        if has_critical_document_finding or has_escalation or has_critical_action:
            # Keep the workflow status value aligned to the enum defined in schemas.py.
            try:
                return WorkflowStatus("REVIEW_REQUIRED")
            except ValueError:
                pass
        try:
            return WorkflowStatus("READY_FOR_REVIEW")
        except ValueError:
            # Last-resort fallback for projects whose enum has different labels.
            return WorkflowStatus("SUCCESS")

    def run(
        self,
        request: SiteActivationRequest | dict[str, Any],
        *,
        as_of: date | None = None,
    ) -> SiteActivationResponse:
        """Run orchestration and validate the final SiteActivationResponse."""
        validated_request = (
            request
            if isinstance(request, SiteActivationRequest)
            else SiteActivationRequest.model_validate(request)
        )

        reference_date = as_of or validated_request.as_of or date.today()
        request_payload = validated_request.model_dump(mode="python")

        raw_result = self.orchestrator.run(
            request=request_payload,
            as_of=reference_date,
        )
        result = self._to_plain(raw_result)

        if not isinstance(result, dict):
            raise TypeError(
                "SiteActivationOrchestrator.run() must return a dictionary "
                "or Pydantic model."
            )

        document_analysis = result.get("document_analysis")
        response_analysis = self._to_plain(result.get("response_analysis", [])) or []
        action_analysis = self._normalize_action_analysis(
            result.get("action_analysis", [])
        )

        workflow_status = result.get("workflow_status")
        if workflow_status is not None:
            try:
                workflow_status = WorkflowStatus(self._enum_value(workflow_status))
            except (ValueError, TypeError):
                workflow_status = self._derive_workflow_status(
                    document_analysis if isinstance(document_analysis, dict) else None,
                    response_analysis,
                    action_analysis,
                )
        else:
            workflow_status = self._derive_workflow_status(
                document_analysis if isinstance(document_analysis, dict) else None,
                response_analysis,
                action_analysis,
            )

        # Include required fields identified by the actual Pydantic validation log:
        # workflow_id and execution_time_seconds.
        response_payload = {
            "workflow_id": result.get("workflow_id"),
            "study_id": result.get("study_id", validated_request.study_id),
            "site_id": result.get("site_id", validated_request.site_id),
            "workflow_status": workflow_status,
            "document_analysis": document_analysis,
            "response_analysis": response_analysis,
            "action_analysis": action_analysis,
            "summary": result.get("summary", {}),
            "errors": result.get("errors", []),
            "execution_time_seconds": result.get("execution_time_seconds", 0.0),
        }

        try:
            return SiteActivationResponse.model_validate(response_payload)
        except ValidationError:
            # Keep detailed validation information in server logs/call stack.
            # The router can return a sanitized error to the client.
            raise


# Compatibility alias for existing imports, if used.
Site_Activation_Implementation = SiteActivationImplementation
