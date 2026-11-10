"""
Site_Response_Intelligence_Agent.py

Clinical Trial Site Activation & Response Intelligence Agent — POC.

Uses:
- Shared Pydantic contracts from src.schemas
- OpenRouter-compatible ChatOpenAI from llm_config.get_llm()
- Structured LLM classification using Pydantic output parsing
- Deterministic fallback rules if the LLM is unavailable or output is invalid

This agent classifies operational site responses only. It does not close actions,
send email, or make regulatory/site activation decisions.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.exceptions import OutputParserException
from app.config.settings import settings
logger = settings.get_logger()
llm = settings.get_model_client()

from app.schemas.schemas import (
    ResponseStatus,
    SiteResponseAnalysis,
    SiteResponseRequest,
)




class SiteResponseIntelligenceAgent:
    """Classify site responses with schema-validated output and safe fallback."""

    def __init__(
        self,
        llm=None,
        enable_llm_classification: bool = True,
    ) -> None:
        """
        Args:
            llm: Optional LangChain chat model. If omitted and enabled,
                initialize via llm_config.get_llm().
            enable_llm_classification: Set False for deterministic-only mode.
        """
        self.llm = llm
        self.enable_llm_classification = enable_llm_classification
        self._classification_chain = None

        if self.enable_llm_classification and self.llm is None:
            try:
                
                self.llm = settings.get_model_client()
            except Exception as exc:
                logger.warning(
                    "LLM initialization unavailable; using deterministic "
                    "classification. error_type=%s",
                    type(exc).__name__,
                )
                self.llm = None

        if self.llm is not None and self.enable_llm_classification:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
You classify clinical-trial site responses for operational follow-up.
Return only a JSON object with these keys:
status, classification_reasons, unresolved_items, follow_up_required,
suggested_next_action.

Allowed status values:
ANSWERED, PARTIALLY_ANSWERED, PENDING, CLARIFICATION_REQUIRED,
ESCALATION_REQUIRED, UNCLASSIFIED.

Rules:
- Use only facts in the supplied request.
- A claim that a document was sent is not independent proof it was received,
  valid, complete, or accepted.
- If the response mentions both completed and outstanding items, use
  PARTIALLY_ANSWERED.
- If the sender requests clarification, use CLARIFICATION_REQUIRED.
- If the response signals inability to comply or a serious unresolved issue,
  use ESCALATION_REQUIRED.
- If a future action is promised or work remains pending, use PENDING unless
  completed and outstanding items are both present.
- If the response is ambiguous, use UNCLASSIFIED or CLARIFICATION_REQUIRED.
- Do not infer medical, regulatory, or site activation decisions.
- Never close the underlying action.
""".strip(),
                    ),
                    (
                        "human",
                        """
Study ID: {study_id}
Site ID: {site_id}
Response ID: {response_id}
Action ID: {action_id}
Action description: {action_description}
Source reference: {source_ref}
Response text:
<response>
{response_text}
</response>
""".strip(),
                    ),
                ]
            )
            self._classification_chain = prompt | self.llm | StrOutputParser()

    @staticmethod
    def _next_action(status: ResponseStatus) -> str:
        actions = {
            ResponseStatus.ANSWERED: (
                "Verify the response against the requested action and "
                "supporting evidence before updating its status."
            ),
            ResponseStatus.PARTIALLY_ANSWERED: (
                "Identify remaining items and request the outstanding information."
            ),
            ResponseStatus.PENDING: (
                "Record the stated commitment if verified and follow up "
                "according to the applicable deadline."
            ),
            ResponseStatus.CLARIFICATION_REQUIRED: (
                "Clarify the request and document the agreed next step."
            ),
            ResponseStatus.ESCALATION_REQUIRED: (
                "Route the issue to an authorized clinical operations reviewer."
            ),
            ResponseStatus.UNCLASSIFIED: (
                "Manually review the response and determine the appropriate status."
            ),
        }
        return actions[status]

    @staticmethod
    def _rule_classify(response_text: str) -> dict[str, Any]:
        """Transparent keyword-based fallback; not a validated NLP model."""
        text = response_text.lower()
        reasons: list[str] = []
        unresolved_items: list[str] = []

        escalation_phrases = [
            "unable to comply",
            "cannot comply",
            "cannot provide",
            "not possible",
            "serious issue",
            "urgent escalation",
            "escalate this",
            "regulatory concern",
        ]
        clarification_phrases = [
            "please clarify",
            "need clarification",
            "unclear",
            "not sure what is required",
            "which document",
            "what do you mean",
            "please explain",
        ]
        partial_phrases = [
            "partially",
            "some documents",
            "remaining documents",
            "still outstanding",
            "will send the rest",
            "awaiting signature",
            "awaiting approval",
            "in progress",
            "working on it",
        ]
        pending_phrases = [
            "will send",
            "will provide",
            "will submit",
            "working on",
            "not yet",
            "pending",
            "awaiting",
            "by next week",
            "by tomorrow",
            "later this week",
        ]
        answered_phrases = [
            "attached",
            "please find",
            "has been submitted",
            "has been provided",
            "completed",
            "already submitted",
            "uploaded",
            "confirmed",
            "done",
        ]

        if not text.strip():
            status = ResponseStatus.UNCLASSIFIED
            reasons.append("No response text was supplied.")
        elif any(phrase in text for phrase in escalation_phrases):
            status = ResponseStatus.ESCALATION_REQUIRED
            reasons.append("Potential escalation language detected.")
        elif any(phrase in text for phrase in clarification_phrases):
            status = ResponseStatus.CLARIFICATION_REQUIRED
            reasons.append("Clarification language detected.")
        elif any(phrase in text for phrase in partial_phrases):
            status = ResponseStatus.PARTIALLY_ANSWERED
            reasons.append(
                "Partial completion or outstanding work indicated."
            )
        elif any(phrase in text for phrase in pending_phrases):
            status = ResponseStatus.PENDING
            reasons.append("Future action or pending work indicated.")
        elif any(phrase in text for phrase in answered_phrases):
            status = ResponseStatus.ANSWERED
            reasons.append("Potential completion language detected.")
        else:
            status = ResponseStatus.UNCLASSIFIED
            reasons.append("No reliable match found using POC keyword rules.")

        # Simple extractive fallback for explicitly marked outstanding items.
        patterns = [
            r"([^.!?]*\b(?:still outstanding|remaining|awaiting|pending)\b[^.!?]*)",
            r"([^.!?]*\b(?:not yet|will send the rest)\b[^.!?]*)",
        ]
        for pattern in patterns:
            for match in re.finditer(pattern, response_text, re.IGNORECASE):
                candidate = match.group(1).strip()
                if candidate and candidate not in unresolved_items:
                    unresolved_items.append(candidate)

        return {
            "status": status,
            "classification_reasons": reasons,
            "unresolved_items": unresolved_items,
            "follow_up_required": status != ResponseStatus.ANSWERED,
            "suggested_next_action": SiteResponseIntelligenceAgent._next_action(
                status
            ),
        }

    @staticmethod
    def _validate_request(
        request: SiteResponseRequest | dict[str, Any],
    ) -> SiteResponseRequest:
        if isinstance(request, SiteResponseRequest):
            return request
        return SiteResponseRequest.model_validate(request)

    @staticmethod
    def _parse_llm_result(text: str) -> dict[str, Any]:
        """Parse JSON returned by the LLM and validate its fields."""
        import json

        content = (text or "").strip()
        # Accommodate providers that wrap JSON in Markdown fences.
        content = re.sub(r"^```(?:json)?\s*", "", content, flags=re.IGNORECASE)
        content = re.sub(r"\s*```$", "", content)

        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("LLM output must be a JSON object.")

        status = ResponseStatus(data["status"])
        reasons = data.get("classification_reasons", [])
        unresolved = data.get("unresolved_items", [])
        follow_up = data.get("follow_up_required")

        if not isinstance(reasons, list) or not all(
            isinstance(item, str) for item in reasons
        ):
            raise ValueError("classification_reasons must be a list of strings.")

        if not isinstance(unresolved, list) or not all(
            isinstance(item, str) for item in unresolved
        ):
            raise ValueError("unresolved_items must be a list of strings.")

        if not isinstance(follow_up, bool):
            raise ValueError("follow_up_required must be a boolean.")

        return {
            "status": status,
            "classification_reasons": reasons,
            "unresolved_items": unresolved,
            "follow_up_required": follow_up,
            "suggested_next_action": SiteResponseIntelligenceAgent._next_action(
                status
            ),
        }

    def analyze(
        self,
        request: SiteResponseRequest | dict[str, Any] | None = None,
        *,
        response_id: Optional[str] = None,
        response_text: Optional[str] = None,
        action_id: Optional[str] = None,
        action_description: str = "",
        site_id: Optional[str] = None,
        study_id: Optional[str] = None,
        source_ref: Optional[str] = None,
        received_at=None,
    ) -> SiteResponseAnalysis:
        """
        Preferred API:
            analyze(SiteResponseRequest(...))

        Backward-compatible API:
            analyze(response_id="...", response_text="...", site_id="...",
                    study_id="...", action_id="...")
        """
        if request is None:
            payload = {
                "response_id": response_id,
                "response_text": response_text,
                "action_id": action_id,
                "action_description": action_description,
                "site_id": site_id,
                "study_id": study_id,
                "source_ref": source_ref,
                "received_at": received_at,
            }
            validated = SiteResponseRequest.model_validate(payload)
        else:
            validated = self._validate_request(request)

        if self._classification_chain is not None:
            try:
                raw = self._classification_chain.invoke(
                    {
                        "study_id": validated.study_id,
                        "site_id": validated.site_id,
                        "response_id": validated.response_id,
                        "action_id": validated.action_id or "Not supplied",
                        "action_description": validated.action_description,
                        "source_ref": validated.source_ref or "Not supplied",
                        "response_text": validated.response_text,
                    }
                )
                classified = self._parse_llm_result(raw)
                method = "llm"
            except Exception as exc:
                logger.warning(
                    "LLM response classification failed; using rules. "
                    "error_type=%s",
                    type(exc).__name__,
                )
                classified = self._rule_classify(validated.response_text)
                method = "keyword_rules_fallback"
        else:
            classified = self._rule_classify(validated.response_text)
            method = "keyword_rules"

        status = classified["status"]
        result = SiteResponseAnalysis.model_validate(
            {
                "agent_name": "Site_Response_Intelligence_Agent",
                "study_id": validated.study_id,
                "site_id": validated.site_id,
                "response_id": validated.response_id,
                "action_id": validated.action_id,
                "source_ref": validated.source_ref,
                "status": status,
                "classification_method": method,
                "classification_reasons": classified[
                    "classification_reasons"
                ],
                "unresolved_items": classified["unresolved_items"],
                "follow_up_required": classified["follow_up_required"],
                "suggested_next_action": classified["suggested_next_action"],
                "requires_human_review": (
                    status != ResponseStatus.ANSWERED or method != "llm"
                ),
                "action_closed": False,
            }
        )
        return result