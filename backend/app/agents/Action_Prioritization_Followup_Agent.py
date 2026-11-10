"""
Action_Prioritization_Followup_Agent.py

Clinical Trial Site Activation & Response Intelligence Agent — POC.

Uses:
- Shared Pydantic contracts from src.schemas
- OpenRouter-compatible ChatOpenAI configuration from llm_config.get_llm()
- Deterministic, explainable priority scoring
- LLM-assisted follow-up drafting with a template fallback
- Pydantic validation on request and response objects

This module does not send email or make site activation decisions.
"""

from __future__ import annotations


import re
from datetime import date
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from app.schemas.schemas import (
    ActionPrioritizationRequest,
    ActionPrioritizationResponse,
    FollowupDraft,
    PriorityAssessment,
    PriorityLevel,
    ResponseStatus,
)
from app.config.settings import settings
logger = settings.get_logger()
llm = settings.get_model_client()




class ActionPrioritizationFollowupAgent:
    """Prioritize one activation action and draft a human-reviewed follow-up."""

    SCORING_VERSION = "poc-v1"

    def __init__(
        self,
        llm=None,
        enable_llm_drafting: bool = True,
    ) -> None:
        """
        Args:
            llm: Optional LangChain chat model. If omitted and LLM drafting
                is enabled, the model is created using llm_config.get_llm().
            enable_llm_drafting: Set False to use templates only.
        """
        self.llm = llm
        self.enable_llm_drafting = enable_llm_drafting
        self._draft_chain = None

        if self.enable_llm_drafting and self.llm is None:
            try:
                # Supports both running from the project root and from src.
                self.llm = settings.get_model_client()
            except Exception as exc:
                logger.warning(
                    "LLM initialization unavailable; template drafting enabled. "
                    "reason=%s",
                    type(exc).__name__,
                )
                self.llm = None

        if self.llm is not None and self.enable_llm_drafting:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
You draft concise, professional clinical-trial site activation follow-up
emails. Use only the facts provided. Do not invent dates, documents,
commitments, regulatory requirements, or prior communications. Do not include
patient identifiers or unnecessary sensitive data. Do not claim that a task
is complete unless the supplied facts establish that. The result is a draft
for human review and must never be sent automatically.

Return exactly this format:
Subject: <short subject>

<body>
""".strip(),
                    ),
                    ("human", "{context}"),
                ]
            )
            self._draft_chain = prompt | self.llm | StrOutputParser()

    @staticmethod
    def _as_request(
        request_or_action: ActionPrioritizationRequest | dict,
        response_context: Optional[dict] = None,
        as_of: Optional[date] = None,
    ) -> ActionPrioritizationRequest:
        """
        Validate an ActionPrioritizationRequest. A legacy-style action dict
        is also accepted for compatibility with the earlier orchestrator.
        """
        if isinstance(request_or_action, ActionPrioritizationRequest):
            return request_or_action

        if not isinstance(request_or_action, dict):
            raise TypeError(
                "Input must be ActionPrioritizationRequest or a dictionary."
            )

        # Accept either the schema's {"action": ..., "response_context": ...}
        # structure or the previous agent API's direct action dictionary.
        if "action" in request_or_action:
            payload = dict(request_or_action)
        else:
            payload = {
                "action": request_or_action,
                "response_context": response_context or {},
                "as_of": as_of or date.today(),
            }

        if response_context is not None and "response_context" in payload:
            payload["response_context"] = response_context
        if as_of is not None:
            payload["as_of"] = as_of

        # The earlier orchestrator uses NO_RESPONSE_FOUND as a display
        # marker; the shared schema represents an absent status as None.
        context = payload.get("response_context") or {}
        if isinstance(context, dict):
            status = context.get("response_status")
            if status == "NO_RESPONSE_FOUND":
                context = {**context, "response_status": None}
                payload["response_context"] = context

        return ActionPrioritizationRequest.model_validate(payload)

    @staticmethod
    def _calculate_priority(
        request: ActionPrioritizationRequest,
    ) -> PriorityAssessment:
        """Calculate a transparent 0–100 POC priority score."""
        action = request.action
        as_of = request.as_of
        score = 0
        reasons: list[str] = []

        criticality_points = {
            PriorityLevel.CRITICAL: 30,
            PriorityLevel.HIGH: 22,
            PriorityLevel.MEDIUM: 12,
            PriorityLevel.LOW: 5,
        }
        base_points = criticality_points[action.criticality]
        score += base_points
        reasons.append(
            f"{action.criticality.value.title()} business criticality (+{base_points})"
        )

        days_overdue = 0
        if action.due_date is not None:
            days_until_due = (action.due_date - as_of).days
            if days_until_due < 0:
                days_overdue = abs(days_until_due)
                deadline_points = min(30, 10 + days_overdue * 2)
                score += deadline_points
                reasons.append(
                    f"Overdue by {days_overdue} day(s) (+{deadline_points})"
                )
            elif days_until_due <= 3:
                score += 20
                reasons.append("Due within 3 days (+20)")
            elif days_until_due <= 7:
                score += 10
                reasons.append("Due within 7 days (+10)")
            else:
                reasons.append("Deadline is more than 7 days away (+0)")
        else:
            reasons.append("No due date supplied (+0)")

        missing_count = len(action.missing_documents)
        if missing_count:
            document_points = min(20, missing_count * 5)
            score += document_points
            reasons.append(
                f"{missing_count} missing document(s) (+{document_points})"
            )

        if action.dependency_blocked:
            score += 20
            reasons.append("Blocked by a dependency (+20)")

        score = min(100, max(0, score))

        if score >= 75:
            priority = PriorityLevel.CRITICAL
        elif score >= 50:
            priority = PriorityLevel.HIGH
        elif score >= 25:
            priority = PriorityLevel.MEDIUM
        else:
            priority = PriorityLevel.LOW

        return PriorityAssessment(
            action_id=action.action_id,
            study_id=action.study_id,
            site_id=action.site_id,
            priority_score=score,
            priority=priority,
            priority_reasons=reasons,
            days_overdue=days_overdue,
            due_date=action.due_date,
            scoring_version=ActionPrioritizationFollowupAgent.SCORING_VERSION,
        )

    @staticmethod
    def _make_context(
        request: ActionPrioritizationRequest,
        assessment: PriorityAssessment,
    ) -> str:
        action = request.action
        response = request.response_context

        # Only include operational fields required to draft the follow-up.
        facts = [
            f"Study ID: {action.study_id}",
            f"Site ID: {action.site_id}",
            f"Action ID: {action.action_id}",
            f"Action: {action.action_title}",
            f"Owner: {action.owner or 'Not supplied'}",
            f"Due date: {assessment.due_date or 'Not supplied'}",
            f"Priority: {assessment.priority.value}",
            f"Priority reasons: {'; '.join(assessment.priority_reasons)}",
            "Missing documents: "
            + (
                ", ".join(action.missing_documents)
                if action.missing_documents
                else "None listed"
            ),
            f"Dependency blocked: {action.dependency_blocked}",
            "Response status: "
            + (
                response.response_status.value
                if response.response_status
                else "Not supplied"
            ),
            f"Response summary: {response.response_summary or 'Not supplied'}",
            "Unresolved items: "
            + (
                ", ".join(response.unresolved_items)
                if response.unresolved_items
                else "Not supplied"
            ),
        ]
        return "\n".join(facts)

    @staticmethod
    def _parse_llm_draft(text: str) -> tuple[str, str]:
        """Extract subject and body from the prompted text format."""
        text = (text or "").strip()
        match = re.search(
            r"(?im)^\s*Subject:\s*(.+?)\s*$",
            text,
        )
        if not match:
            raise ValueError("LLM draft did not contain a Subject line.")

        subject = match.group(1).strip()
        body = text[match.end():].strip()
        body = re.sub(r"^\s*\n+", "", body)

        if not subject or not body:
            raise ValueError("LLM draft subject or body is empty.")

        return subject, body

    @staticmethod
    def _template_draft(
        request: ActionPrioritizationRequest,
        assessment: PriorityAssessment,
    ) -> tuple[str, str]:
        action = request.action
        response = request.response_context
        subject = f"Follow-up: {action.action_title}"

        lines = [
            "Dear Site Team,",
            "",
            f"I am following up regarding: {action.action_title}.",
            "",
        ]

        if action.missing_documents:
            lines.extend(
                [
                    "Our current records indicate that the following "
                    "items may be outstanding:",
                    *[f"- {item}" for item in action.missing_documents],
                    "",
                ]
            )
        else:
            lines.extend(
                [
                    "Please provide an update on the current status of "
                    "this action.",
                    "",
                ]
            )

        if assessment.due_date:
            lines.append(
                f"The recorded due date is {assessment.due_date.isoformat()}."
            )
        else:
            lines.append("Please confirm the expected completion date.")

        if response.response_status:
            lines.append(
                f"The recorded response status is "
                f"{response.response_status.value}."
            )

        if response.unresolved_items:
            lines.extend(
                [
                    "",
                    "Items identified for clarification:",
                    *[f"- {item}" for item in response.unresolved_items],
                ]
            )

        lines.extend(
            [
                "",
                "Could you please confirm the current status and advise "
                "on any remaining steps or dependencies?",
                "",
                "Thank you,",
                "Clinical Trial Operations Team",
                "",
                "Draft only — review and approve before sending.",
            ]
        )

        return subject, "\n".join(lines)

    def _draft_followup(
        self,
        request: ActionPrioritizationRequest,
        assessment: PriorityAssessment,
    ) -> FollowupDraft:
        draft_method = "template_fallback"
        subject: str
        body: str

        if self._draft_chain is not None:
            try:
                raw_text = self._draft_chain.invoke(
                    {"context": self._make_context(request, assessment)}
                )
                subject, body = self._parse_llm_draft(raw_text)
                draft_method = "llm"
            except Exception as exc:
                logger.warning(
                    "LLM follow-up drafting failed; using template. "
                    "error_type=%s",
                    type(exc).__name__,
                )
                subject, body = self._template_draft(request, assessment)
        else:
            subject, body = self._template_draft(request, assessment)

        # Validate all returned draft fields against the shared schema.
        return FollowupDraft.model_validate(
            {
                "subject": subject,
                "body": body,
                "draft_method": draft_method,
                "requires_human_approval": True,
                "external_email_sent": False,
            }
        )

    def run(
        self,
        request: ActionPrioritizationRequest | dict | None = None,
        response_context: Optional[dict] = None,
        as_of: Optional[date] = None,
        *,
        action: Optional[dict] = None,
    ) -> ActionPrioritizationResponse:
        """
        Run the agent.

        Preferred API:
            run(ActionPrioritizationRequest(...))

        Backward-compatible API:
            run(action_dict, response_context={...}, as_of=date(...))

        Returns a validated ActionPrioritizationResponse.
        """
        # Backward compatibility with the previous orchestrator call:
        # agent.run(action=..., response_context=..., as_of=...).
        if request is None:
            request = action
        elif action is not None:
            raise ValueError("Provide either 'request' or 'action', not both.")

        if request is None:
            raise ValueError("A request or action payload is required.")

        validated_request = self._as_request(
            request,
            response_context=response_context,
            as_of=as_of,
        )

        assessment = self._calculate_priority(validated_request)
        draft = self._draft_followup(validated_request, assessment)

        return ActionPrioritizationResponse.model_validate(
            {
                "priority_result": assessment,
                "followup_draft": draft,
                "human_review_required": True,
                "external_action_taken": False,
            }
        )