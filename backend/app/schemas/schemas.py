
from __future__ import annotations

from datetime import date, datetime, timezone
from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    model_validator,
)


# ---------------------------------------------------------
# Common enums
# ---------------------------------------------------------

class PriorityLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class FindingSeverity(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class ResponseStatus(str, Enum):
    ANSWERED = "ANSWERED"
    PARTIALLY_ANSWERED = "PARTIALLY_ANSWERED"
    PENDING = "PENDING"
    CLARIFICATION_REQUIRED = "CLARIFICATION_REQUIRED"
    ESCALATION_REQUIRED = "ESCALATION_REQUIRED"
    UNCLASSIFIED = "UNCLASSIFIED"


class WorkflowStatus(str, Enum):
    SUCCESS = "SUCCESS"
    PARTIAL_SUCCESS = "PARTIAL_SUCCESS"
    FAILED = "FAILED"


# ---------------------------------------------------------
# Base schema
# ---------------------------------------------------------

class Schema(BaseModel):
    model_config = ConfigDict(
        extra="forbid",
        str_strip_whitespace=True,
        validate_assignment=True,
    )


# ---------------------------------------------------------
# 1. Document Intelligence Agent
# ---------------------------------------------------------

class DocumentRecord(Schema):
    document_id: str | None = None
    document_type: str = Field(min_length=1)
    status: str = "received"

    issue_date: date | None = None
    expiry_date: date | None = None

    version: str | None = None
    source_ref: str | None = None


class DocumentIntelligenceRequest(Schema):
    study_id: str = Field(min_length=1)
    site_id: str = Field(min_length=1)

    documents: list[DocumentRecord] = Field(default_factory=list)
    required_document_types: list[str] = Field(
        default_factory=list
    )
    as_of: date = Field(
        default_factory=date.today
    )


class DocumentFinding(Schema):
    document_id: str | None = None
    document_type: str
    finding_code: str
    severity: FindingSeverity
    message: str
    recommendation: str
    source_ref: str | None = None


class DocumentIntelligenceResponse(Schema):
    agent_name: str = "Document_Intelligence_Agent"
    study_id: str
    site_id: str

    status: WorkflowStatus = WorkflowStatus.SUCCESS
    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    documents_examined: int = Field(ge=0)
    findings_count: int = Field(ge=0)
    severity_counts: dict[str, int] = Field(
        default_factory=dict
    )
    findings: list[DocumentFinding] = Field(
        default_factory=list
    )

    requires_human_review: bool = True
    activation_decision: str = "NOT_EVALUATED"


# ---------------------------------------------------------
# 2. Site Response Intelligence Agent
# ---------------------------------------------------------

class SiteResponseRequest(Schema):
    study_id: str
    site_id: str

    response_id: str = Field(min_length=1)
    response_text: str = Field(min_length=1)

    action_id: str | None = None
    action_description: str = ""
    source_ref: str | None = None

    received_at: datetime | None = None


class SiteResponseAnalysis(Schema):
    agent_name: str = "Site_Response_Intelligence_Agent"

    study_id: str
    site_id: str
    response_id: str

    action_id: str | None = None
    source_ref: str | None = None

    status: ResponseStatus
    classification_method: str
    classification_reasons: list[str] = Field(
        default_factory=list
    )
    unresolved_items: list[str] = Field(
        default_factory=list
    )

    follow_up_required: bool
    suggested_next_action: str

    requires_human_review: bool = True
    action_closed: bool = False


# ---------------------------------------------------------
# 3. Action Prioritization & Follow-up Agent
# ---------------------------------------------------------

class ActivationAction(Schema):
    action_id: str = Field(min_length=1)
    study_id: str
    site_id: str

    action_title: str = Field(min_length=1)
    owner: str | None = None

    due_date: date | None = None
    criticality: PriorityLevel = PriorityLevel.MEDIUM

    missing_documents: list[str] = Field(
        default_factory=list
    )
    dependency_blocked: bool = False

    source_ref: str | None = None


class FollowupContext(Schema):
    response_status: ResponseStatus | None = None
    response_summary: str | None = None
    unresolved_items: list[str] = Field(
        default_factory=list
    )


class ActionPrioritizationRequest(Schema):
    action: ActivationAction
    response_context: FollowupContext = Field(
        default_factory=FollowupContext
    )
    as_of: date = Field(default_factory=date.today)


class PriorityAssessment(Schema):
    action_id: str
    study_id: str
    site_id: str

    priority_score: int = Field(ge=0, le=100)
    priority: PriorityLevel
    priority_reasons: list[str] = Field(
        default_factory=list
    )

    days_overdue: int = Field(ge=0)
    due_date: date | None = None
    scoring_version: str = "poc-v1"


class FollowupDraft(Schema):
    subject: str
    body: str
    draft_method: str

    requires_human_approval: bool = True
    external_email_sent: bool = False


class ActionPrioritizationResponse(Schema):
    agent_name: str = "Action_Prioritization_Followup_Agent"

    priority_result: PriorityAssessment
    followup_draft: FollowupDraft

    human_review_required: bool = True
    external_action_taken: bool = False


# ---------------------------------------------------------
# 4. Orchestrator request and response
# ---------------------------------------------------------

class SiteActivationRequest(Schema):
    study_id: str = Field(min_length=1)
    site_id: str = Field(min_length=1)

    documents: list[DocumentRecord] = Field(
        default_factory=list
    )
    responses: list[SiteResponseRequest] = Field(
        default_factory=list
    )
    actions: list[ActivationAction] = Field(
        default_factory=list
    )

    as_of: date = Field(default_factory=date.today)

    @model_validator(mode="after")
    def validate_child_identifiers(self):
        for document_response in self.responses:
            if document_response.study_id != self.study_id:
                raise ValueError(
                    "Response study_id must match request study_id."
                )
            if document_response.site_id != self.site_id:
                raise ValueError(
                    "Response site_id must match request site_id."
                )

        for action in self.actions:
            if action.study_id != self.study_id:
                raise ValueError(
                    "Action study_id must match request study_id."
                )
            if action.site_id != self.site_id:
                raise ValueError(
                    "Action site_id must match request site_id."
                )

        return self


class AgentError(Schema):
    component: str
    error_code: str


class WorkflowSummary(Schema):
    documents_examined: int = 0
    document_findings: int = 0
    responses_analyzed: int = 0
    actions_processed: int = 0
    actions_requiring_review: int = 0


class SiteActivationResponse(Schema):
    workflow_id: str
    workflow_name: str = (
        "Clinical Trial Site Activation Intelligence"
    )
    workflow_status: WorkflowStatus

    study_id: str
    site_id: str

    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )

    document_analysis: DocumentIntelligenceResponse | None = None
    response_analysis: list[SiteResponseAnalysis] = Field(
        default_factory=list
    )
    action_analysis: list[ActionPrioritizationResponse] = Field(
        default_factory=list
    )

    errors: list[AgentError] = Field(default_factory=list)
    summary: WorkflowSummary = Field(
        default_factory=WorkflowSummary
    )
    execution_time_seconds: float = Field(ge=0)

    human_review_required: bool = True
    activation_decision: str = "NOT_EVALUATED"
    external_emails_sent: bool = False
