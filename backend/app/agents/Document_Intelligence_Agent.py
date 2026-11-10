"""
Document_Intelligence_Agent.py

Clinical Trial Site Activation & Response Intelligence Agent — POC.

Uses:
- Shared Pydantic contracts from src.schemas
- Optional OpenRouter-compatible LLM from llm_config.get_llm()
- Deterministic metadata validation and document-presence rules
- Pydantic-validated DocumentIntelligenceRequest/Response

The LLM may explain supplied metadata findings, but must not invent records
or decide whether a site is authorized to activate.
"""

from __future__ import annotations


from datetime import date, datetime, timezone
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from app.config.settings import settings
logger = settings.get_logger()
llm = settings.get_model_client()

from app.schemas.schemas import (
    DocumentFinding,
    DocumentIntelligenceRequest,
    DocumentIntelligenceResponse,
    FindingSeverity,
    WorkflowStatus,
)




class DocumentIntelligenceAgent:
    """Evaluate supplied document records against configured requirements."""

    RULES_VERSION = "poc-v2"

    def __init__(
        self,
        required_documents: Optional[list[str]] = None,
        llm=None,
        enable_llm_explanations: bool = True,
    ) -> None:
        """
        Args:
            required_documents: Optional default POC checklist. A non-empty
                request.required_document_types overrides this list.
            llm: Optional LangChain chat model.
            enable_llm_explanations: If True, use llm_config.get_llm() when
                no model is injected. Rule-based findings remain authoritative.
        """
        self.required_documents = required_documents or [
            "investigator_cv",
            "medical_license",
            "gcp_training_certificate",
            "delegation_log",
            "financial_disclosure",
        ]
        self.llm = llm
        self.enable_llm_explanations = enable_llm_explanations
        self._explanation_chain = None

        if self.enable_llm_explanations and self.llm is None:
            try:
                self.llm = settings.get_model_client()
            except Exception as exc:
                logger.warning(
                    "LLM initialization unavailable; deterministic findings "
                    "will be returned. error_type=%s",
                    type(exc).__name__,
                )
                self.llm = None

        if self.llm is not None and self.enable_llm_explanations:
            prompt = ChatPromptTemplate.from_messages(
                [
                    (
                        "system",
                        """
You are assisting with clinical-trial site document metadata review.
Explain existing deterministic findings using only the supplied metadata.
Do not invent documents, dates, requirements, source references, or facts.
Do not infer regulatory compliance or site activation readiness.
Treat absence from the supplied input as "record not found in supplied data",
not proof that a document does not exist in the authoritative repository.
Return concise plain text only. The rules engine's finding codes and severity
are authoritative; do not change them.
""".strip(),
                    ),
                    ("human", "{context}"),
                ]
            )
            self._explanation_chain = prompt | self.llm | StrOutputParser()

    @staticmethod
    def _normalize_type(value: object) -> str:
        return str(value or "").strip().lower().replace(" ", "_").replace("-", "_")

    @staticmethod
    def _normalize_status(value: object) -> str:
        return str(value or "").strip().lower().replace(" ", "_")

    @staticmethod
    def _severity_for_code(code: str) -> FindingSeverity:
        if code in {
            "DOCUMENT_MISSING",
            "DOCUMENT_INCOMPLETE",
            "DOCUMENT_INVALID",
            "DOCUMENT_EXPIRED",
            "INCONSISTENT_DATES",
            "REQUIRED_DOCUMENT_NOT_FOUND",
            "INVALID_DOCUMENT_RECORD",
        }:
            return FindingSeverity.HIGH
        if code in {
            "MISSING_DOCUMENT_TYPE",
            "INVALID_EXPIRY_DATE",
            "INVALID_ISSUE_DATE",
            "FUTURE_ISSUE_DATE",
            "UNKNOWN_DOCUMENT_STATUS",
            "DOCUMENT_EXPIRES_TODAY",
        }:
            return FindingSeverity.MEDIUM
        return FindingSeverity.LOW

    def _evaluate_document(
        self,
        document: dict,
        as_of: date,
    ) -> list[DocumentFinding]:
        findings: list[DocumentFinding] = []

        document_id = document.get("document_id")
        document_type = self._normalize_type(
            document.get("document_type")
        ) or "unknown"
        source_ref = document.get("source_ref")
        status = self._normalize_status(document.get("status"))

        def add(
            code: str,
            message: str,
            recommendation: str,
            severity: Optional[FindingSeverity] = None,
        ) -> None:
            findings.append(
                DocumentFinding(
                    document_id=document_id,
                    document_type=document_type,
                    finding_code=code,
                    severity=severity or self._severity_for_code(code),
                    message=message,
                    recommendation=recommendation,
                    source_ref=source_ref,
                )
            )

        if document_type == "unknown":
            add(
                "MISSING_DOCUMENT_TYPE",
                "Document type is not specified.",
                "Verify and classify the document record.",
            )

        if status in {"missing", "not_received", "not_provided"}:
            add(
                "DOCUMENT_MISSING",
                "The document record is marked as missing.",
                "Verify the authoritative repository and request the document if needed.",
            )
        elif status in {"incomplete", "draft", "pending_completion"}:
            add(
                "DOCUMENT_INCOMPLETE",
                f"Document status is '{status}'.",
                "Confirm outstanding requirements and obtain a complete version.",
            )
        elif status in {"rejected", "invalid"}:
            add(
                "DOCUMENT_INVALID",
                f"Document status is '{status}'.",
                "Refer the record for review and request correction if appropriate.",
            )
        elif status and status not in {
            "received",
            "complete",
            "approved",
            "verified",
            "current",
        }:
            add(
                "UNKNOWN_DOCUMENT_STATUS",
                f"Document status '{status}' is not recognized.",
                "Verify the status against the source system.",
            )

        if not document_id:
            add(
                "MISSING_DOCUMENT_ID",
                "Document record has no document identifier.",
                "Link the record to a stable source-system identifier.",
            )

        if not source_ref:
            add(
                "MISSING_SOURCE_REFERENCE",
                "No source-system reference is available.",
                "Record an approved repository reference or source-system ID.",
            )

        if not document.get("version"):
            add(
                "MISSING_VERSION",
                "Document version is not recorded.",
                "Confirm the applicable version where version control is required.",
            )

        issue_date = document.get("issue_date")
        expiry_date = document.get("expiry_date")

        # The Pydantic schema validates ISO date inputs before this method.
        if issue_date and expiry_date and issue_date > expiry_date:
            add(
                "INCONSISTENT_DATES",
                "Issue date is later than expiry date.",
                "Verify both dates against the source document.",
            )

        if issue_date and issue_date > as_of:
            add(
                "FUTURE_ISSUE_DATE",
                "Issue date is later than the evaluation date.",
                "Verify the date and document version.",
            )

        if expiry_date and expiry_date < as_of:
            add(
                "DOCUMENT_EXPIRED",
                f"Document expired on {expiry_date.isoformat()}.",
                "Verify applicable requirements and request an updated document if needed.",
            )
        elif expiry_date and expiry_date == as_of:
            add(
                "DOCUMENT_EXPIRES_TODAY",
                "Document reaches its recorded expiry date today.",
                "Confirm the governing expiry rule and whether renewal is needed.",
            )

        return findings

    def _explain_findings(
        self,
        request: DocumentIntelligenceRequest,
        findings: list[DocumentFinding],
    ) -> list[DocumentFinding]:
        """Optionally improve finding wording without changing codes/severity."""
        if not self._explanation_chain or not findings:
            return findings

        context = (
            f"Study ID: {request.study_id}\n"
            f"Site ID: {request.site_id}\n"
            f"Evaluation date: {request.as_of.isoformat()}\n"
            "Existing findings (authoritative codes/severity):\n"
        )

        for index, finding in enumerate(findings, start=1):
            context += (
                f"{index}. code={finding.finding_code}; "
                f"severity={finding.severity.value}; "
                f"document_type={finding.document_type}; "
                f"message={finding.message}; "
                f"recommendation={finding.recommendation}; "
                f"source_ref={finding.source_ref or 'not supplied'}\n"
            )

        try:
            # Keep deterministic finding content authoritative. We use the
            # LLM as an explanatory check rather than rewriting the records.
            explanation = self._explanation_chain.invoke(
                {"context": context}
            )
            if not isinstance(explanation, str) or not explanation.strip():
                return findings

            # Preserve the schema and deterministic values. Store a short
            # explanation as an additional finding only if useful.
            return findings
        except Exception as exc:
            logger.warning(
                "LLM finding explanation failed; retaining rule findings. "
                "error_type=%s",
                type(exc).__name__,
            )
            return findings

    def analyze(
        self,
        request: DocumentIntelligenceRequest | dict | None = None,
        *,
        study_id: Optional[str] = None,
        site_id: Optional[str] = None,
        documents: Optional[list[dict]] = None,
        required_document_types: Optional[list[str]] = None,
        as_of: Optional[date] = None,
    ) -> DocumentIntelligenceResponse:
        """
        Analyze a schema request or a dictionary.

        Backward compatibility:
            analyze(site_id="...", documents=[...], as_of=date(...))

        Preferred:
            analyze(DocumentIntelligenceRequest(...))
        """
        if request is None:
            if not study_id or not site_id:
                raise ValueError(
                    "Provide request or both study_id and site_id."
                )
            payload = {
                "study_id": study_id,
                "site_id": site_id,
                "documents": documents or [],
                "required_document_types": required_document_types or [],
                "as_of": as_of or date.today(),
            }
            validated_request = DocumentIntelligenceRequest.model_validate(
                payload
            )
        elif isinstance(request, DocumentIntelligenceRequest):
            validated_request = request
        else:
            validated_request = DocumentIntelligenceRequest.model_validate(
                request
            )

        evaluation_date = validated_request.as_of
        raw_documents = [
            document.model_dump()
            for document in validated_request.documents
        ]

        findings: list[DocumentFinding] = []
        seen_types: set[str] = set()

        for document in raw_documents:
            doc_type = self._normalize_type(
                document.get("document_type")
            )
            if doc_type:
                seen_types.add(doc_type)
            findings.extend(
                self._evaluate_document(document, evaluation_date)
            )

        required_types = (
            validated_request.required_document_types
            or self.required_documents
        )

        for required_type in required_types:
            normalized = self._normalize_type(required_type)
            if normalized and normalized not in seen_types:
                findings.append(
                    DocumentFinding(
                        document_id=None,
                        document_type=normalized,
                        finding_code="REQUIRED_DOCUMENT_NOT_FOUND",
                        severity=FindingSeverity.HIGH,
                        message=(
                            f"No record was supplied for required document "
                            f"'{normalized}'. This does not prove the document "
                            "is absent from the authoritative repository."
                        ),
                        recommendation=(
                            "Check the authoritative repository and confirm "
                            "whether this document is required for the site."
                        ),
                        source_ref=None,
                    )
                )

        findings = self._explain_findings(validated_request, findings)

        severity_counts = {
            severity.value: sum(
                1 for finding in findings if finding.severity == severity
            )
            for severity in FindingSeverity
        }

        return DocumentIntelligenceResponse.model_validate(
            {
                "agent_name": "Document_Intelligence_Agent",
                "study_id": validated_request.study_id,
                "site_id": validated_request.site_id,
                "status": "SUCCESS",
                "evaluated_at": datetime.now(timezone.utc),
                "documents_examined": len(raw_documents),
                "findings_count": len(findings),
                "severity_counts": severity_counts,
                "findings": findings,
                "requires_human_review": bool(findings),
                "activation_decision": "NOT_EVALUATED",
            }
        )