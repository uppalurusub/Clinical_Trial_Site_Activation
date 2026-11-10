"""
Site activation service layer.

Purpose:
    Validate incoming input, call SiteActivationImplementation, and return a
    schema-validated SiteActivationResponse.

Expected project layout:
    src/
      schemas.py
      Site_Activation_Implementation.py
      Site_Activation_Service.py

This POC uses synthetic data and does not activate sites, send emails, or
replace the CTMS as the system of record.
"""

from __future__ import annotations

from typing import Any

from pydantic import ValidationError

from app.schemas.schemas import SiteActivationRequest, SiteActivationResponse
from app.implementations.Site_Activation_Implementation import SiteActivationImplementation


class SiteActivationService:
    """Service boundary between the API router and implementation layer."""

    def __init__(
        self,
        implementation: SiteActivationImplementation | None = None,
    ) -> None:
        self.implementation = implementation or SiteActivationImplementation()

    @staticmethod
    def _to_plain(value: Any) -> Any:
        """Convert Pydantic models and nested containers to plain Python data."""
        if hasattr(value, "model_dump"):
            return SiteActivationService._to_plain(value.model_dump(mode="python"))
        if isinstance(value, dict):
            return {key: SiteActivationService._to_plain(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [SiteActivationService._to_plain(item) for item in value]
        return value

    def analyze_site(
        self,
        request: SiteActivationRequest | dict[str, Any],
    ) -> SiteActivationResponse:
        """
        Analyze a site's activation readiness information.

        Args:
            request: A SiteActivationRequest or a compatible dictionary.

        Returns:
            Validated SiteActivationResponse from the implementation layer.

        Raises:
            pydantic.ValidationError: If request or response data fails validation.
            Exception: Propagates orchestration/implementation failures for the
                router to translate into an appropriate HTTP response.
        """
        validated_request = (
            request
            if isinstance(request, SiteActivationRequest)
            else SiteActivationRequest.model_validate(request)
        )

        # Delegate all workflow execution to Site_Activation_Implementation.py.
        implementation_result = self.implementation.run(validated_request)

        # Revalidate at the service boundary so callers always receive the
        # declared response type, even if an injected implementation returns a dict.
        plain_result = self._to_plain(implementation_result)
        return SiteActivationResponse.model_validate(plain_result)


# Optional compatibility alias for code using the module-style class name.
Site_Activation_Service = SiteActivationService


