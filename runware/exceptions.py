from typing import Any, Dict, Optional


class RunwareError(Exception):
    """Base exception for all Runware SDK errors."""

    def __init__(
        self,
        message: str,
        code: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        parameter: Optional[str] = None,
        error_type: Optional[str] = None,
        documentation: Optional[str] = None,
        task_uuid: Optional[str] = None,
    ):
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
        self.parameter = parameter
        self.error_type = error_type
        self.documentation = documentation
        self.task_uuid = task_uuid

    def format_error(self):
        """Format error for backward compatibility."""
        return {
            "errors": [
                {
                    "code": self.code,
                    "message": self.message,
                    "parameter": self.parameter,
                    "type": self.error_type,
                    "documentation": self.documentation,
                    "taskUUID": self.task_uuid,
                }
            ]
        }

    def __str__(self):
        return str(self.format_error())


class RunwareConnectionError(RunwareError):
    """Raised when there are connection-related issues."""

    def __init__(self, message: str, connection_state: Optional[str] = None, **kwargs):
        super().__init__(message, **kwargs)
        self.connection_state = connection_state


class RunwareAuthenticationError(RunwareError):
    """Raised when authentication fails."""

    def __init__(self, message: str = "Authentication failed", **kwargs):
        super().__init__(message, **kwargs)


class RunwareOperationError(RunwareError):
    """Raised when an operation fails."""

    def __init__(
        self,
        message: str,
        operation_id: str,
        operation_type: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(message, **kwargs)
        self.operation_id = operation_id
        self.operation_type = operation_type


class RunwareTimeoutError(RunwareError):
    """Raised when an operation times out."""

    def __init__(
        self, message: str, timeout_duration: Optional[float] = None, **kwargs
    ):
        super().__init__(message, **kwargs)
        self.timeout_duration = timeout_duration


class RunwareParseError(RunwareOperationError):
    """Raised when response parsing fails."""

    def __init__(
        self,
        message: str,
        operation_id: str = "unknown",
        operation_type: str = "unknown",
        raw_data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(message, operation_id, operation_type, **kwargs)
        self.raw_data = raw_data or {}


class RunwareAPIError(Exception):
    """API error for backward compatibility with old SDK."""

    def __init__(self, error_data: Dict[str, Any]):
        self.error_data = error_data
        self.code = error_data.get("code")
        super().__init__(str(error_data))

    def __str__(self):
        return f"RunwareAPIError: {self.error_data}"


class RunwareValidationError(RunwareOperationError):
    """Raised when validation fails."""

    pass


class RunwareResourceError(RunwareOperationError):
    """Raised when resource constraints are hit."""

    pass


class RunwareServerError(RunwareOperationError):
    """Raised when server returns an error."""

    pass
