from contextlib import asynccontextmanager

from ..exceptions import RunwareOperationError


class ErrorContext:
    """Provides context for error handling."""

    def __init__(self, operation_id: str, operation_type: str):
        self.operation_id = operation_id
        self.operation_type = operation_type
        self.context_stack = []

    @asynccontextmanager
    async def phase(self, phase_name: str):
        """Context manager for operation phases."""
        self.context_stack.append(phase_name)
        try:
            yield
        except Exception as e:
            if not hasattr(e, "_runware_context"):
                e._runware_context = {
                    "operation_id": self.operation_id,
                    "operation_type": self.operation_type,
                    "phase": phase_name,
                    "stack": self.context_stack.copy(),
                }
            raise
        finally:
            if self.context_stack and self.context_stack[-1] == phase_name:
                self.context_stack.pop()

    def wrap_error(self, error: Exception) -> RunwareOperationError:
        """Wrap exception with full context."""
        context = {
            "operation_id": self.operation_id,
            "operation_type": self.operation_type,
            "phases": self.context_stack,
            "original_error": str(error),
            "error_type": type(error).__name__,
        }

        current_phase = (
            self.context_stack[-1] if self.context_stack else "unknown phase"
        )

        wrapped_error = RunwareOperationError(
            message=f"{self.operation_type} failed in {current_phase}: {error}",
            operation_id=self.operation_id,
            operation_type=self.operation_type,
        )

        # Store context in the error for debugging
        wrapped_error.details = context

        return wrapped_error
