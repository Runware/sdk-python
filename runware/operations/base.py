import asyncio
import time
import uuid
import weakref
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Awaitable, Callable, Dict, List, Optional

from ..core.error_context import ErrorContext
from ..core.types import OperationContext, OperationStatus, ProgressUpdate
from ..exceptions import RunwareOperationError, RunwareTimeoutError, RunwareParseError
from ..logging_config import get_logger
from ..types import ETaskType
from ..utils import instantiateDataclass

if TYPE_CHECKING:
    from ..client import RunwareClient

logger = get_logger(__name__)


class BaseOperation(ABC):
    """
    Abstract base class for all Runware operations.

    The execution flow follows these steps:
    1. Initialization and validation
    2. Request payload building
    3. Request sending
    4. Response waiting and processing
    5. Cleanup
    """

    # Class-level configuration that subclasses should override
    field_mappings: Dict[str, str] = {}
    response_class: Any = None
    status: OperationStatus

    def __init__(
        self, operation_id: Optional[str] = None, client: "RunwareClient" = None
    ):
        """
        Initialize the base operation.

        Args:
            operation_id: Unique identifier for this operation
            client: Reference to the Runware client
        """
        self.operation_id = operation_id or self._generate_operation_id()

        # Use weak reference to client to prevent circular references
        self._client_ref = weakref.ref(client) if client else None

        # Operation state
        self._initialize_state()

        # Logging
        self.logger = get_logger(f"{self.__class__.__name__}.{self.operation_id}")

        # Event handling
        self._initialize_events()

        # Message handling
        self._message_handlers: Dict[
            str, Callable[[Dict[str, Any]], Awaitable[None]]
        ] = {}
        self._setup_message_handlers()

        # Timeout management
        self._timeout_task: Optional[asyncio.Task] = None
        self._is_cancelled = False

        self.logger.debug(
            f"Operation {self.operation_id} created ({self.operation_type})"
        )

    def _generate_operation_id(self) -> str:
        """Generate a unique operation ID."""
        return str(uuid.uuid4())

    def _initialize_state(self):
        """Initialize operation state variables."""
        self.status = OperationStatus.PENDING
        self.created_at = time.time()
        self.completed_at: Optional[float] = None
        self.results: List[Any] = []
        self.error: Optional[Exception] = None
        self.progress: float = 0.0
        self.metadata: Dict[str, Any] = {}

    def _initialize_events(self):
        """Initialize event handling components."""
        self.completion_event = asyncio.Event()
        self.progress_callbacks: List[Callable[[ProgressUpdate], None]] = []
        self.completion_callbacks: List[Callable[["BaseOperation"], None]] = []

    @property
    def client(self) -> Optional["RunwareClient"]:
        """Get client from weak reference."""
        return self._client_ref() if self._client_ref else None

    @property
    @abstractmethod
    def operation_type(self) -> str:
        """Return the type of this operation. Must be implemented by subclasses."""
        pass

    async def execute(self) -> Any:
        """
        Main execution method implementing the Template Method pattern.

        This method defines the skeleton of the operation execution algorithm.
        Subclasses can override hook methods to customize specific steps.
        """
        error_ctx = ErrorContext(self.operation_id, self.operation_type)

        try:
            # Step 1: Pre-execution initialization
            async with error_ctx.phase("initialization"):
                await self._pre_execution_hook()
                await self._validate_operation()

            # Step 2: Build request payload
            async with error_ctx.phase("build_payload"):
                request_payload = await self._build_request_payload()
                await self._post_payload_build_hook(request_payload)

            # Step 3: Send request
            async with error_ctx.phase("send_request"):
                await self._send_request(request_payload)
                await self._post_request_send_hook()

            # Step 4: Wait for completion and process results
            async with error_ctx.phase("wait_completion"):
                results = await self._wait_for_results()
                processed_results = await self._process_results(results)

            # Step 5: Post-execution cleanup and finalization
            async with error_ctx.phase("finalization"):
                await self._post_execution_hook(processed_results)

            self.logger.info(f"Operation {self.operation_id} completed successfully")
            return processed_results

        except RunwareOperationError:
            raise
        except Exception as e:
            wrapped = error_ctx.wrap_error(e)
            await self._handle_error(wrapped)
            raise wrapped

    # Abstract methods that subclasses must implement
    @abstractmethod
    async def _build_request_payload(self) -> List[Dict[str, Any]]:
        """Build the request payload for this operation. Must be implemented by subclasses."""
        pass

    @abstractmethod
    def _setup_message_handlers(self):
        """Setup message handlers for this operation. Must be implemented by subclasses."""
        pass

    async def _pre_execution_hook(self):
        """Hook called before operation execution begins."""
        self.logger.info(f"Operation {self.operation_id} starting execution")

    async def _post_payload_build_hook(self, request_payload: List[Dict[str, Any]]):
        """Hook called after request payload is built."""
        self.logger.debug(f"Operation {self.operation_id} built request payload")

    async def _post_request_send_hook(self):
        """Hook called after request is sent."""
        self.logger.debug(f"Operation {self.operation_id} request sent")

    async def _post_execution_hook(self, results: Any):
        """Hook called after successful execution."""
        self.logger.debug(f"Operation {self.operation_id} execution completed")

    # Core operation methods
    async def _validate_operation(self):
        """Validate operation preconditions."""
        if not self.client:
            raise RunwareOperationError(
                "No client available for operation",
                operation_id=self.operation_id,
                operation_type=self.operation_type,
            )

        if not self.client.connection_manager:
            raise RunwareOperationError(
                "No connection manager available",
                operation_id=self.operation_id,
                operation_type=self.operation_type,
            )

    async def _send_request(self, request_payload: List[Dict[str, Any]]):
        """Send the request payload to the server."""
        if self.client and self.client.connection_manager:
            self.logger.debug(f"Operation {self.operation_id} sending request payload")
            await self.client.connection_manager.send_message(request_payload)
        else:
            raise RunwareOperationError(
                "No connection manager available",
                operation_id=self.operation_id,
                operation_type=self.operation_type,
            )

    async def _wait_for_results(self) -> Any:
        """Wait for operation completion and return results."""
        return await self.wait_for_completion()

    async def _process_results(self, results: Any) -> Any:
        """Process and validate results before returning."""
        return results

    # Operation lifecycle methods
    async def start(self, timeout: Optional[float] = None) -> Any:
        """
        Start the operation execution.

        Args:
            timeout: Optional timeout in seconds

        Returns:
            Operation results
        """
        if self.status != OperationStatus.PENDING:
            raise RunwareOperationError(
                f"Operation {self.operation_id} cannot be started in status {self.status.value}",
                operation_id=self.operation_id,
                operation_type=self.operation_type,
            )

        self.status = OperationStatus.EXECUTING
        self.logger.info(
            f"Starting operation {self.operation_id} ({self.operation_type})"
        )

        if timeout:
            self._timeout_task = asyncio.create_task(self._handle_timeout(timeout))

        try:
            return await self.execute()
        except Exception as e:
            self.logger.error(
                f"Operation {self.operation_id} execution failed", exc_info=e
            )
            await self._handle_error(e)
            raise
        finally:
            await self._cleanup()

    async def wait_for_completion(self, timeout: Optional[float] = None) -> Any:
        """
        Wait for the operation to complete.

        Args:
            timeout: Optional timeout in seconds

        Returns:
            Operation results
        """
        if self.status == OperationStatus.COMPLETED:
            return self.get_results()

        if self.status in [OperationStatus.FAILED, OperationStatus.TIMEOUT]:
            if self.error:
                raise self.error
            raise RunwareOperationError(
                f"Operation {self.operation_id} failed with status {self.status.value}",
                operation_id=self.operation_id,
                operation_type=self.operation_type,
            )

        try:
            await asyncio.wait_for(self.completion_event.wait(), timeout)
            return self.get_results()
        except asyncio.TimeoutError:
            await self.cancel()
            raise RunwareTimeoutError(
                f"Operation {self.operation_id} timed out after {timeout}s",
                timeout_duration=timeout,
            )

    async def cancel(self):
        """Cancel the operation."""
        if self.status in [
            OperationStatus.COMPLETED,
            OperationStatus.FAILED,
            OperationStatus.CANCELLED,
            OperationStatus.TIMEOUT,
        ]:
            return

        if self._is_cancelled:
            return

        self._is_cancelled = True
        self.status = OperationStatus.CANCELLED
        self.logger.info(f"Cancelling operation {self.operation_id}")

        self.completion_event.set()
        await self._cleanup()

    def get_results(self) -> Any:
        """Get operation results."""
        if (
            self.status in [OperationStatus.FAILED, OperationStatus.TIMEOUT]
            and self.error
        ):
            raise self.error
        return self.results

    def get_context(self) -> OperationContext:
        """Get operation context for monitoring."""
        return OperationContext(
            operation_id=self.operation_id,
            operation_type=self.operation_type,
            status=self.status,
            created_at=self.created_at,
            completed_at=self.completed_at,
            results=self.results,
            error=self.error,
            progress=self.progress,
            metadata=self.metadata,
        )

    # Event handling methods
    def add_progress_callback(self, callback: Callable[[ProgressUpdate], None]):
        """Add a progress callback."""
        self.progress_callbacks.append(callback)

    def add_completion_callback(self, callback: Callable[["BaseOperation"], None]):
        """Add a completion callback."""
        self.completion_callbacks.append(callback)

    # Message handling methods
    async def handle_message(self, message: Dict[str, Any]):
        """
        Handle incoming messages from the server.

        Args:
            message: Message from the server
        """
        # Check if operation is already completed or cancelled
        if self.status in [
            OperationStatus.COMPLETED,
            OperationStatus.CANCELLED,
            OperationStatus.FAILED,
        ]:
            return

        try:
            message_type = message.get("taskType")
            handler = self._message_handlers.get(message_type)

            if handler:
                await handler(message)
            else:
                await self._handle_unknown_message(message)

        except Exception as e:
            self.logger.error(
                f"Error handling message for operation {self.operation_id}", exc_info=e
            )
            await self._handle_error(e)

    async def _handle_unknown_message(self, message: Dict[str, Any]):
        """Handle unknown message types."""
        message_type = message.get("taskType", "unknown")
        self.logger.warning(
            f"Unknown message type '{message_type}' for operation {self.operation_id}"
        )

    # Error handling and completion methods
    async def _handle_error(self, error: Exception):
        """Handle operation errors."""
        if self.status in [
            OperationStatus.COMPLETED,
            OperationStatus.CANCELLED,
            OperationStatus.FAILED,
            OperationStatus.TIMEOUT,
        ]:
            return

        self.logger.error(
            f"Operation {self.operation_id} encountered error: {str(error)}",
            exc_info=error,
        )
        self.error = error

        # Set appropriate status based on error type
        if isinstance(error, RunwareTimeoutError):
            self.status = OperationStatus.TIMEOUT
        else:
            self.status = OperationStatus.FAILED

        self.completed_at = time.time()
        self.completion_event.set()
        await self._notify_completion()

    async def _handle_timeout(self, timeout: float):
        """Handle operation timeout."""
        try:
            await asyncio.sleep(timeout)
            if self.status == OperationStatus.EXECUTING:
                self.logger.error(
                    f"Operation {self.operation_id} timed out after {timeout}s"
                )
                await self._handle_error(
                    RunwareTimeoutError(
                        f"Operation {self.operation_id} timed out after {timeout}s",
                        timeout_duration=timeout,
                    )
                )
        except asyncio.CancelledError:
            pass

    async def _complete_operation(self, results: Any = None):
        """Complete the operation successfully."""
        if self.status in [
            OperationStatus.COMPLETED,
            OperationStatus.CANCELLED,
            OperationStatus.FAILED,
            OperationStatus.TIMEOUT,
        ]:
            return

        if results is not None:
            if isinstance(results, list):
                self.results.extend(results)
            else:
                self.results.append(results)

        self.logger.info(
            f"Operation {self.operation_id} completed with {len(self.results)} results"
        )
        self.status = OperationStatus.COMPLETED
        self.completed_at = time.time()
        self.progress = 1.0

        self.completion_event.set()
        await self._notify_progress()
        await self._notify_completion()

    # Progress and notification methods
    async def _update_progress(
        self,
        progress: float,
        message: Optional[str] = None,
        partial_results: Optional[List[Any]] = None,
    ):
        """Update operation progress."""
        if self.status in [
            OperationStatus.COMPLETED,
            OperationStatus.CANCELLED,
            OperationStatus.FAILED,
        ]:
            return

        self.progress = max(0.0, min(1.0, progress))

        if partial_results:
            self.results.extend(partial_results)

        await self._notify_progress(message, partial_results)

    async def _notify_progress(
        self, message: Optional[str] = None, partial_results: Optional[List[Any]] = None
    ):
        """Notify progress callbacks."""
        if not self.progress_callbacks:
            return

        progress_update = ProgressUpdate(
            operation_id=self.operation_id,
            progress=self.progress,
            message=message,
            partial_results=partial_results,
        )

        for callback in self.progress_callbacks:
            try:
                callback(progress_update)
            except Exception as e:
                self.logger.error(
                    f"Error in progress callback for operation {self.operation_id}",
                    exc_info=e,
                )

    async def _notify_completion(self):
        """Notify completion callbacks."""
        for callback in self.completion_callbacks:
            try:
                callback(self)
            except Exception as e:
                self.logger.error(
                    f"Error in completion callback for operation {self.operation_id}",
                    exc_info=e,
                )

    # Response parsing methods
    def _parse_response(self, message: Dict[str, Any]) -> Any:
        """
        Parse server response into expected data structure.

        This method uses the field_mappings and response_class defined by subclasses
        to convert server messages into properly typed response objects.
        """
        if not self.field_mappings or not self.response_class:
            raise RunwareParseError(
                f"Operation missing field_mappings or response_class",
                operation_id=self.operation_id,
                operation_type=self.operation_type,
                raw_data=message,
            )

        try:
            processed_fields = {}

            for field_name, message_key in self.field_mappings.items():
                if message_key in message:
                    value = message[message_key]

                    if field_name == "taskType" and hasattr(ETaskType, "value"):
                        # Only convert to enum if the response class expects an enum
                        if hasattr(self.response_class, "__annotations__"):
                            field_type = self.response_class.__annotations__.get(
                                field_name
                            )
                            if field_type == ETaskType:
                                processed_fields[field_name] = ETaskType(value)
                            else:
                                processed_fields[field_name] = value
                        else:
                            processed_fields[field_name] = value
                    elif field_name == "cost" and value is not None:
                        processed_fields[field_name] = float(value)
                    else:
                        processed_fields[field_name] = value

            return instantiateDataclass(self.response_class, processed_fields)

        except Exception as e:
            self.logger.error(
                f"Operation {self.operation_id} failed to parse response",
                exc_info=e,
            )
            raise RunwareParseError(
                f"Failed to parse response: {e}",
                operation_id=self.operation_id,
                operation_type=self.operation_type,
                raw_data=message,
            )

    async def _handle_error_message(self, message: Dict[str, Any]):
        error_message = message.get("message", "Unknown error")

        logger.error(f"Operation {self.operation_id} received error: {error_message}")

        error = RunwareOperationError(
            f"{self.__class__.__name__} error: {error_message}",
            operation_id=self.operation_id,
            operation_type=self.operation_type,
            code=message.get("code"),
            parameter=message.get("parameter"),
            error_type=message.get("type"),
            documentation=message.get("documentation"),
            task_uuid=message.get("taskUUID", self.operation_id),
        )

        await self._handle_error(error)

    # Cleanup methods
    async def _cleanup(self):
        """Cleanup operation resources."""
        if self._timeout_task and not self._timeout_task.done():
            self._timeout_task.cancel()
            try:
                await asyncio.wait_for(self._timeout_task, timeout=1.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._timeout_task = None

        # Clear callbacks to prevent memory leaks
        self.progress_callbacks.clear()
        self.completion_callbacks.clear()

    # String representation methods
    def __str__(self):
        return f"{self.__class__.__name__}({self.operation_id}, {self.status.value})"

    def __repr__(self):
        return self.__str__()
