import asyncio
import time
from typing import Any, Callable, Dict, List, Optional

from .base import BaseOperation
from ..core.types import OperationContext, OperationStatus
from ..exceptions import RunwareOperationError, RunwareResourceError
from ..logging_config import get_logger

logger = get_logger(__name__)


class OperationManager:
    """Manages operation execution and lifecycle."""

    def __init__(
        self, max_concurrent_operations: int = 100, operation_timeout: float = 300.0
    ):
        self.operations: Dict[str, BaseOperation] = {}
        self.max_concurrent_operations = max_concurrent_operations
        self.default_operation_timeout = operation_timeout

        self._completion_callbacks: List[Callable[[BaseOperation], None]] = []
        self._cleanup_task: Optional[asyncio.Task] = None
        self._is_running = False
        self._operation_semaphore = asyncio.Semaphore(max_concurrent_operations)

        logger.debug(
            f"OperationManager initialized with max_concurrent_operations={max_concurrent_operations}"
        )

    async def start(self):
        if self._is_running:
            return

        self._is_running = True
        self._cleanup_task = asyncio.create_task(self._periodic_cleanup())
        logger.info(
            f"Operation manager started (max concurrent: {self.max_concurrent_operations})"
        )

    async def stop(self):
        if not self._is_running:
            return

        self._is_running = False

        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await asyncio.wait_for(self._cleanup_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self._cleanup_task = None

        cancelled_count = await self._cancel_all_operations()
        logger.info(
            f"Operation manager stopped, cancelled {cancelled_count} operations"
        )

    async def register_operation(self, operation: BaseOperation) -> BaseOperation:
        if len(self.operations) >= self.max_concurrent_operations:
            raise RunwareResourceError(
                f"Maximum concurrent operations limit reached ({self.max_concurrent_operations})",
                resource_type="operation_slots",
            )

        if operation.operation_id in self.operations:
            raise RunwareOperationError(
                f"Operation {operation.operation_id} already registered",
                operation_id=operation.operation_id,
                operation_type=operation.operation_type,
            )

        self.operations[operation.operation_id] = operation
        operation.add_completion_callback(self._on_operation_completed)

        logger.debug(
            f"Registered operation {operation.operation_id} ({operation.operation_type})"
        )
        return operation

    async def unregister_operation(
        self, operation_id: str
    ) -> Optional[OperationContext]:
        operation = self.operations.pop(operation_id, None)
        if not operation:
            return None

        context = operation.get_context()
        logger.debug(
            f"Unregistered operation {operation_id}, status: {context.status.value}"
        )
        return context

    async def execute_operation(
        self, operation: BaseOperation, timeout: Optional[float] = None
    ) -> Any:
        operation_timeout = timeout or self.default_operation_timeout
        logger.info(
            f"Executing operation {operation.operation_id} ({operation.operation_type}) with timeout {operation_timeout}s"
        )

        async with self._operation_semaphore:
            await self.register_operation(operation)

            try:
                start_time = time.time()
                result = await operation.start(operation_timeout)
                execution_time = time.time() - start_time
                logger.info(
                    f"Operation {operation.operation_id} completed successfully in {execution_time:.2f}s"
                )
                return result
            except Exception as e:
                logger.error(f"Operation {operation.operation_id} failed", exc_info=e)
                raise
            finally:
                await self.unregister_operation(operation.operation_id)

    def get_operation(self, operation_id: str) -> Optional[BaseOperation]:
        return self.operations.get(operation_id)

    def get_operation_context(self, operation_id: str) -> Optional[OperationContext]:
        operation = self.operations.get(operation_id)
        if operation:
            return operation.get_context()
        return None

    def list_operations(
        self,
        status_filter: Optional[OperationStatus] = None,
        operation_type_filter: Optional[str] = None,
    ) -> List[OperationContext]:
        contexts = []

        for operation in self.operations.values():
            context = operation.get_context()

            if status_filter and context.status != status_filter:
                continue
            if (
                operation_type_filter
                and context.operation_type != operation_type_filter
            ):
                continue

            contexts.append(context)

        return contexts

    async def cancel_operation(self, operation_id: str) -> bool:
        operation = self.operations.get(operation_id)
        if not operation:
            return False

        logger.info(f"Cancelling operation {operation_id}")
        await operation.cancel()
        return True

    async def cancel_operations_by_type(self, operation_type: str) -> int:
        operations_to_cancel = [
            op for op in self.operations.values() if op.operation_type == operation_type
        ]

        logger.info(
            f"Cancelling {len(operations_to_cancel)} operations of type {operation_type}"
        )

        for operation in operations_to_cancel:
            try:
                await operation.cancel()
            except Exception as e:
                logger.error(
                    f"Error cancelling operation {operation.operation_id}", exc_info=e
                )

        return len(operations_to_cancel)

    async def cancel_all_operations(self) -> int:
        return await self._cancel_all_operations()

    def add_completion_callback(self, callback: Callable[[BaseOperation], None]):
        self._completion_callbacks.append(callback)

    def remove_completion_callback(self, callback: Callable[[BaseOperation], None]):
        if callback in self._completion_callbacks:
            self._completion_callbacks.remove(callback)

    async def wait_for_all_operations(self, timeout: Optional[float] = None) -> bool:
        if not self.operations:
            return True

        operations = list(self.operations.values())
        logger.info(f"Waiting for {len(operations)} operations to complete")
        tasks = [op.wait_for_completion() for op in operations]

        try:
            if timeout:
                await asyncio.wait_for(
                    asyncio.gather(*tasks, return_exceptions=True), timeout
                )
            else:
                await asyncio.gather(*tasks, return_exceptions=True)
            logger.info("All operations completed")
            return True
        except asyncio.TimeoutError:
            logger.warning(
                f"Timeout waiting for operations to complete after {timeout}s"
            )
            return False

    def _on_operation_completed(self, operation: BaseOperation):
        logger.debug(
            f"Operation {operation.operation_id} completed with status: {operation.status.value}"
        )

        for callback in self._completion_callbacks:
            try:
                callback(operation)
            except Exception as e:
                logger.error("Error in completion callback", exc_info=e)

    async def _cancel_all_operations(self) -> int:
        operations_to_cancel = list(self.operations.values())

        if not operations_to_cancel:
            return 0

        logger.info(f"Cancelling {len(operations_to_cancel)} operations")
        cancel_tasks = []

        for op in operations_to_cancel:
            try:
                cancel_tasks.append(op.cancel())
            except Exception as e:
                logger.error(
                    f"Error creating cancel task for operation {op.operation_id}",
                    exc_info=e,
                )

        if cancel_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*cancel_tasks, return_exceptions=True), timeout=5.0
                )
            except asyncio.TimeoutError:
                logger.warning("Some operations did not cancel within timeout")

        return len(operations_to_cancel)

    async def _periodic_cleanup(self):
        logger.debug("Starting periodic cleanup loop")
        try:
            while self._is_running:
                try:
                    await asyncio.sleep(60.0)  # Check every minute
                    await self._cleanup_completed_operations()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("Error in periodic cleanup", exc_info=e)
                    if not self._is_running:
                        break
        except asyncio.CancelledError:
            pass
        finally:
            logger.debug("Periodic cleanup stopped")

    async def _cleanup_completed_operations(self):
        operations_to_remove = []

        for operation_id, operation in self.operations.items():
            if operation.status in [
                OperationStatus.COMPLETED,
                OperationStatus.FAILED,
                OperationStatus.CANCELLED,
                OperationStatus.TIMEOUT,
            ]:
                operations_to_remove.append(operation_id)

        for operation_id in operations_to_remove:
            await self.unregister_operation(operation_id)

        if operations_to_remove:
            logger.debug(f"Cleaned up {len(operations_to_remove)} completed operations")
