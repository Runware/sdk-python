import asyncio
import time
from collections import defaultdict
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from ..logging_config import get_logger

if TYPE_CHECKING:
    from .. import BaseOperation

logger = get_logger(__name__)


class MessageRouter:
    """Routes messages to appropriate operations."""

    def __init__(self):
        self.operations: Dict[str, "BaseOperation"] = {}
        self.message_queue = asyncio.Queue(maxsize=1000)
        self.router_task: Optional[asyncio.Task] = None
        self.is_running = False

        # Store messages for operations that aren't registered yet
        self._pending_messages: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self._message_ttl = 300.0  # 5 minutes
        logger.info("MessageRouter initialized")

    async def start(self):
        if self.is_running:
            return

        self.is_running = True
        self.router_task = asyncio.create_task(self._route_messages())
        logger.info("Message router started")

    async def stop(self):
        if not self.is_running:
            return

        self.is_running = False

        if self.router_task and not self.router_task.done():
            self.router_task.cancel()
            try:
                await asyncio.wait_for(self.router_task, timeout=3.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
            self.router_task = None

        # Clear queues
        while not self.message_queue.empty():
            try:
                self.message_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

    async def register_operation(self, operation: "BaseOperation"):
        if operation.operation_id in self.operations:
            logger.warning(f"Operation {operation.operation_id} already registered")
            return

        self.operations[operation.operation_id] = operation

        # Deliver any pending messages for this operation
        pending_messages = self._pending_messages.pop(operation.operation_id, [])
        if pending_messages:
            logger.info(
                f"Delivering {len(pending_messages)} pending messages to operation {operation.operation_id}"
            )
            for message in pending_messages:
                await self._deliver_message_to_operation(operation, message)

        logger.info(
            f"Registered operation {operation.operation_id} ({operation.operation_type})"
        )

    async def unregister_operation(self, operation_id: str):
        if operation_id in self.operations:
            del self.operations[operation_id]
            logger.debug(f"Unregistered operation {operation_id}")

        # Remove any pending messages for this operation
        if operation_id in self._pending_messages:
            del self._pending_messages[operation_id]

    async def route_message(self, message: Dict[str, Any]):
        if not self.is_running:
            logger.warning("Message router not running, dropping message")
            return

        try:
            await asyncio.wait_for(self.message_queue.put(message), timeout=5.0)
        except asyncio.TimeoutError:
            logger.error("Message queue full, dropping message")

    async def _route_messages(self):
        logger.debug("Starting message routing loop")

        try:
            while self.is_running:
                try:
                    message = await asyncio.wait_for(
                        self.message_queue.get(), timeout=1.0
                    )
                    await self._process_message(message)
                except asyncio.TimeoutError:
                    # Cleanup expired pending messages
                    await self._cleanup_expired_messages()
                    continue
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error("Error in message routing loop", exc_info=e)
        except asyncio.CancelledError:
            pass
        finally:
            logger.debug("Message routing loop stopped")

    async def _process_message(self, raw_message: Dict[str, Any]):
        try:
            logger.debug(f"Processing raw message: {raw_message}")
            if self._is_error_message(raw_message):
                logger.error(f"Processing ERROR message: {raw_message}")
                await self._handle_error_message(raw_message)
            elif self._is_data_message(raw_message):
                await self._handle_data_message(raw_message)
            else:
                logger.warning(f"Unknown message format: {raw_message}")

        except Exception as e:
            logger.error("Error processing message", exc_info=e)

    def _is_data_message(self, message: Dict[str, Any]) -> bool:
        has_data = (
            "data" in message
            and isinstance(message["data"], list)
            and len(message["data"]) > 0
        )
        has_errors = (
            "errors" in message
            and isinstance(message["errors"], list)
            and len(message["errors"]) > 0
        )
        return has_data and not has_errors

    def _is_error_message(self, message: Dict[str, Any]) -> bool:
        return (
            "errors" in message
            and isinstance(message["errors"], list)
            and len(message["errors"]) > 0
        )

    async def _handle_data_message(self, raw_message: Dict[str, Any]):
        data_items = raw_message.get("data", [])
        logger.debug(f"Handling data message with {len(data_items)} items")

        for item in data_items:
            operation_id = item.get("taskUUID")
            task_type = item.get("taskType")

            logger.debug(
                f"Processing data item - taskUUID: {operation_id}, taskType: {task_type}"
            )

            if not operation_id:
                logger.warning(f"Data item missing taskUUID: {item}")
                continue

            operation = self.operations.get(operation_id)
            if operation:
                logger.debug(f"Delivering message to operation {operation_id}")
                await self._deliver_message_to_operation(operation, item)
            else:
                logger.debug(
                    f"Storing message for unregistered operation {operation_id}"
                )
                item["_received_at"] = time.time()
                self._pending_messages[operation_id].append(item)

    async def _handle_error_message(self, raw_message: Dict[str, Any]):
        error_items = raw_message.get("errors", [])
        logger.error(f"Handling error message with {len(error_items)} error items")

        for item in error_items:
            operation_id = item.get("taskUUID")
            error_message = item.get("message", "Unknown error")
            error_code = item.get("code")

            logger.error(
                f"Processing server error - taskUUID: {operation_id}, message: {error_message}, code: {error_code}"
            )
            logger.error(f"Full error item: {item}")

            error_message_data = {**item, "taskType": "error"}

            if operation_id and operation_id != "N/A":
                operation = self.operations.get(operation_id)
                if operation:
                    logger.error(f"Delivering error to operation {operation_id}")
                    await self._deliver_message_to_operation(
                        operation, error_message_data
                    )
                else:
                    # Store error for later delivery
                    logger.error(
                        f"Storing error for unregistered operation {operation_id}"
                    )
                    error_message_data["_received_at"] = time.time()
                    self._pending_messages[operation_id].append(error_message_data)
            else:
                # Unmatched error - deliver to all active operations
                logger.error(
                    f"Unmatched error, delivering to all active operations: {error_message}"
                )
                await self._handle_unmatched_error(error_message_data)

    async def _handle_unmatched_error(self, error_message: Dict[str, Any]):
        """Handle errors that don't have a specific taskUUID."""
        if not self.operations:
            logger.warning("No active operations to deliver unmatched error to")
            return

        error_text = error_message.get("message", "")

        target_operations = []

        if "image" in error_text.lower():
            target_operations = [
                op
                for op in self.operations.values()
                if op.operation_type
                in ["imageInference", "imageGeneration", "photoMaker"]
            ]
        elif "video" in error_text.lower():
            target_operations = [
                op
                for op in self.operations.values()
                if op.operation_type == "videoInference"
            ]

        if not target_operations:
            target_operations = [
                max(self.operations.values(), key=lambda op: op.created_at)
            ]
            logger.warning(
                f"Delivering unmatched error to most recent operation: {target_operations[0].operation_id}"
            )

        for operation in target_operations:
            enhanced_error = {
                **error_message,
                "taskUUID": operation.operation_id,
                "message": f"Server error: {error_text}",
            }
            logger.debug(
                f"Delivering unmatched error to operation {operation.operation_id}"
            )
            await self._deliver_message_to_operation(operation, enhanced_error)

    async def _deliver_message_to_operation(self, operation, message: Dict[str, Any]):
        try:
            task_type = message.get("taskType")
            operation_id = operation.operation_id

            logger.debug(
                f"Delivering message to operation {operation_id}: taskType={task_type}"
            )

            # Special logging for different message types
            match task_type:
                case "getResponse":
                    status = message.get("status")
                    error_info = message.get("error") or message.get("message", "")
                    logger.info(
                        f"Delivering getResponse to operation {operation_id}: status={status}, error={error_info}, full_message={message}"
                    )
                case "videoInference":
                    status = message.get("status")
                    error_info = message.get("error", "")
                    logger.info(
                        f"Delivering videoInference to operation {operation_id}: status={status}, error={error_info}"
                    )
                case "error":
                    error_msg = message.get("message", "Unknown error")
                    error_code = message.get("code", "")
                    logger.error(
                        f"Delivering ERROR to operation {operation_id}: {error_msg} (code: {error_code})"
                    )

            await operation.handle_message(message)
            logger.debug(f"Successfully delivered message to operation {operation_id}")

        except Exception as e:
            logger.error(
                f"Error delivering message to operation {operation.operation_id}",
                exc_info=e,
            )

    async def _cleanup_expired_messages(self):
        """Remove messages that are too old."""
        current_time = time.time()
        expired_operations = []
        total_expired = 0

        for operation_id, messages in self._pending_messages.items():
            valid_messages = []
            for message in messages:
                received_at = message.get("_received_at", current_time)
                if current_time - received_at < self._message_ttl:
                    valid_messages.append(message)
                else:
                    total_expired += 1

            if valid_messages:
                self._pending_messages[operation_id] = valid_messages
            else:
                expired_operations.append(operation_id)

        for operation_id in expired_operations:
            del self._pending_messages[operation_id]

        if total_expired > 0:
            logger.debug(f"Cleaned up {total_expired} expired messages")
