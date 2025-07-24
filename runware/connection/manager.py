import asyncio
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

import websockets
from websockets.protocol import State

from .types import ConnectionState
from ..core.cpu_bound import cpu_executor
from ..exceptions import RunwareAuthenticationError, RunwareConnectionError
from ..logging_config import get_logger

if TYPE_CHECKING:
    from ..messaging.router import MessageRouter

logger = get_logger(__name__)


class ConnectionManager:
    """Manages WebSocket connection with proper error handling."""

    def __init__(self, api_key: str, url: str, message_router: "MessageRouter"):
        self.api_key = api_key
        self.url = url
        self.message_router = message_router

        self.state = ConnectionState.DISCONNECTED
        self.websocket: Optional[websockets.WebSocketServerProtocol] = None
        self.connection_session_uuid: Optional[str] = None

        # Message sending
        self.message_queue = asyncio.Queue(maxsize=1000)

        # Lifecycle management
        self.is_running = False
        self._should_reconnect = True

        # Background tasks
        self._message_handler_task: Optional[asyncio.Task] = None
        self._sender_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None

        # Events for synchronization
        self._connection_event = asyncio.Event()
        self._authentication_event = asyncio.Event()
        self._authentication_error_event = asyncio.Event()
        self._auth_error: Optional[Exception] = None

        # Configuration
        self._heartbeat_interval = 30.0
        self._heartbeat_timeout = 60.0
        self._last_pong_time = 0.0

        # Reconnection settings
        self._reconnect_attempts = 0
        self._max_reconnect_attempts = 3
        self._base_reconnect_delay = 2.0
        self._max_reconnect_delay = 30.0

        # Callbacks
        self._connection_callbacks: List[Callable[[ConnectionState], None]] = []

        logger.info(f"ConnectionManager initialized for URL: {url}")

    async def start(self):
        if self.is_running:
            return

        self.is_running = True
        self._should_reconnect = True
        logger.info("Starting ConnectionManager")

        try:
            await self._connect()
        except Exception as e:
            self.is_running = False
            raise

    async def stop(self):
        if not self.is_running:
            return

        logger.info("Stopping ConnectionManager")
        self.is_running = False
        self._should_reconnect = False

        await self._disconnect()
        await self._cleanup_tasks()

    async def send_message(self, content: List[Dict[str, Any]]) -> str:
        if not self.is_connected():
            raise RunwareConnectionError("Cannot send message: not connected")

        message_id = f"msg_{int(time.time() * 1000000)}"
        message_data = {"id": message_id, "content": content}

        try:
            await asyncio.wait_for(self.message_queue.put(message_data), timeout=5.0)
            return message_id
        except asyncio.TimeoutError:
            raise RunwareConnectionError("Message queue full")

    def is_connected(self) -> bool:
        return (
            self.websocket is not None
            and self.websocket.state == State.OPEN
            and self.state in [ConnectionState.CONNECTED, ConnectionState.AUTHENTICATED]
        )

    def is_authenticated(self) -> bool:
        return (
            self.state == ConnectionState.AUTHENTICATED
            and self.connection_session_uuid is not None
        )

    async def wait_for_authentication(self, timeout: Optional[float] = 30.0) -> bool:
        try:
            await asyncio.wait_for(self._authentication_event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def add_connection_callback(self, callback: Callable[[ConnectionState], None]):
        self._connection_callbacks.append(callback)

    def remove_connection_callback(self, callback: Callable[[ConnectionState], None]):
        if callback in self._connection_callbacks:
            self._connection_callbacks.remove(callback)

    async def _connect(self):
        if self.state in [ConnectionState.CONNECTING, ConnectionState.RECONNECTING]:
            return

        self._set_state(ConnectionState.CONNECTING)

        try:
            logger.info(f"Connecting to WebSocket: {self.url}")

            self.websocket = await asyncio.wait_for(
                websockets.connect(
                    self.url,
                    close_timeout=2,
                    max_size=None,
                    ping_interval=None,
                    ping_timeout=None,
                ),
                timeout=15.0,
            )

            self._set_state(ConnectionState.CONNECTED)
            self._connection_event.set()

            # Start background tasks
            await self._start_connection_tasks()

            # Authenticate - this will throw RunwareAuthenticationError if auth fails
            await self._authenticate()

            logger.info("WebSocket connection authenticated successfully")

        except RunwareAuthenticationError:
            # Authentication errors are not recoverable - don't retry
            self._should_reconnect = False
            self._set_state(ConnectionState.FAILED)
            raise
        except Exception as e:
            self._set_state(ConnectionState.FAILED)

            if self.is_running and self._should_reconnect:
                await self._schedule_reconnect()
            else:
                raise RunwareConnectionError(f"Connection failed: {e}")

    async def _start_connection_tasks(self):
        self._message_handler_task = asyncio.create_task(self._handle_messages())
        self._sender_task = asyncio.create_task(self._send_messages())
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    async def _authenticate(self):
        if not self.is_connected():
            raise RunwareConnectionError("Cannot authenticate: not connected")

        self._authentication_event.clear()
        self._authentication_error_event.clear()
        self._auth_error = None

        auth_message = [{"taskType": "authentication", "apiKey": self.api_key}]

        if self.connection_session_uuid:
            auth_message[0]["connectionSessionUUID"] = self.connection_session_uuid

        await self.send_message(auth_message)

        # Wait for either success or error
        done, pending = await asyncio.wait(
            [
                asyncio.create_task(self._authentication_event.wait()),
                asyncio.create_task(self._authentication_error_event.wait()),
            ],
            timeout=15.0,
            return_when=asyncio.FIRST_COMPLETED,
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()

        if not done:
            raise RunwareAuthenticationError("Authentication timeout")

        # Check if there was an error
        if self._authentication_error_event.is_set():
            if self._auth_error:
                raise self._auth_error
            else:
                raise RunwareAuthenticationError("Authentication failed")

    async def _send_messages(self):
        try:
            while self.is_running and self.is_connected():
                try:
                    message = await asyncio.wait_for(
                        self.message_queue.get(), timeout=1.0
                    )
                    await self._send_single_message(message)
                except asyncio.TimeoutError:
                    continue
                except Exception as e:
                    logger.error("Error in message sender", exc_info=e)
                    if not self.is_running:
                        break
        except asyncio.CancelledError:
            pass

    async def _send_single_message(self, message: Dict[str, Any]):
        try:
            serialized = await cpu_executor.serialize_json(message["content"])
            await self.websocket.send(serialized)
        except Exception as e:
            logger.error(f"Failed to send message {message['id']}", exc_info=e)
            raise

    async def _handle_messages(self):
        try:
            async for raw_message in self.websocket:
                if not self.is_running:
                    break
                await self._process_incoming_message(raw_message)
        except websockets.exceptions.ConnectionClosed:
            if self.is_running and self._should_reconnect:
                await self._handle_connection_loss()
        except Exception as e:
            logger.error("Error in message handler", exc_info=e)
            if self.is_running and self._should_reconnect:
                await self._handle_connection_loss()

    async def _process_incoming_message(self, raw_message: str):
        try:
            message = await cpu_executor.parse_json(raw_message)

            # Handle system messages first
            if await self._handle_system_message(message):
                return

            # Route to operations
            await self.message_router.route_message(message)

        except Exception as e:
            logger.error("Error processing incoming message", exc_info=e)

    async def _handle_system_message(self, message: Dict[str, Any]) -> bool:
        if "data" in message:
            for item in message["data"]:
                if item.get("taskType") == "authentication":
                    return await self._handle_authentication_response(item)
                elif item.get("pong"):
                    self._last_pong_time = time.time()
                    return True

        if "errors" in message:
            for error in message["errors"]:
                if error.get("taskType") == "authentication":
                    await self._handle_authentication_error(error)
                    return True

        return False

    async def _handle_authentication_response(self, auth_data: Dict[str, Any]) -> bool:
        self.connection_session_uuid = auth_data.get("connectionSessionUUID")

        if self.connection_session_uuid:
            self._set_state(ConnectionState.AUTHENTICATED)
            self._authentication_event.set()
            self._reconnect_attempts = 0
            return True
        else:
            self._auth_error = RunwareAuthenticationError(
                "Authentication response missing session UUID"
            )
            self._authentication_error_event.set()
            return False

    async def _handle_authentication_error(self, error_data: Dict[str, Any]):
        error_message = error_data.get("message", "Authentication failed")

        # Store the error and signal error event
        self._auth_error = RunwareAuthenticationError(error_message)
        self._authentication_error_event.set()

        # Don't allow reconnection for auth errors
        self._should_reconnect = False
        self._set_state(ConnectionState.FAILED)

    async def _heartbeat_loop(self):
        try:
            while self.is_running and self.is_connected():
                try:
                    ping_message = [{"taskType": "ping", "ping": True}]
                    await self.send_message(ping_message)

                    await asyncio.sleep(self._heartbeat_interval)

                    if (time.time() - self._last_pong_time) > self._heartbeat_timeout:
                        if self.is_running and self._should_reconnect:
                            await self._handle_connection_loss()
                            break

                except Exception as e:
                    logger.error("Heartbeat error", exc_info=e)
                    if not self.is_running:
                        break
        except asyncio.CancelledError:
            pass

    async def _handle_connection_loss(self):
        if self.state == ConnectionState.RECONNECTING:
            return

        await self._disconnect()

        if self.is_running and self._should_reconnect:
            await self._schedule_reconnect()

    async def _schedule_reconnect(self):
        self._set_state(ConnectionState.RECONNECTING)

        while (
            self.is_running
            and self._should_reconnect
            and self._reconnect_attempts < self._max_reconnect_attempts
            and self.state != ConnectionState.AUTHENTICATED
        ):
            self._reconnect_attempts += 1

            delay = min(
                self._base_reconnect_delay * (2 ** (self._reconnect_attempts - 1)),
                self._max_reconnect_delay,
            )

            logger.info(
                f"Reconnect attempt {self._reconnect_attempts}/{self._max_reconnect_attempts} in {delay:.2f}s"
            )

            try:
                await asyncio.sleep(delay)
            except asyncio.CancelledError:
                break

            try:
                await self._connect()
                if self.is_authenticated():
                    self._reconnect_attempts = 0
                    break
            except RunwareAuthenticationError:
                # Authentication errors should stop reconnection attempts
                self._should_reconnect = False
                break
            except Exception as e:
                logger.error(
                    f"Reconnect attempt {self._reconnect_attempts} failed", exc_info=e
                )

        if self._reconnect_attempts >= self._max_reconnect_attempts:
            self._set_state(ConnectionState.FAILED)

    async def _disconnect(self):
        if self.state == ConnectionState.DISCONNECTED:
            return

        self._set_state(ConnectionState.DISCONNECTED)

        if self.websocket and self.websocket.state == State.OPEN:
            try:
                await asyncio.wait_for(self.websocket.close(), timeout=3.0)
            except Exception:
                pass

        self.websocket = None
        self.connection_session_uuid = None

        # Clear events
        self._connection_event.clear()
        self._authentication_event.clear()
        self._authentication_error_event.clear()

    async def _cleanup_tasks(self):
        tasks = [self._message_handler_task, self._sender_task, self._heartbeat_task]

        for task in tasks:
            if task and not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass

        # Clear task references
        self._message_handler_task = None
        self._sender_task = None
        self._heartbeat_task = None

    def _set_state(self, new_state: ConnectionState):
        if self.state != new_state:
            old_state = self.state
            self.state = new_state

            logger.info(
                f"Connection state changed: {old_state.value} -> {new_state.value}"
            )

            for callback in self._connection_callbacks:
                try:
                    callback(new_state)
                except Exception as e:
                    logger.error("Error in connection callback", exc_info=e)
