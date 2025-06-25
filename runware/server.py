import asyncio
import json
import logging
import websockets
from websockets.protocol import State
from typing import Any, Dict, Optional


from .types import SdkType
from .utils import (
    BASE_RUNWARE_URLS,
    PING_INTERVAL,
    PING_TIMEOUT_DURATION,
    TIMEOUT_DURATION,
)
from .base import RunwareBase
from .types import (
    Environment,
    ListenerType,
)

from .logging_config import configure_logging


class RunwareServer(RunwareBase):
    def __init__(
        self,
        api_key: str,
        url: str = BASE_RUNWARE_URLS[Environment.PRODUCTION],
        log_level=logging.CRITICAL,
        timeout: int = TIMEOUT_DURATION,
    ):
        super().__init__(api_key=api_key, url=url, timeout=timeout)
        self._instantiated: bool = False
        self._reconnecting_task: Optional[asyncio.Task] = None
        self._pingTimeout: Optional[asyncio.Task] = None
        self._pongListener: Optional[ListenerType] = None
        self._loginListener: Optional[ListenerType] = None
        self._sdkType: SdkType = SdkType.SERVER
        self._apiKey: str = api_key
        self._message_handler_task: Optional[asyncio.Task] = None
        self._last_pong_time: float = 0.0
        self._is_shutting_down: bool = False

        # Configure logging
        configure_logging(log_level)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)

    async def connect(self):
        self.logger.info("Connecting to Runware server from server")

        try:
            self._ws = await websockets.connect(self._url)
            # update close_timeout so that we end the script sooner for inference examples
            self._ws.close_timeout = 1
            self._ws.max_size = None
            self.logger.info(f"Connected to WebSocket URL: {self._url}")

            async def on_open(ws):
                def login_check(m):
                    if (
                        m.get("data")
                        and len(m["data"]) > 0
                        and m["data"][0].get("connectionSessionUUID")
                    ):
                        return True
                    if m.get("errors"):
                        for error in m["errors"]:
                            if error.get("taskType") == "authentication":
                                return True
                    return False

                def login_lis(m):
                    if m.get("errors"):
                        for error in m["errors"]:
                            if error.get("taskType") == "authentication":
                                err_msg = "Authentication error"
                                self._invalidAPIkey = error.get("message") or err_msg
                                self._connection_session_uuid_event.set()
                                return
                    if m.get("data") and len(m["data"]) > 0:
                        self._connectionSessionUUID = m["data"][0].get(
                            "connectionSessionUUID"
                        )
                        self._invalidAPIkey = None
                        self._connection_session_uuid_event.set()

                if not self._loginListener:
                    self._loginListener = self.addListener(
                        check=login_check, lis=login_lis
                    )

                def pong_check(m):
                    return m.get("data", [])[0].get("pong") if m.get("data") else None

                def pong_lis(m):
                    if m.get("data", [])[0].get("pong"):
                        self._last_pong_time = asyncio.get_event_loop().time()

                self._connection_session_uuid_event = asyncio.Event()

                if not self._pongListener:
                    self._pongListener = self.addListener(
                        check=pong_check, lis=pong_lis
                    )

                if self._reconnecting_task:
                    self._reconnecting_task.cancel()

                if self._connectionSessionUUID and self.isWebsocketReadyState():
                    self.logger.info(
                        f"Starting new connection with connectionSessionUUID {self._connectionSessionUUID}"
                    )
                    await self.send(
                        [
                            {
                                "taskType": "authentication",
                                "apiKey": self._apiKey,
                                "connectionSessionUUID": self._connectionSessionUUID,
                            }
                        ]
                    )
                elif self.isWebsocketReadyState():
                    self.logger.info("Starting new connection with apiKey only")
                    await self.send(
                        [
                            {
                                "taskType": "authentication",
                                "apiKey": self._apiKey,
                            }
                        ]
                    )

                if self.isWebsocketReadyState():
                    self.logger.info("Starting heartbeat task")
                    self._heartbeat_task = asyncio.create_task(
                        self.heartBeat(), name="Task_Heartbeat"
                    )

            self._message_handler_task = asyncio.create_task(
                self._handle_messages(), name="Task_Message_Handler"
            )
            await on_open(self._ws)
            # Wait for the _connectionSessionUUID to be set
            await self._connection_session_uuid_event.wait()

        except websockets.exceptions.ConnectionClosedError:
            await self.handleClose()

    async def on_message(self, ws, message):
        if not message:
            return

        try:
            m = json.loads(message)
        except json.JSONDecodeError as e:
            self.logger.error(f"Failed to parse JSON message:", exc_info=e)
            return

        for lis in self._listeners:
            try:
                result = lis.listener(m)
            except Exception as e:
                self.logger.error(f"Error in listener {lis.key}:", exc_info=e)
                continue
            if result:
                return

    async def _handle_messages(self):
        try:
            self.logger.debug(
                f"Starting message handler task {self._message_handler_task}"
            )
            async for message in self._ws:
                if self._is_shutting_down:
                    break
                try:
                    await self.on_message(self._ws, message)
                except Exception as e:
                    self.logger.error(f"Error in on_message:", exc_info=e)
                    continue
        except websockets.exceptions.ConnectionClosedError as e:
            if not self._is_shutting_down:
                self.logger.error(f"Connection Closed Error:", exc_info=e)
                await self.handleClose()
        except Exception as e:
            self.logger.error(f"Critical error in _handle_messages:", exc_info=e)
            if not self._is_shutting_down:
                await self.handleClose()

    async def send(self, msg: Dict[str, Any]):
        self.logger.debug(f"Sending message: {msg}")
        if self._ws and self._ws.state is State.OPEN and not self._is_shutting_down:
            await self._ws.send(json.dumps(msg))

    def _get_task_by_name(self, name):
        tasks = asyncio.all_tasks()
        for task in tasks:
            if task.get_name() == name:
                return task
        return None

    async def handleClose(self):
        self.logger.debug("Handling close")

        if self._invalidAPIkey:
            self.logger.error(f"Error: {self._invalidAPIkey}")
            return

        reconnecting_task = self._get_task_by_name("Task_Reconnecting")
        if reconnecting_task is not None:
            if not reconnecting_task.done() and not reconnecting_task.cancelled():
                self.logger.debug(f"Cancelling Task_Reconnecting {reconnecting_task}")
                try:
                    reconnecting_task.cancel()
                except Exception as e:
                    self.logger.error(f"Error while cancelling Task_Reconnecting:", exc_info=e)

        message_handler_task = self._get_task_by_name("Task_Message_Handler")
        if message_handler_task is not None:
            if not message_handler_task.done() and not message_handler_task.cancelled():
                self.logger.debug(
                    f"Cancelling Task_Message_Handler {message_handler_task}"
                )
                try:
                    message_handler_task.cancel()
                except Exception as e:
                    self.logger.error(
                        f"Error while cancelling Task_Message_Handler:", exc_info=e
                    )

        heartbeat_task = self._get_task_by_name("Task_Heartbeat")
        if heartbeat_task is not None:
            if not heartbeat_task.done() and not heartbeat_task.cancelled():
                self.logger.debug(f"Cancelling Task_Heartbeat {heartbeat_task}")
                try:
                    heartbeat_task.cancel()
                except Exception as e:
                    self.logger.error(f"Error while cancelling Task_Heartbeat:", exc_info=e)

        async def reconnect():
            reconnect_attempts = 0
            max_reconnect_attempts = 5

            while reconnect_attempts < max_reconnect_attempts and not self._is_shutting_down:
                self.logger.info(f"Reconnecting... (attempt {reconnect_attempts + 1})")
                await asyncio.sleep(min(reconnect_attempts * 2 + 1, 10))
                try:
                    await self.connect()
                    if self.isWebsocketReadyState():
                        self.logger.info("Reconnected successfully")
                        break  # Break out of the loop if the connection is successful and in a ready state
                    else:
                        self.logger.warning(
                            "WebSocket connection is not in a ready state after reconnecting"
                        )
                except Exception as e:
                    self.logger.error(f"Error while reconnecting:", exc_info=e)

                reconnect_attempts += 1

            if reconnect_attempts >= max_reconnect_attempts:
                self.logger.error("Max reconnection attempts reached. Giving up.")
                self._is_shutting_down = True

        # Attempting to reconnect...
        if not self._is_shutting_down:
            self._reconnecting_task = asyncio.create_task(
                reconnect(), name="Task_Reconnecting"
            )

    async def heartBeat(self):
        while not self._is_shutting_down:
            if self.isWebsocketReadyState():
                self.logger.debug("Sending ping")
                try:
                    await self.send([{"taskType": "ping", "ping": True}])
                except websockets.exceptions.ConnectionClosedError as e:
                    self.logger.error(
                        f"Error sending ping. Connection likely closed.", exc_info=e
                    )
                    break
                except Exception as e:
                    self.logger.error(f"Unexpected error sending ping", exc_info=e)
                    break

                await asyncio.sleep(PING_INTERVAL / 1000)

                if (
                        asyncio.get_event_loop().time() - self._last_pong_time
                        > PING_TIMEOUT_DURATION / 1000
                ):
                    self.logger.warning("No pong received. Connection may be lost.")
                    await self.handleClose()
                    break
            else:
                break
