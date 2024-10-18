import asyncio
import json
import logging
import websockets
import inspect
import pprint
from typing import Any, Callable, Dict, List, Union, Optional, TypeVar


from .types import RunwareBaseType, SdkType
from .utils import (
    delay,
    getUUID,
    removeListener,
    BASE_RUNWARE_URLS,
    PING_INTERVAL,
    PING_TIMEOUT_DURATION,
)
from .base import RunwareBase
from .types import (
    Environment,
    EPreProcessor,
    EPreProcessorGroup,
    ListenerType,
    IControlNet,
    File,
    GetWithPromiseCallBackType,
)

from .logging_config import configure_logging


class RunwareServer(RunwareBase):
    def __init__(
        self,
        api_key: str,
        url: str = BASE_RUNWARE_URLS[Environment.PRODUCTION],
        log_level=logging.CRITICAL,
    ):
        super().__init__(api_key=api_key, url=url)
        self._instantiated: bool = False
        self._reconnecting_task: Optional[asyncio.Task] = None
        self._pingTimeout: Optional[asyncio.Task] = None
        self._pongListener: Optional[ListenerType] = None
        self._loginListener: Optional[ListenerType] = None
        self._sdkType: SdkType = SdkType.SERVER
        self._apiKey: str = api_key
        self._message_handler_task: Optional[asyncio.Task] = None
        self._last_pong_time: float = 0.0

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
            self.logger.info(f"Connected to WebSocket URL: {self._url}")

            async def on_open(ws):

                def login_check(m):
                    return (
                        m.get("data", [])[0].get("connectionSessionUUID")
                        if m.get("data")
                        else None
                    )

                def login_lis(m):
                    if m.get("error"):
                        if m["errorId"] == 19:
                            self._invalidAPIkey = "Invalid API key"
                        else:
                            self._invalidAPIkey = "Error connection"
                        return

                    self._connectionSessionUUID = m.get("data", [])[0].get(
                        "connectionSessionUUID"
                    )
                    self._invalidAPIkey = None
                    self._connection_session_uuid_event.set()  # Set the event when _connectionSessionUUID is received

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
                        "Starting new connection with connectionSessionUUID"
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
        m = json.loads(message)
        # print(
        #     f"\n\n\n================================================ Received message ============================================================"
        # )
        # print(f"{m}")

        # print(f"Listenerse:")
        # for lis in self._listeners:
        #     print(lis, "\n")
        # print(
        #     f"============================================= End received message ============================================================\n\n\n"
        # )
        for lis in self._listeners:
            try:
                # result = True
                result = lis.listener(m)
            except Exception as e:
                print(f"Unexpected error in on_message: {e}")
                print(dir(lis))
                print(f"Listeners: {self._listeners}")
                for lis in self._listeners:
                    print(dir(lis), "\n")
                return
            if result:
                return

    async def _handle_messages(self):
        try:
            self.logger.debug(
                f"Starting message handler task {self._message_handler_task}"
            )
            async for message in self._ws:
                try:
                    await self.on_message(self._ws, message)
                except Exception as e:
                    print(f"Unexpected error in async loop: {e}")
                    print(self.on_message)
                    exit()
        except websockets.exceptions.ConnectionClosedError as e:
            self.logger.error(f"Connection Closed Error: {e}")
            await self.handleClose()
        except Exception as e:
            print(f"Unexpected error in _handle_messages: {e}")
            print(self.on_message)
            exit()
            await self._ws.close()

    async def send(self, msg: Dict[str, Any]):
        self.logger.debug(f"Sending message: {msg}")
        # print(
        #     f"\n\n\n================================================= Sending message ================================================================="
        # )
        # print(f"{msg}")
        # print(
        #     f"=============================================== End sending message ===============================================================\n\n\n"
        # )
        if self._ws and self._ws.open:
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
                    self.logger.error(f"Error while cancelling Task_Reconnecting: {e}")

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
                        f"Error while cancelling Task_Message_Handler: {e}"
                    )

        heartbeat_task = self._get_task_by_name("Task_Heartbeat")
        if heartbeat_task is not None:
            if not heartbeat_task.done() and not heartbeat_task.cancelled():
                self.logger.debug(f"Cancelling Task_Heartbeat {heartbeat_task}")
                try:
                    heartbeat_task.cancel()
                except Exception as e:
                    self.logger.error(f"Error while cancelling Task_Heartbeat: {e}")

        async def reconnect():
            while True:
                self.logger.info("Reconnecting...")
                await asyncio.sleep(1)
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
                    self.logger.error(f"Error while reconnecting: {e}")

        # TODO: I don't need to close self._ws here, as it will be cleaned by sockets library based on it's interrnal ping mechanism
        # Attempting to reconnect...
        self._reconnecting_task = asyncio.create_task(
            reconnect(), name="Task_Reconnecting"
        )

    async def heartBeat(self):
        # TODO: Not sure if we need this, as the websocket server responds to default PING messages
        # 2024-04-29 10:46:23,193 - websockets.client - DEBUG - % sending keepalive ping
        # 2024-04-29 10:46:23,194 - websockets.client - DEBUG - > PING f2 0b eb 3d [binary, 4 bytes]
        # 2024-04-29 10:46:23,197 - runware.server - DEBUG - Sending ping
        # 2024-04-29 10:46:23,197 - runware.server - DEBUG - Sending message: {'ping': True}
        # 2024-04-29 10:46:23,197 - websockets.client - DEBUG - > TEXT '{"ping": true}' [14 bytes]
        # 2024-04-29 10:46:23,241 - websockets.client - DEBUG - < PONG f2 0b eb 3d [binary, 4 bytes]
        # 2024-04-29 10:46:23,241 - websockets.client - DEBUG - % received keepalive pong
        # 2024-04-29 10:46:23,244 - websockets.client - DEBUG - < TEXT '{"pong":true}' [13 bytes]
        while True:
            if self.isWebsocketReadyState():
                self.logger.debug("Sending ping")
                try:
                    await self.send([{"taskType": "ping", "ping": True}])
                except websockets.exceptions.ConnectionClosedError as e:
                    self.logger.error(
                        f"Error sending ping: {e}. Connection likely closed."
                    )
                    # Potentially handle reconnection here
                except Exception as e:  # Catch other potential exceptions
                    self.logger.error(f"Unexpected error sending ping: {e}")
                    # Handle unexpected errors appropriately
                await asyncio.sleep(PING_INTERVAL / 1000)

                if (
                    asyncio.get_event_loop().time() - self._last_pong_time
                    > PING_TIMEOUT_DURATION / 1000
                ):  # No pong received within the timeout period
                    self.logger.warning("No pong received. Connection may be lost.")
                    # Initiate a reconnection
                    await self.handleClose()
