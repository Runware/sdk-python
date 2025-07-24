from enum import Enum
from typing import Callable


class ConnectionState(Enum):
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    RECONNECTING = "reconnecting"
    FAILED = "failed"


ConnectionStateCallback = Callable[[ConnectionState], None]
