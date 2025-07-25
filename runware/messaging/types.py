import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Dict


class MessageType(Enum):
    IMAGE_INFERENCE = "imageInference"
    VIDEO_INFERENCE = "videoInference"
    PHOTO_MAKER = "photoMaker"
    IMAGE_UPLOAD = "imageUpload"
    IMAGE_UPSCALE = "imageUpscale"
    IMAGE_BACKGROUND_REMOVAL = "imageBackgroundRemoval"
    IMAGE_CAPTION = "imageCaption"
    PROMPT_ENHANCE = "promptEnhance"
    AUTHENTICATION = "authentication"
    MODEL_UPLOAD = "modelUpload"
    MODEL_SEARCH = "modelSearch"
    GET_RESPONSE = "getResponse"
    PING = "ping"
    ERROR = "error"


@dataclass
class Message:
    message_type: MessageType
    operation_id: str
    data: Dict[str, Any]
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()


MessageHandler = Callable[[Message], Awaitable[None]]
ProgressCallback = Callable[[Any], None]
CompletionCallback = Callable[[Any], None]
