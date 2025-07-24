import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class OperationStatus(Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class MessageType(Enum):
    OPERATION_UPDATE = "operation_update"
    OPERATION_COMPLETE = "operation_complete"
    OPERATION_ERROR = "operation_error"
    PROGRESS_UPDATE = "progress_update"
    SERVER_MESSAGE = "server_message"


@dataclass
class OperationContext:
    operation_id: str
    operation_type: str
    status: OperationStatus
    progress: float = 0.0
    results: List[Any] = None
    error: Optional[Exception] = None
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    metadata: Dict[str, Any] = None
    completed_at: Optional[float] = None
    created_at: Optional[float] = None

    def __post_init__(self):
        if self.results is None:
            self.results = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ProgressUpdate:
    operation_id: str
    progress: float
    message: str = ""
    partial_results: List[Any] = None
    timestamp: float = None

    def __post_init__(self):
        if self.partial_results is None:
            self.partial_results = []
        if self.timestamp is None:
            self.timestamp = time.time()


@dataclass
class Message:
    type: MessageType
    operation_id: Optional[str]
    data: Dict[str, Any]
    timestamp: float = None

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
