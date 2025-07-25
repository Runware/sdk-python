from typing import Any, Dict

from .base import BaseOperation
from ..core.cpu_bound import cpu_executor
from ..exceptions import RunwareOperationError
from ..logging_config import get_logger
from ..types import ETaskType, IUploadModelBaseType, IUploadModelResponse

logger = get_logger(__name__)


class ModelUploadOperation(BaseOperation):
    field_mappings = {
        "air": "air",
        "taskUUID": "taskUUID",
        "taskType": "taskType",
    }
    response_class = IUploadModelResponse

    def __init__(self, request: IUploadModelBaseType, client=None):
        super().__init__(operation_id=None, client=client)
        self.request = request
        self._processed_statuses = set()
        logger.info(f"Model upload operation {self.operation_id} initialized")

    @property
    def operation_type(self) -> str:
        return "modelUpload"

    async def execute(self) -> IUploadModelResponse:
        results = await super().execute()
        if results is not None:
            return results[0]
        return None

    def _setup_message_handlers(self):
        self._message_handlers = {
            "modelUpload": self._handle_model_upload,
            "error": self._handle_error_message,
        }

    async def _build_request_payload(self) -> list[Dict[str, Any]]:
        base_fields = await cpu_executor.serialize_dataclass(
            {
                "taskType": ETaskType.MODEL_UPLOAD.value,
                "taskUUID": self.operation_id,
                "air": self.request.air,
                "name": self.request.name,
                "downloadURL": self.request.downloadURL,
                "uniqueIdentifier": self.request.uniqueIdentifier,
                "version": self.request.version,
                "format": self.request.format,
                "private": self.request.private,
                "category": self.request.category,
                "architecture": self.request.architecture,
            }
        )

        optional_fields = [
            "retry",
            "heroImageURL",
            "tags",
            "shortDescription",
            "comment",
            "positiveTriggerWords",
            "type",
            "negativeTriggerWords",
            "defaultWeight",
            "defaultStrength",
            "defaultGuidanceScale",
            "defaultSteps",
            "defaultScheduler",
            "conditioning",
        ]

        request_object = {
            **base_fields,
            **{
                field: getattr(self.request, field)
                for field in optional_fields
                if getattr(self.request, field, None) is not None
            },
        }

        return [request_object]

    async def _handle_model_upload(self, message: Dict[str, Any]):
        try:
            logger.debug(
                f"Operation {self.operation_id} handling model upload message: {message}"
            )

            if message.get("code"):
                error = RunwareOperationError(
                    f"Model upload error: {message.get('message', 'Unknown error')}",
                    operation_id=self.operation_id,
                    operation_type=self.operation_type,
                    code=message.get("code"),
                )
                await self._handle_error(error)
                return

            status = message.get("status")
            if not status:
                logger.warning(
                    f"Operation {self.operation_id} received message without status"
                )
                return

            # Track unique statuses
            if status not in self._processed_statuses:
                self._processed_statuses.add(status)

            # Check for error in status
            if status and "error" in str(status).lower():
                error = RunwareOperationError(
                    f"Model upload failed with status: {status}",
                    operation_id=self.operation_id,
                    operation_type=self.operation_type,
                )
                await self._handle_error(error)
                return

            # Complete when status is "ready"
            if status == "ready":
                logger.info(f"Operation {self.operation_id} model upload ready")
                upload_response = self._parse_response(message)
                await self._complete_operation([upload_response])

        except Exception as e:
            logger.error(
                f"Error handling model upload message for operation {self.operation_id}",
                exc_info=e,
            )
            await self._handle_error(e)
