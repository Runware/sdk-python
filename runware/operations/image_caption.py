from typing import Any, Dict

from .base import BaseOperation
from ..core.cpu_bound import cpu_executor
from ..logging_config import get_logger
from ..types import ETaskType, IImageCaption, IImageToText

logger = get_logger(__name__)


class ImageCaptionOperation(BaseOperation):
    field_mappings = {
        "taskType": "taskType",
        "taskUUID": "taskUUID",
        "text": "text",
        "cost": "cost",
    }
    response_class = IImageToText

    def __init__(self, request: IImageCaption, client=None):
        super().__init__(operation_id=None, client=client)
        self.request = request
        self._image_uploaded = None

        logger.info(f"Image caption operation {self.operation_id} initialized")

    @property
    def operation_type(self) -> str:
        return "imageCaption"

    async def execute(self) -> IImageToText | None:
        results = await super().execute()
        if results is not None:
            return results[0]
        return None

    def _setup_message_handlers(self):
        self._message_handlers = {
            "imageCaption": self._handle_image_caption,
            "error": self._handle_error_message,
        }

    async def _build_request_payload(self) -> list[Dict[str, Any]]:
        self._image_uploaded = await self.client.uploadImage(self.request.inputImage)
        task_params = await cpu_executor.serialize_dataclass(
            {
                "taskType": ETaskType.IMAGE_CAPTION.value,
                "taskUUID": self.operation_id,
                "inputImage": (
                    self._image_uploaded.imageUUID
                    if self._image_uploaded
                    else self.request.inputImage
                ),
            }
        )

        if self.request.includeCost:
            task_params["includeCost"] = self.request.includeCost

        return [task_params]

    async def _handle_image_caption(self, message: Dict[str, Any]):
        try:
            logger.debug(
                f"Operation {self.operation_id} handling image caption message: {message}"
            )
            image_to_text = self._parse_response(message)
            await self._complete_operation([image_to_text])

        except Exception as e:
            logger.error(
                f"Error handling image caption message for operation {self.operation_id}",
                exc_info=e,
            )
            await self._handle_error(e)
