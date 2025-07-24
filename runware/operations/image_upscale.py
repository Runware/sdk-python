from typing import Any, Dict, List

from .base import BaseOperation
from ..core.cpu_bound import cpu_executor
from ..exceptions import RunwareOperationError
from ..logging_config import get_logger
from ..types import ETaskType, IImage, IImageUpscale

logger = get_logger(__name__)


class ImageUpscaleOperation(BaseOperation):
    field_mappings = {
        "taskType": "taskType",
        "imageUUID": "imageUUID",
        "taskUUID": "taskUUID",
        "inputImageUUID": "inputImageUUID",
        "imageURL": "imageURL",
        "imageBase64Data": "imageBase64Data",
        "imageDataURI": "imageDataURI",
        "cost": "cost",
    }
    response_class = IImage

    def __init__(self, request: IImageUpscale, client=None):
        super().__init__(operation_id=None, client=client)
        self.request = request
        self._image_uploaded = None

        logger.info(f"Image upscale operation {self.operation_id} initialized")

    @property
    def operation_type(self) -> str:
        return "imageUpscale"

    def _setup_message_handlers(self):
        self._message_handlers = {
            "imageUpscale": self._handle_image_upscale,
            "error": self._handle_error_message,
        }

    async def _build_request_payload(self) -> List[Dict[str, Any]]:
        logger.debug(f"Operation {self.operation_id} uploading image")
        self._image_uploaded = await self.client.uploadImage(self.request.inputImage)
        if not self._image_uploaded or not self._image_uploaded.imageUUID:
            raise RunwareOperationError(
                "Failed to upload image",
                operation_id=self.operation_id,
                operation_type=self.operation_type,
            )
        task_params = await cpu_executor.serialize_dataclass(
            {
                "taskType": ETaskType.IMAGE_UPSCALE.value,
                "taskUUID": self.operation_id,
                "inputImage": (
                    self._image_uploaded.imageUUID
                    if self._image_uploaded
                    else self.request.inputImage
                ),
                "upscaleFactor": self.request.upscaleFactor,
            }
        )

        optional_fields = {
            "outputType": self.request.outputType,
            "outputFormat": self.request.outputFormat,
            "includeCost": self.request.includeCost,
        }

        for key, value in optional_fields.items():
            if value is not None:
                task_params[key] = value

        return [task_params]

    async def _handle_image_upscale(self, message: Dict[str, Any]):
        try:
            logger.debug(
                f"Operation {self.operation_id} handling image upscale message: {message}"
            )
            image_data = self._parse_response(message)
            await self._complete_operation([image_data])

        except Exception as e:
            logger.error(
                f"Error handling image upscale message for operation {self.operation_id}",
                exc_info=e,
            )
            await self._handle_error(e)
