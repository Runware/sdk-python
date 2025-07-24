from typing import Any, Dict, List

from .base import BaseOperation
from ..core.cpu_bound import cpu_executor
from ..logging_config import get_logger
from ..types import ETaskType, IImage, IImageBackgroundRemoval

logger = get_logger(__name__)


class ImageBackgroundRemovalOperation(BaseOperation):
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

    def __init__(self, request: IImageBackgroundRemoval, client=None):
        super().__init__(request.taskUUID, client)
        self.request = request
        self._image_uploaded = None

        logger.info(
            f"Image background removal operation {self.operation_id} initialized"
        )

    @property
    def operation_type(self) -> str:
        return "imageBackgroundRemoval"

    def _setup_message_handlers(self):
        self._message_handlers = {
            "imageBackgroundRemoval": self._handle_background_removal,
            "error": self._handle_error_message,
        }

    async def _build_request_payload(self) -> List[Dict[str, Any]]:
        self._image_uploaded = await self.client.uploadImage(self.request.inputImage)
        task_params = await cpu_executor.serialize_dataclass(
            {
                "taskType": ETaskType.IMAGE_BACKGROUND_REMOVAL.value,
                "taskUUID": self.operation_id,
                "inputImage": (
                    self._image_uploaded.imageUUID
                    if self._image_uploaded
                    else self.request.inputImage
                ),
            }
        )

        optional_fields = {
            "outputType": self.request.outputType,
            "outputFormat": self.request.outputFormat,
            "includeCost": self.request.includeCost,
            "model": self.request.model,
            "outputQuality": self.request.outputQuality,
        }

        for key, value in optional_fields.items():
            if value is not None:
                task_params[key] = value

        if self.request.settings:
            settings_dict = {
                k: v for k, v in vars(self.request.settings).items() if v is not None
            }
            task_params.update(settings_dict)

        return [task_params]

    async def _handle_background_removal(self, message: Dict[str, Any]):
        try:
            logger.debug(
                f"Operation {self.operation_id} handling background removal message: {message}"
            )
            image_data = self._parse_response(message)
            await self._complete_operation([image_data])

        except Exception as e:
            logger.error(
                f"Error handling background removal message for operation {self.operation_id}",
                exc_info=e,
            )
            await self._handle_error(e)
