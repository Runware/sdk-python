from typing import Any, Dict, List

from .base import BaseOperation
from ..core.cpu_bound import cpu_executor
from ..logging_config import get_logger
from ..types import ETaskType, IImage, IPhotoMaker
from ..utils import process_image

logger = get_logger(__name__)


class PhotoMakerOperation(BaseOperation):
    field_mappings = {
        "taskType": "taskType",
        "imageUUID": "imageUUID",
        "taskUUID": "taskUUID",
        "seed": "seed",
        "inputImageUUID": "inputImageUUID",
        "imageURL": "imageURL",
        "imageBase64Data": "imageBase64Data",
        "imageDataURI": "imageDataURI",
        "NSFWContent": "NSFWContent",
        "cost": "cost",
    }
    response_class = IImage

    def __init__(self, request: IPhotoMaker, client=None):
        super().__init__(request.taskUUID, client)
        self.request = request
        self.expected_results = request.numberResults
        self.received_results = 0
        self._processed_images: Dict[str, Any] = {}

        logger.info(f"Photo maker operation {self.operation_id} initialized")
        logger.debug(
            f"Operation {self.operation_id} expects {self.expected_results} results"
        )

    @property
    def operation_type(self) -> str:
        return "photoMaker"

    def _setup_message_handlers(self):
        self._message_handlers = {
            "photoMaker": self._handle_photo_maker,
            "error": self._handle_error_message,
        }

    async def _process_input_images(self):
        if self.request.inputImages:
            try:
                processed_images = []
                for image in self.request.inputImages:
                    processed_image = await process_image(image)
                    processed_images.append(processed_image)
                self.request.inputImages = processed_images
            except Exception as e:
                logger.error(
                    f"Operation {self.operation_id} failed to process input images",
                    exc_info=e,
                )
                raise

    async def _build_request_payload(self) -> List[Dict[str, Any]]:
        await self._process_input_images()
        request_object = await cpu_executor.serialize_dataclass(
            {
                "taskUUID": self.operation_id,
                "model": self.request.model,
                "positivePrompt": self.request.positivePrompt.strip(),
                "numberResults": self.expected_results,
                "height": self.request.height,
                "width": self.request.width,
                "taskType": ETaskType.PHOTO_MAKER.value,
                "style": self.request.style,
                "strength": self.request.strength,
            }
        )

        if self.request.inputImages:
            request_object["inputImages"] = self.request.inputImages

        optional_fields = {
            "steps": self.request.steps,
            "outputFormat": self.request.outputFormat,
            "includeCost": self.request.includeCost,
            "outputType": self.request.outputType,
        }

        for key, value in optional_fields.items():
            if value is not None:
                request_object[key] = value

        return [request_object]

    async def _handle_photo_maker(self, message: Dict[str, Any]):
        try:
            logger.debug(
                f"Operation {self.operation_id} handling photo maker message: {message}"
            )

            image_uuid = message.get("imageUUID")
            if not image_uuid:
                logger.warning(
                    f"Operation {self.operation_id} photo maker message missing imageUUID: {message}"
                )
                return

            if image_uuid in self._processed_images:
                logger.debug(
                    f"Operation {self.operation_id} already processed image {image_uuid}"
                )
                return

            logger.info(
                f"Operation {self.operation_id} processing new image {image_uuid}"
            )

            image_data = self._parse_response(message)
            self._processed_images[image_uuid] = image_data
            self.received_results += 1

            progress = min(self.received_results / self.expected_results, 1.0)
            await self._update_progress(progress, partial_results=[image_data])

            if self.received_results >= self.expected_results:
                logger.info(
                    f"Operation {self.operation_id} received all expected results"
                )
                await self._complete_operation()

        except Exception as e:
            logger.error(
                f"Error handling photo maker message for operation {self.operation_id}",
                exc_info=e,
            )
            await self._handle_error(e)
