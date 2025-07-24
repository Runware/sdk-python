from typing import Any, Dict, List, Optional

from .base import BaseOperation
from ..core.cpu_bound import cpu_executor
from ..logging_config import get_logger
from ..types import ETaskType, IImage, IImageInference
from ..utils import process_image

logger = get_logger(__name__)


class ImageInferenceOperation(BaseOperation):
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

    def __init__(self, request: IImageInference, client=None):
        super().__init__(request.taskUUID, client)
        self.request = request
        self.expected_results = request.numberResults or 1
        self.received_results = 0
        self._processed_images: Dict[str, Any] = {}

        logger.info(f"Image inference operation {self.operation_id} initialized")
        logger.debug(
            f"Operation {self.operation_id} expects {self.expected_results} results"
        )

    @property
    def operation_type(self) -> str:
        return "imageInference"

    def _setup_message_handlers(self):
        """Setup message handlers for different message types"""
        self._message_handlers = {
            # Primary message types for image inference
            "imageInference": self._handle_image_inference,
            "imageGeneration": self._handle_image_inference,
            # Alternative message types the server might send
            "image_inference": self._handle_image_inference,
            "image_generation": self._handle_image_inference,
            "inference": self._handle_image_inference,
            "generation": self._handle_image_inference,
            # Error handlers
            "error": self._handle_error_message,
            "Error": self._handle_error_message,
        }

        logger.info(
            f"Operation {self.operation_id} setup message handlers: {list(self._message_handlers.keys())}"
        )

    async def handle_message(self, message: Dict[str, Any]):
        """Enhanced message handling with detailed logging"""
        logger.debug(f"Operation {self.operation_id} received message: {message}")

        try:
            message_type = message.get("taskType")
            logger.debug(f"Operation {self.operation_id} message type: {message_type}")

            handler = self._message_handlers.get(message_type)

            if handler:
                logger.debug(
                    f"Operation {self.operation_id} found handler for message type: {message_type}"
                )
                await handler(message)
            else:
                logger.warning(
                    f"Operation {self.operation_id} no handler for message type: {message_type}"
                )
                await self._handle_unknown_message(message)

        except Exception as e:
            logger.error(
                f"Error handling message for operation {self.operation_id}", exc_info=e
            )
            await self._handle_error(e)

    async def _handle_unknown_message(self, message: Dict[str, Any]):
        """Enhanced fallback handler for unknown message types"""
        logger.warning(f"Operation {self.operation_id} handling unknown message type")
        logger.debug(f"Operation {self.operation_id} unknown message: {message}")

        # Try to detect if this looks like an image inference result
        has_image_fields = any(
            field in message for field in ["imageUUID", "imageURL", "imageBase64Data"]
        )
        has_task_uuid = message.get("taskUUID") == self.operation_id

        if has_image_fields and has_task_uuid:
            logger.info(
                f"Operation {self.operation_id} unknown message looks like image result, treating as image inference"
            )
            await self._handle_image_inference(message)
        else:
            logger.warning(
                f"Operation {self.operation_id} unknown message doesn't look like image result, ignoring"
            )

    async def _build_request_payload(self) -> List[Dict[str, Any]]:
        logger.debug(f"Operation {self.operation_id} building request payload")

        control_net_data: List[Dict[str, Any]] = []

        # Process images
        try:
            if self.request.maskImage:
                logger.debug(f"Operation {self.operation_id} processing mask image")
                self.request.maskImage = await process_image(self.request.maskImage)
            if self.request.seedImage:
                logger.debug(f"Operation {self.operation_id} processing seed image")
                self.request.seedImage = await process_image(self.request.seedImage)
            if self.request.referenceImages:
                logger.debug(
                    f"Operation {self.operation_id} processing reference images"
                )
                self.request.referenceImages = await process_image(
                    self.request.referenceImages
                )
        except Exception as e:
            logger.error(
                f"Operation {self.operation_id} failed to process images", exc_info=e
            )
            raise

        # Process ControlNet with ThreadPoolExecutor
        if self.request.controlNet:
            logger.debug(
                f"Operation {self.operation_id} processing {len(self.request.controlNet)} ControlNet items"
            )
            try:
                # Prepare control data
                control_items = []
                for i, control_data in enumerate(self.request.controlNet):
                    if self.client:
                        image_uploaded = await self.client.uploadImage(
                            control_data.guideImage
                        )
                        if image_uploaded:
                            control_data.guideImage = image_uploaded.imageUUID
                            control_items.append(control_data)

                control_net_data = await cpu_executor.batch_serialize_dataclasses(
                    control_items
                )

            except Exception as e:
                logger.error(
                    f"Operation {self.operation_id} failed to process ControlNet",
                    exc_info=e,
                )
                raise

        # Process InstantID
        instant_id_data = {}
        if self.request.instantID:
            logger.debug(f"Operation {self.operation_id} processing InstantID")
            try:
                instant_id_data = {
                    k: v
                    for k, v in vars(self.request.instantID).items()
                    if v is not None
                }
                if "inputImage" in instant_id_data:
                    instant_id_data["inputImage"] = await process_image(
                        instant_id_data["inputImage"]
                    )
                if "poseImage" in instant_id_data:
                    instant_id_data["poseImage"] = await process_image(
                        instant_id_data["poseImage"]
                    )
            except Exception as e:
                logger.error(
                    f"Operation {self.operation_id} failed to process InstantID",
                    exc_info=e,
                )
                raise

        # Process IP Adapters
        ip_adapters_data = []
        if self.request.ipAdapters:
            logger.debug(
                f"Operation {self.operation_id} processing {len(self.request.ipAdapters)} IP Adapters"
            )
            try:
                for ip_adapter in self.request.ipAdapters:
                    ip_adapter_data = {
                        k: v for k, v in vars(ip_adapter).items() if v is not None
                    }
                    if "guideImage" in ip_adapter_data:
                        ip_adapter_data["guideImage"] = await process_image(
                            ip_adapter_data["guideImage"]
                        )
                    ip_adapters_data.append(ip_adapter_data)
            except Exception as e:
                logger.error(
                    f"Operation {self.operation_id} failed to process IP Adapters",
                    exc_info=e,
                )
                raise

        # Process ACE++
        ace_plus_plus_data = {}
        if self.request.acePlusPlus:
            logger.debug(f"Operation {self.operation_id} processing ACE++")
            try:
                ace_plus_plus_data = {
                    "inputImages": [],
                    "repaintingScale": self.request.acePlusPlus.repaintingScale,
                    "type": self.request.acePlusPlus.taskType,
                }
                if self.request.acePlusPlus.inputImages:
                    ace_plus_plus_data["inputImages"] = await process_image(
                        self.request.acePlusPlus.inputImages
                    )
                if self.request.acePlusPlus.inputMasks:
                    ace_plus_plus_data["inputMasks"] = await process_image(
                        self.request.acePlusPlus.inputMasks
                    )
            except Exception as e:
                logger.error(
                    f"Operation {self.operation_id} failed to process ACE++", exc_info=e
                )
                raise

        # Build main request object
        request_object = {
            "taskType": ETaskType.IMAGE_INFERENCE.value,
            "taskUUID": self.operation_id,
            "modelId": self.request.model,
            "positivePrompt": self.request.positivePrompt.strip(),
            "numberResults": self.expected_results,
        }

        # Add optional fields
        optional_fields = {
            "steps": self.request.steps,
            "height": self.request.height,
            "width": self.request.width,
            "controlNet": control_net_data if control_net_data else None,
            "lora": (
                [
                    {"model": lora.model, "weight": lora.weight}
                    for lora in self.request.lora
                ]
                if self.request.lora
                else None
            ),
            "lycoris": (
                [
                    {"model": lycoris.model, "weight": lycoris.weight}
                    for lycoris in self.request.lycoris
                ]
                if self.request.lycoris
                else None
            ),
            "embeddings": (
                [{"model": embedding.model} for embedding in self.request.embeddings]
                if self.request.embeddings
                else None
            ),
            "seed": self.request.seed,
            "refiner": self._build_refiner_data() if self.request.refiner else None,
            "instantID": instant_id_data if instant_id_data else None,
            "outpaint": (
                {k: v for k, v in vars(self.request.outpaint).items() if v is not None}
                if self.request.outpaint
                else None
            ),
            "ipAdapters": ip_adapters_data if ip_adapters_data else None,
            "acePlusPlus": ace_plus_plus_data if ace_plus_plus_data else None,
            "outputType": self.request.outputType,
            "outputFormat": self.request.outputFormat,
            "includeCost": self.request.includeCost,
            "checkNSFW": self.request.checkNsfw,
            "negativePrompt": self.request.negativePrompt,
            "CFGScale": self.request.CFGScale,
            "seedImage": self.request.seedImage,
            "maskImage": self.request.maskImage,
            "referenceImages": self.request.referenceImages,
            "strength": self.request.strength,
            "scheduler": self.request.scheduler,
            "vae": self.request.vae,
            "promptWeighting": self.request.promptWeighting,
            "maskMargin": self.request.maskMargin,
            "outputQuality": self.request.outputQuality,
        }

        # Add accelerator options
        if self.request.acceleratorOptions:
            pipeline_options_dict = {
                k: v
                for k, v in vars(self.request.acceleratorOptions).items()
                if v is not None
            }
            optional_fields["acceleratorOptions"] = pipeline_options_dict

        # Add advanced features
        if self.request.advancedFeatures:
            pipeline_options_dict = {
                k: v.__dict__
                for k, v in vars(self.request.advancedFeatures).items()
                if v is not None
            }
            optional_fields["advancedFeatures"] = pipeline_options_dict

        # Add non-null optional fields
        for key, value in optional_fields.items():
            if value is not None:
                request_object[key] = value

        # Add extra args
        if hasattr(self.request, "extraArgs") and isinstance(
            self.request.extraArgs, dict
        ):
            request_object.update(self.request.extraArgs)

        logger.debug(
            f"Operation {self.operation_id} built request payload: {request_object}"
        )
        return [request_object]

    def _build_refiner_data(self) -> Optional[Dict[str, Any]]:
        if not self.request.refiner:
            return None

        refiner_data = {"model": self.request.refiner.model}

        if self.request.refiner.startStep is not None:
            refiner_data["startStep"] = self.request.refiner.startStep

        if self.request.refiner.startStepPercentage is not None:
            refiner_data["startStepPercentage"] = (
                self.request.refiner.startStepPercentage
            )

        return refiner_data

    async def _handle_image_inference(self, message: Dict[str, Any]):
        try:
            logger.debug(
                f"Operation {self.operation_id} handling image inference message: {message}"
            )

            image_uuid = message.get("imageUUID")
            if not image_uuid:
                logger.warning(
                    f"Operation {self.operation_id} image inference message missing imageUUID: {message}"
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

            # Call user callback if provided
            if self.request.onPartialImages:
                try:
                    self.request.onPartialImages([image_data], None)
                except Exception as e:
                    logger.error(f"Error in onPartialImages callback", exc_info=e)

            # Check if we have all expected results
            if self.received_results >= self.expected_results:
                logger.info(
                    f"Operation {self.operation_id} received all expected results, completing"
                )
                await self._complete_operation()

        except Exception as e:
            logger.error(
                f"Error handling image inference message for operation {self.operation_id}",
                exc_info=e,
            )
            await self._handle_error(e)
