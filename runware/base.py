import asyncio
from dataclasses import asdict
from doctest import debug
import json
from os import error
import os
import re
import base64
import uuid
import inspect
from typing import List, Union, Optional, Callable, Any, Dict
from urllib.parse import urlparse

from .utils import (
    BASE_RUNWARE_URLS,
    delay,
    getUUID,
    removeListener,
    accessDeepObject,
    getPreprocessorType,
    getTaskType,
    fileToBase64,
    isValidUUID,
    createImageFromResponse,
    createImageToTextFromResponse,
    createEnhancedPromptsFromResponse,
    instantiateDataclassList,
    RunwareAPIError,
    RunwareError,
)
from .async_retry import asyncRetry
from .types import (
    Environment,
    SdkType,
    RunwareBaseType,
    IImage,
    ILora,
    EControlMode,
    IControlNetGeneral,
    IControlNetA,
    IControlNetCanny,
    IControlNetHandsAndFace,
    IControlNet,
    IControlNetWithUUID,
    IError,
    IImageInference,
    IPhotoMaker,
    IImageCaption,
    IImageToText,
    IImageBackgroundRemoval,
    IPromptEnhance,
    IEnhancedPrompt,
    IImageUpscale,
    IUploadModelBaseType,
    IUploadModelResponse,
    ReconnectingWebsocketProps,
    UploadImageType,
    GetWithPromiseCallBackType,
    EPreProcessorGroup,
    EPreProcessor,
    EOpenPosePreProcessor,
    RequireAtLeastOne,
    RequireOnlyOne,
    ListenerType,
    File,
    ETaskType, IControlNetBaseWithUUID, IControlNetCannyWithUUID, IControlNetHandsAndFaceWithUUID, IControlNetAWithUUID,
)

from typing import List, Optional, Union, Callable, Any, Dict
from .types import IImage, IError, SdkType, ListenerType
from .utils import (
    accessDeepObject,
    getIntervalWithPromise,
    removeListener,
    LISTEN_TO_IMAGES_KEY,
)

import logging

from .logging_config import configure_logging

# Configure logging
# configure_logging(log_level=logging.CRITICAL)

logger = logging.getLogger(__name__)


class RunwareBase:
    def __init__(
            self, api_key: str, url: str = BASE_RUNWARE_URLS[Environment.PRODUCTION]
    ):
        self._ws: Optional[ReconnectingWebsocketProps] = None
        self._listeners: List[ListenerType] = []
        self._apiKey: str = api_key
        self._url: Optional[str] = url
        self._globalMessages: Dict[str, Any] = {}
        self._globalImages: List[IImage] = []
        self._globalError: Optional[IError] = None
        self._connectionSessionUUID: Optional[str] = None
        self._invalidAPIkey: Optional[str] = None
        self._sdkType: SdkType = SdkType.SERVER

    def isWebsocketReadyState(self) -> bool:
        return self._ws and self._ws.open

    def isAuthenticated(self):
        return self._connectionSessionUUID is not None

    def addListener(
            self,
            lis: Callable[[Any], Any],
            check: Callable[[Any], Any],
            groupKey: Optional[str] = None,
    ) -> Dict[str, Callable[[], None]]:
        # Get the current frame
        current_frame = inspect.currentframe()

        # Get the caller's frame
        caller_frame = current_frame.f_back

        # Get the caller's function name
        caller_name = caller_frame.f_code.co_name

        # Get the caller's line number
        caller_line_number = caller_frame.f_lineno

        debug_message = f"Listener {self.addListener.__name__} created by {caller_name} at line {caller_line_number} with listener: {lis} and check: {check}"
        # logger.debug(debug_message)

        if not lis or not check:
            raise ValueError("Listener and check functions are required")

        def listener(msg: Any) -> None:
            if not lis or not check:
                raise ValueError("Listener and check functions are required")
            if msg.get("error"):
                lis(msg)
            elif check(msg):
                lis(msg)

        groupListener: ListenerType = ListenerType(
            key=getUUID(),
            listener=listener,
            group_key=groupKey,
            debug_message=debug_message,
        )
        self._listeners.append(groupListener)

        def destroy() -> None:
            self._listeners = removeListener(self._listeners, groupListener)

        return {"destroy": destroy}

    def handle_connection_response(self, m):
        if m.get("error"):
            if m["errorId"] == 19:
                self._invalidAPIkey = "Invalid API key"
            else:
                self._invalidAPIkey = "Error connection"
            return
        self._connectionSessionUUID = m.get("newConnectionSessionUUID", {}).get(
            "connectionSessionUUID"
        )
        self._invalidAPIkey = None

    async def photoMaker(self, requestPhotoMaker: IPhotoMaker):
        retry_count = 0

        try:
            await self.ensureConnection()

            task_uuid = requestPhotoMaker.taskUUID or getUUID()
            requestPhotoMaker.taskUUID = task_uuid

            for i, image in enumerate(requestPhotoMaker.inputImages):
                if self._isLocalFile(image) and not str(image).startswith("http"):
                    requestPhotoMaker.inputImages[i] = await fileToBase64(image)

            prompt = f"{requestPhotoMaker.positivePrompt}".strip()
            request_object = {
                "taskUUID": requestPhotoMaker.taskUUID,
                "model": requestPhotoMaker.model,
                "positivePrompt": prompt,
                "numberResults": requestPhotoMaker.numberResults,
                "height": requestPhotoMaker.height,
                "width": requestPhotoMaker.width,
                "taskType": ETaskType.PHOTO_MAKER.value,
                "style": requestPhotoMaker.style,
                "strength": requestPhotoMaker.strength,
                **({"inputImages": requestPhotoMaker.inputImages} if requestPhotoMaker.inputImages else {}),
                **({"steps": requestPhotoMaker.steps} if requestPhotoMaker.steps else {}),
            }

            if requestPhotoMaker.outputFormat is not None:
                request_object["outputFormat"] = requestPhotoMaker.outputFormat
            if requestPhotoMaker.includeCost:
                request_object["includeCost"] = requestPhotoMaker.includeCost
            if requestPhotoMaker.outputType:
                request_object["outputType"] = requestPhotoMaker.outputType

            await self.send(
                [
                    request_object
                ]
            )

            lis = self.globalListener(
                taskUUID=task_uuid,
            )

            numberOfResults = requestPhotoMaker.numberResults

            def check(resolve: callable, reject: callable, *args: Any) -> bool:
                photo_maker_list = self._globalMessages.get(task_uuid, [])
                unique_results = {}

                for made_photo in photo_maker_list:
                    if made_photo.get("code"):
                        raise RunwareAPIError(made_photo)

                    if made_photo.get("taskType") != "photoMaker":
                        continue

                    image_uuid = made_photo.get("imageUUID")
                    if image_uuid not in unique_results:
                        unique_results[image_uuid] = made_photo

                if len(unique_results) >= numberOfResults:
                    del self._globalMessages[task_uuid]
                    resolve(list(unique_results.values()))
                    return True

                return False

            response = await getIntervalWithPromise(check, debugKey="photo-maker")

            lis["destroy"]()

            if "code" in response:
                # This indicates an error response
                raise RunwareAPIError(response)

            if response:
                if not isinstance(response, list):
                    response = [response]

            return instantiateDataclassList(IImage, response)

        except Exception as e:
            if retry_count >= 2:
                self.logger.error(f"Error in photoMaker request: {e}")
                exit()
                return self.handle_incomplete_images(task_uuids=task_uuids, error=e)
            else:
                raise e

    def create_control_net_with_uuid(self, data: Dict) -> IControlNetBaseWithUUID:
        # Determine the class based on data keys or attributes
        if "low_threshold_canny" in data and "high_threshold_canny" in data:
            return IControlNetCannyWithUUID(**data)
        elif "include_hands_and_face_open_pose" in data:
            return IControlNetHandsAndFaceWithUUID(**data)
        else:
            return IControlNetAWithUUID(**data)

    async def imageInference(
            self, requestImage: IImageInference
    ) -> Union[List[IImage], None]:
        let_lis: Optional[Any] = None
        request_object: Optional[Dict[str, Any]] = None
        task_uuids: List[str] = []
        retry_count = 0

        try:
            await self.ensureConnection()
            control_net_data: List[IControlNetWithUUID] = []

            if requestImage.maskImage:
                if self._isLocalFile(requestImage.maskImage):
                    if not requestImage.maskImage.startswith("http"):
                        requestImage.maskImage = await fileToBase64(requestImage.maskImage)

            if requestImage.seedImage:
                if self._isLocalFile(requestImage.seedImage):
                    if not requestImage.seedImage.startswith("http"):
                        requestImage.seedImage = await fileToBase64(requestImage.seedImage)

            if requestImage.controlNet:
                for control_data in requestImage.controlNet:
                    any_control_data = (
                        control_data  # Type cast to access additional attributes
                    )
                    preprocessor = control_data.preprocessor
                    end_step = control_data.end_step
                    start_step = control_data.start_step
                    weight = control_data.weight
                    guide_image = control_data.guide_image
                    guide_image_unprocessed = control_data.guide_image_unprocessed
                    control_mode = control_data.control_mode

                    def get_canny_object() -> Dict[str, int]:
                        if control_data.preprocessor == "canny":
                            return {
                                "low_threshold_canny": any_control_data.low_threshold_canny,
                                "high_threshold_canny": any_control_data.high_threshold_canny,
                            }
                        else:
                            return {}

                    image_uploaded = await (
                        self.uploadUnprocessedImage(
                            file=guide_image_unprocessed,
                            preProcessorType=getPreprocessorType(preprocessor),
                            includeHandsAndFaceOpenPose=any_control_data.include_hands_and_face_open_pose,
                            **get_canny_object(),
                        )
                        if guide_image_unprocessed
                        else self.uploadImage(guide_image)
                    )

                    if not image_uploaded:
                        return []

                    control_net_common_data = {
                        "guide_image_uuid": image_uploaded.imageUUID,
                        "end_step": end_step,
                        "preprocessor": preprocessor.value,
                        "start_step": start_step,
                        "guide_image": guide_image,
                        "guide_image_unprocessed": guide_image_unprocessed,
                        "weight": weight,
                        "control_mode": control_mode or EControlMode.CONTROL_NET,
                        **get_canny_object(),
                    }

                    control_net_instance = self.create_control_net_with_uuid(control_net_common_data)
                    control_net_data.append(control_net_instance)

            prompt = f"{requestImage.positivePrompt}".strip()

            control_net_data_dicts = [asdict(item) for item in control_net_data]

            request_object = {
                "offset": 0,
                "taskUUID": requestImage.taskUUID,
                "modelId": requestImage.model,
                "positivePrompt": prompt,
                "numberResults": requestImage.numberResults,
                "height": requestImage.height,
                "width": requestImage.width,
                "taskType": ETaskType.IMAGE_INFERENCE.value,
                **({"steps": requestImage.steps} if requestImage.steps else {}),
                **({"controlNet": control_net_data_dicts} if control_net_data_dicts else {}),
                **(
                    {
                        "lora": [
                            {"model": lora.model, "weight": lora.weight}
                            for lora in requestImage.lora
                        ]
                    }
                    if requestImage.lora
                    else {}
                ),
                **(
                    {
                        "embeddings": [
                            {"model": embedding.model}
                            for embedding in requestImage.embeddings
                        ]
                    }
                    if requestImage.embeddings
                    else {}
                ),
                **({"seed": requestImage.seed} if requestImage.seed else {}),
                **(
                    {
                        "refiner": {
                            "model": requestImage.refiner.model,
                            **(
                                {"startStep": requestImage.refiner.startStep}
                                if requestImage.refiner.startStep is not None
                                else {}
                            ),
                            **(
                                {"startStepPercentage": requestImage.refiner.startStepPercentage}
                                if requestImage.refiner.startStepPercentage is not None
                                else {}
                            ),
                        }
                    } if requestImage.refiner else {}
                ),
            }

            # Add optional parameters if they are provided
            if requestImage.outputType is not None:
                request_object["outputType"] = requestImage.outputType
            if requestImage.outputFormat is not None:
                request_object["outputFormat"] = requestImage.outputFormat
            if requestImage.includeCost:
                request_object["includeCost"] = requestImage.includeCost
            if requestImage.checkNsfw:
                request_object["checkNSFW"] = requestImage.checkNsfw

            if requestImage.negativePrompt:
                request_object["negativePrompt"] = requestImage.negativePrompt
            if requestImage.CFGScale:
                request_object["CFGScale"] = requestImage.CFGScale
            if requestImage.seedImage:
                request_object["seedImage"] = requestImage.seedImage
            if requestImage.maskImage:
                request_object["maskImage"] = requestImage.maskImage
            if requestImage.strength:
                request_object["strength"] = requestImage.strength
            if requestImage.scheduler:
                request_object["scheduler"] = requestImage.scheduler
            if requestImage.vae:
                request_object["vae"] = requestImage.vae

            return await asyncRetry(
                lambda: self._requestImages(
                    request_object=request_object,
                    task_uuids=task_uuids,
                    let_lis=let_lis,
                    retry_count=retry_count,
                    number_of_images=requestImage.numberResults,
                    on_partial_images=requestImage.onPartialImages,
                )
            )
        except Exception as e:
            if retry_count >= 2:
                self.logger.error(f"Error in requestImages: {e}")
                exit()
                return self.handle_incomplete_images(task_uuids=task_uuids, error=e)
            else:
                raise e

    async def _requestImages(
            self,
            request_object: Dict[str, Any],
            task_uuids: List[str],
            let_lis: Optional[Any],
            retry_count: int,
            number_of_images: int,
            on_partial_images: Optional[Callable[[List[IImage], Optional[IError]], None]],
    ) -> List[IImage]:
        retry_count += 1
        if let_lis:
            let_lis["destroy"]()
        images_with_similar_task = [
            img for img in self._globalImages if img.get("taskUUID") in task_uuids
        ]

        task_uuid = request_object.get("taskUUID")
        if task_uuid is None:
            task_uuid = getUUID()

        task_uuids.append(task_uuid)

        image_remaining = number_of_images - len(images_with_similar_task)
        new_request_object = {
            "newTask": {
                **request_object,
                "taskUUID": task_uuid,
                "numberResults": image_remaining,
            }
        }
        await self.send(new_request_object)

        let_lis = await self.listenToImages(
            onPartialImages=on_partial_images,
            taskUUID=task_uuid,
            groupKey=LISTEN_TO_IMAGES_KEY.REQUEST_IMAGES,
        )
        images = await self.getSimililarImage(
            taskUUID=task_uuids,
            numberOfImages=number_of_images,
            lis=let_lis,
        )

        let_lis["destroy"]()
        # TODO: NameError("name 'image_path' is not defined"). I think I remove the images when I have onPartialImages
        if images:
            if "code" in images:
                # This indicates an error response
                raise RunwareAPIError(images)

            return instantiateDataclassList(IImage, images)

        # return images

    async def imageCaption(self, requestImageToText: IImageCaption) -> IImageToText:
        try:
            await self.ensureConnection()
            return await asyncRetry(
                lambda: self._requestImageToText(requestImageToText)
            )
        except Exception as e:
            raise e

    async def _requestImageToText(
            self, requestImageToText: IImageCaption
    ) -> IImageToText:
        inputImage = requestImageToText.inputImage

        image_uploaded = await self.uploadImage(inputImage)

        if not image_uploaded or not image_uploaded.imageUUID:
            return None

        taskUUID = getUUID()

        # Create a dictionary with mandatory parameters
        task_params = {
            "taskType": ETaskType.IMAGE_CAPTION.value,
            "taskUUID": taskUUID,
            "inputImage": image_uploaded.imageUUID,
        }

        # Add optional parameters if they are provided
        if requestImageToText.includeCost:
            task_params["includeCost"] = requestImageToText.includeCost

        # Send the task with all applicable parameters
        await self.send([task_params])

        lis = self.globalListener(
            taskUUID=taskUUID,
        )

        def check(resolve: callable, reject: callable, *args: Any) -> bool:
            response = self._globalMessages.get(taskUUID)
            # TODO: Check why I need a conversion here?
            if response:
                image_to_text = response[0]
            else:
                image_to_text = response
            if image_to_text and image_to_text.get("error"):
                reject(image_to_text)
                return True

            if image_to_text:
                del self._globalMessages[taskUUID]
                resolve(image_to_text)
                return True

            return False

        response = await getIntervalWithPromise(check, debugKey="image-to-text")

        lis["destroy"]()

        if "code" in response:
            # This indicates an error response
            raise RunwareAPIError(response)

        if response:
            return createImageToTextFromResponse(response)
        else:
            return None

    async def imageBackgroundRemoval(
            self, removeImageBackgroundPayload: IImageBackgroundRemoval
    ) -> List[IImage]:
        try:
            await self.ensureConnection()
            return await asyncRetry(
                lambda: self._removeImageBackground(removeImageBackgroundPayload)
            )
        except Exception as e:
            raise e

    async def _removeImageBackground(
            self, removeImageBackgroundPayload: IImageBackgroundRemoval
    ) -> List[IImage]:
        inputImage = removeImageBackgroundPayload.inputImage

        image_uploaded = await self.uploadImage(inputImage)

        if not image_uploaded or not image_uploaded.imageUUID:
            return []

        taskUUID = getUUID()

        # Create a dictionary with mandatory parameters
        task_params = {
            "taskType": ETaskType.IMAGE_BACKGROUND_REMOVAL.value,
            "taskUUID": taskUUID,
            "inputImage": image_uploaded.imageUUID,
        }

        # Add optional parameters if they are provided
        if removeImageBackgroundPayload.outputType is not None:
            task_params["outputType"] = removeImageBackgroundPayload.outputType
        if removeImageBackgroundPayload.outputFormat is not None:
            task_params["outputFormat"] = removeImageBackgroundPayload.outputFormat
        if removeImageBackgroundPayload.rgba:
            task_params["rgba"] = removeImageBackgroundPayload.rgba
        if removeImageBackgroundPayload.postProcessMask:
            task_params["postProcessMask"] = (
                removeImageBackgroundPayload.postProcessMask
            )
        if removeImageBackgroundPayload.returnOnlyMask:
            task_params["returnOnlyMask"] = removeImageBackgroundPayload.returnOnlyMask
        if removeImageBackgroundPayload.alphaMatting:
            task_params["alphaMatting"] = removeImageBackgroundPayload.alphaMatting
        if removeImageBackgroundPayload.alphaMattingForegroundThreshold is not None:
            task_params["alphaMattingForegroundThreshold"] = (
                removeImageBackgroundPayload.alphaMattingForegroundThreshold
            )
        if removeImageBackgroundPayload.alphaMattingBackgroundThreshold is not None:
            task_params["alphaMattingBackgroundThreshold"] = (
                removeImageBackgroundPayload.alphaMattingBackgroundThreshold
            )
        if removeImageBackgroundPayload.alphaMattingErodeSize is not None:
            task_params["alphaMattingErodeSize"] = (
                removeImageBackgroundPayload.alphaMattingErodeSize
            )
        if removeImageBackgroundPayload.includeCost:
            task_params["includeCost"] = removeImageBackgroundPayload.includeCost

        # Send the task with all applicable parameters
        await self.send([task_params])

        lis = self.globalListener(
            taskUUID=taskUUID,
        )

        def check(resolve: callable, reject: callable, *args: Any) -> bool:
            response = self._globalMessages.get(taskUUID)
            # TODO: Check why I need a conversion here?
            if response:
                new_remove_background = response[0]
            else:
                new_remove_background = response
            if new_remove_background and new_remove_background.get("error"):
                reject(new_remove_background)
                return True

            if new_remove_background:
                del self._globalMessages[taskUUID]
                resolve(new_remove_background)
                return True

            return False

        response = await getIntervalWithPromise(
            check, debugKey="remove-image-background"
        )

        lis["destroy"]()

        if "code" in response:
            # This indicates an error response
            raise RunwareAPIError(response)

        image = createImageFromResponse(response)
        image_list: List[IImage] = [image]

        return image_list

    async def imageUpscale(self, upscaleGanPayload: IImageUpscale) -> List[IImage]:
        try:
            await self.ensureConnection()
            return await asyncRetry(lambda: self._upscaleGan(upscaleGanPayload))
        except Exception as e:
            raise e

    async def _upscaleGan(self, upscaleGanPayload: IImageUpscale) -> List[IImage]:
        inputImage = upscaleGanPayload.inputImage
        upscaleFactor = upscaleGanPayload.upscaleFactor

        image_uploaded = await self.uploadImage(inputImage)

        if not image_uploaded or not image_uploaded.imageUUID:
            return []

        taskUUID = getUUID()

        # Create a dictionary with mandatory parameters
        task_params = {
            "taskType": ETaskType.IMAGE_UPSCALE.value,
            "taskUUID": taskUUID,
            "inputImage": image_uploaded.imageUUID,
            "upscaleFactor": upscaleGanPayload.upscaleFactor,
        }

        # Add optional parameters if they are provided
        if upscaleGanPayload.outputType is not None:
            task_params["outputType"] = upscaleGanPayload.outputType
        if upscaleGanPayload.outputFormat is not None:
            task_params["outputFormat"] = upscaleGanPayload.outputFormat
        if upscaleGanPayload.includeCost:
            task_params["includeCost"] = upscaleGanPayload.includeCost

        # Send the task with all applicable parameters
        await self.send([task_params])

        lis = self.globalListener(
            taskUUID=taskUUID,
        )

        def check(resolve: callable, reject: callable, *args: Any) -> bool:
            response = self._globalMessages.get(taskUUID)
            # TODO: Check why I need a conversion here?
            if response:
                upscaled_image = response[0]
            else:
                upscaled_image = response
            if upscaled_image and upscaled_image.get("error"):
                reject(upscaled_image)
                return True

            if upscaled_image:
                del self._globalMessages[taskUUID]
                resolve(upscaled_image)
                return True

            return False

        response = await getIntervalWithPromise(check, debugKey="upscale-gan")

        lis["destroy"]()

        if "code" in response:
            # This indicates an error response
            raise RunwareAPIError(response)

        image = createImageFromResponse(response)
        # TODO: The respones has an upscaleImageUUID field, should I return it as well?
        image_list: List[IImage] = [image]
        return image_list

    async def promptEnhance(
            self, promptEnhancer: IPromptEnhance
    ) -> List[IEnhancedPrompt]:
        """
        Enhance the given prompt by generating multiple versions of it.

        :param promptEnhancer: An IPromptEnhancer object containing the prompt details.
        :return: A list of IEnhancedPrompt objects representing the enhanced versions of the prompt.
        :raises: Any error that occurs during the enhancement process.
        """
        try:
            await self.ensureConnection()
            return await asyncRetry(lambda: self._enhancePrompt(promptEnhancer))
        except Exception as e:
            raise e

    async def _enhancePrompt(
            self, promptEnhancer: IPromptEnhance
    ) -> List[IEnhancedPrompt]:
        """
        Internal method to perform the actual prompt enhancement.

        :param promptEnhancer: An IPromptEnhancer object containing the prompt details.
        :return: A list of IEnhancedPrompt objects representing the enhanced versions of the prompt.
        """
        prompt = promptEnhancer.prompt
        promptMaxLength = getattr(promptEnhancer, 'promptMaxLength', 380)

        promptVersions = promptEnhancer.promptVersions or 1

        taskUUID = getUUID()

        # Create a dictionary with mandatory parameters
        task_params = {
            "taskType": ETaskType.PROMPT_ENHANCE.value,
            "taskUUID": taskUUID,
            "prompt": prompt,
            "promptMaxLength": promptMaxLength,
            "promptVersions": promptVersions,
        }

        # Add optional parameters if they are provided
        if promptEnhancer.includeCost:
            task_params["includeCost"] = promptEnhancer.includeCost

        # Send the task with all applicable parameters
        await self.send([task_params])

        lis = self.globalListener(
            taskUUID=taskUUID,
        )

        def check(resolve: Any, reject: Any, *args: Any) -> bool:
            response = self._globalMessages.get(taskUUID)
            if isinstance(response, dict) and response.get("error"):
                reject(response)
                return True
            # if response and len(response) >= promptVersions:
            if response:
                del self._globalMessages[taskUUID]
                resolve(response)
                return True

            return False

        response = await getIntervalWithPromise(check, debugKey="enhance-prompt")

        lis["destroy"]()

        if "code" in response[0]:
            # This indicates an error response
            raise RunwareAPIError(response[0])

        # Transform the response to a list of IEnhancedPrompt objects
        enhanced_prompts = createEnhancedPromptsFromResponse(response)

        return list(set(enhanced_prompts))

    async def uploadImage(self, file: Union[File, str]) -> Optional[UploadImageType]:
        try:
            await self.ensureConnection()
            return await asyncRetry(lambda: self._uploadImage(file))
        except Exception as e:
            raise e

    def _isLocalFile(self, file):
        if os.path.isfile(file):
            return True

        # Check if the string is a valid UUID
        if isValidUUID(file):
            return False

        # Check if the string is a valid URL
        parsed_url = urlparse(file)
        if parsed_url.scheme and parsed_url.netloc:
            return False  # Use the URL as is
        else:
            # Handle case with no scheme and no netloc
            if not parsed_url.scheme and not parsed_url.netloc or parsed_url.scheme == 'data':
                # Check if it's a base64 string (with or without data URI prefix)
                if file.startswith("data:") or re.match(r"^[A-Za-z0-9+/]+={0,2}$", file):
                    # Assume it's a base64 string (with or without data URI prefix)
                    return False

                # Assume it's a URL without scheme (e.g., 'example.com/some/path')
                # Add 'https://' in front and treat it as a valid URL
                file = f"https://{file}"
                parsed_url = urlparse(file)
                if parsed_url.netloc:  # Now it should have a valid netloc
                    return False
                else:
                    raise FileNotFoundError(f"File or URL '{file}' not found.")

        raise FileNotFoundError(f"File or URL '{file}' not valid or not found.")

    async def _uploadImage(self, file: Union[File, str]) -> Optional[UploadImageType]:
        task_uuid = getUUID()
        local_file = True
        if isinstance(file, str):
            if os.path.exists(file):
                local_file = True
            else:
                local_file = self._isLocalFile(file)

                # Check if it's a base64 string (with or without data URI prefix)
                if file.startswith("data:") or re.match(r"^[A-Za-z0-9+/]+={0,2}$", file):
                    # Assume it's a base64 string (with or without data URI prefix)
                    local_file = False

        if not local_file:
            return UploadImageType(
                imageUUID=file,
                imageURL=file,
                taskUUID=task_uuid,
            )

        image_base64 = await fileToBase64(file) if isinstance(file, str) else file

        await self.send(
            [
                {
                    "taskType": ETaskType.IMAGE_UPLOAD.value,
                    "taskUUID": task_uuid,
                    "image": image_base64,
                }
            ]
        )

        lis = self.globalListener(
            taskUUID=task_uuid,
        )

        def check(resolve: callable, reject: callable, *args: Any) -> bool:
            uploaded_image_list = self._globalMessages.get(task_uuid)
            # TODO: Update to support multiple images
            uploaded_image = uploaded_image_list[0] if uploaded_image_list else None

            if uploaded_image and uploaded_image.get("error"):
                reject(uploaded_image)
                return True

            if uploaded_image:
                del self._globalMessages[task_uuid]
                resolve(uploaded_image)
                return True

            return False

        response = await getIntervalWithPromise(check, debugKey="upload-image")

        lis["destroy"]()

        if "code" in response:
            # This indicates an error response
            raise RunwareAPIError(response)

        if response:
            image = UploadImageType(
                imageUUID=response["imageUUID"],
                imageURL=response["imageURL"],
                taskUUID=response["taskUUID"],
            )
        else:
            image = None
        return image

    async def uploadUnprocessedImage(
            self,
            file: Union[File, str],
            preProcessorType: EPreProcessorGroup,
            width: int = None,
            height: int = None,
            lowThresholdCanny: int = None,
            highThresholdCanny: int = None,
            includeHandsAndFaceOpenPose: bool = True,
    ) -> Optional[UploadImageType]:
        # Create a dummy UploadImageType object
        uploaded_unprocessed_image = UploadImageType(
            imageUUID=str(uuid.uuid4()),
            imageURL="https://example.com/uploaded_unprocessed_image.jpg",
            taskUUID=str(uuid.uuid4()),
        )

        return uploaded_unprocessed_image

    async def listenToImages(
            self,
            onPartialImages: Optional[Callable[[List[IImage], Optional[IError]], None]],
            taskUUID: str,
            groupKey: LISTEN_TO_IMAGES_KEY,
    ) -> Dict[str, Callable[[], None]]:
        """
        Set up a listener to receive partial image updates for a specific task.

        :param onPartialImages: A callback function to be invoked with the filtered images and any error.
        :param taskUUID: The unique identifier of the task to filter images for.
        :param groupKey: The group key to categorize the listener.
        :return: A dictionary containing a 'destroy' function to remove the listener.
        """
        logger.debug("Setting up images listener for taskUUID: %s", taskUUID)

        def listen_to_images_lis(m: Dict[str, Any]) -> None:
            # Handle successful image generation
            if isinstance(m.get("data"), list):
                images = [
                    img
                    for img in m["data"]
                    if img.get("taskType") == "imageInference"
                       and img.get("taskUUID") == taskUUID
                ]

                if images:
                    self._globalImages.extend(images)
                    try:
                        partial_images = instantiateDataclassList(IImage, images)
                        if onPartialImages:
                            onPartialImages(
                                partial_images, None
                            )  # No error in this case
                    except Exception as e:
                        print(
                            f"Error occurred in user on_partial_images callback function: {e}"
                        )

            # Handle error messages
            elif isinstance(m.get("errors"), list):
                errors = [
                    error for error in m["errors"] if error.get("taskUUID") == taskUUID
                ]
                if errors:
                    error = IError(
                        error=True,  # Since this is an error message, we set this to True
                        error_message=errors[0].get("message", "Unknown error"),
                        task_uuid=errors[0].get("taskUUID", ""),
                        error_code=errors[0].get("code"),
                        error_type=errors[0].get("type"),
                        parameter=errors[0].get("parameter"),
                        documentation=errors[0].get("documentation"),
                    )
                    self._globalError = (
                        error  # Store the first error related to this task
                    )
                    if onPartialImages:
                        onPartialImages(
                            [], self._globalError
                        )  # Empty list for images, pass the error

        def listen_to_images_check(m):
            logger.debug("Images check message: %s", m)
            # Check for successful image inference messages
            image_inference_check = isinstance(m.get("data"), list) and any(
                item.get("taskType") == "imageInference" for item in m["data"]
            )
            # Check for error messages with matching taskUUID
            error_check = isinstance(m.get("errors"), list) and any(
                error.get("taskUUID") == taskUUID for error in m["errors"]
            )
            error_code_check = True if any([error.get('code') for error in m.get('errors', [])]) else False
            if error_code_check:
                self._globalError = IError(
                    error=True,
                    error_message=f"Error in image inference: {m.get('errors')}",
                    task_uuid=taskUUID,
                )

            response = image_inference_check or error_check
            return response

        temp_listener = self.addListener(
            check=listen_to_images_check, lis=listen_to_images_lis, groupKey=groupKey
        )

        logger.debug("listenToImages :: Temp listener: %s", temp_listener)

        return temp_listener

    def globalListener(self, taskUUID: str) -> Dict[str, Callable[[], None]]:
        """
        Set up a global listener to capture specific messages based on the provided taskUUID.

        :param taskUUID: The unique identifier of the task associated with the listener.
        :return: A dictionary containing a 'destroy' function to remove the listener.
        """
        logger.debug("Setting up global listener for taskUUID: %s", taskUUID)

        def global_lis(m: Dict[str, Any]) -> None:
            logger.debug("Global listener message: %s", m)
            logger.debug("Global listener taskUUID: %s", taskUUID)
            # logger.debug("Global listener taskKey: %s", taskKey)

            if m.get("error"):
                self._globalMessages[taskUUID] = m
                return

            value = accessDeepObject(
                taskUUID, m
            )  # I think this is the taskType now, and it returns the content of 'data'

            if isinstance(value, list):
                for v in value:
                    self._globalMessages[v["taskUUID"]] = self._globalMessages.get(
                        v["taskUUID"], []
                    ) + [v]
                    logger.debug("Global messages v: %s", v)
                    logger.debug(
                        "self._globalMessages[v[taskUUID]]: %s",
                        self._globalMessages[v["taskUUID"]],
                    )
            else:
                self._globalMessages[value["taskUUID"]] = value

        def global_check(m):
            logger.debug("Global check message: %s", m)
            return accessDeepObject(taskUUID, m)

        logger.debug("Global Listener taskUUID: %s", taskUUID)

        temp_listener = self.addListener(check=global_check, lis=global_lis)
        logger.debug("globalListener :: Temp listener: %s", temp_listener)

        return temp_listener

    def handleIncompleteImages(
            self, taskUUIDs: List[str], error: Any
    ) -> Optional[List[IImage]]:
        """
        Handle scenarios where the requested number of images is not fully received.

        :param taskUUIDs: A list of task UUIDs to filter the images.
        :param error: The error object to raise if there are no or only one image.
        :return: A list of available images if there are more than one, otherwise None.
        :raises: The provided error if there are no or only one image.
        """
        imagesWithSimilarTask = [
            img for img in self._globalImages if img["taskUUID"] in taskUUIDs
        ]
        if len(imagesWithSimilarTask) > 1:
            self._globalImages = [
                img for img in self._globalImages if img["taskUUID"] not in taskUUIDs
            ]
            return imagesWithSimilarTask
        else:
            raise error

    async def ensureConnection(self) -> None:
        """
        Ensure that a connection is established with the server.

        This method checks if the current connection is active and, if not, initiates a new connection.
        It handles authentication and retries the connection if necessary.

        :raises: An error message if the connection cannot be established due to an invalid API key or other reasons.
        """
        isConnected = self.connected() and self._ws.open
        # print(f"Is connected: {isConnected}")

        try:
            if self._invalidAPIkey:
                raise self._invalidAPIkey

            if not isConnected:
                await self.connect()
                # await asyncio.sleep(2)

        except Exception as e:
            raise self._invalidAPIkey or "Could not connect to server. Ensure your API key is correct"

    async def getSimililarImage(
            self,
            taskUUID: Union[str, List[str]],
            numberOfImages: int,
            shouldThrowError: bool = False,
            lis: Optional[ListenerType] = None,
    ) -> Union[List[IImage], IError]:
        """
        Retrieve similar images based on the provided task UUID(s) and desired number of images.

        :param taskUUID: A single task UUID or a list of task UUIDs to filter images.
        :param numberOfImages: The desired number of images to retrieve.
        :param shouldThrowError: A flag indicating whether to throw an error if the desired number of images is not reached.
        :param lis: An optional listener to handle image updates.
        :return: A list of retrieved images or an error object if the desired number of images is not reached.
        """
        taskUUIDs = taskUUID if isinstance(taskUUID, list) else [taskUUID]

        def check(
                resolve: Callable[[List[IImage]], None],
                reject: Callable[[IError], None],
                intervalId: Any,
        ) -> Optional[bool]:
            # print(f"Check # Task UUIDs: {taskUUIDs}")
            # print(f"Check # Global images: {self._globalImages}")
            # print(f"Check # reject: {reject}")
            # print(f"Check # resolve: {resolve}")
            logger.debug(f"Check # Global images: {self._globalImages}")
            imagesWithSimilarTask = [
                img
                for img in self._globalImages
                if img.get("taskType") == "imageInference"
                   and img.get("taskUUID") in taskUUIDs
            ]
            # logger.debug(f"Check # imagesWithSimilarTask: {imagesWithSimilarTask}")

            if self._globalError:
                logger.debug(f"Check # _globalError: {self._globalError}")

                error = self._globalError
                self._globalError = None
                logger.debug(f"Rejecting with error: {error}")
                logger.debug(f"Rejecting function: {reject}")

                reject(RunwareError(error))
                return True
            elif len(imagesWithSimilarTask) >= numberOfImages:
                resolve(imagesWithSimilarTask[:numberOfImages])
                self._globalImages = [
                    img
                    for img in self._globalImages
                    if img.get("taskType") == "imageInference"
                       and img.get("taskUUID") not in taskUUIDs
                ]
                return True
            # return False

        return await getIntervalWithPromise(
            check, debugKey="getting images", shouldThrowError=shouldThrowError
        )

    async def _modelUpload(self, requestModel: IUploadModelBaseType) -> Optional[IUploadModelResponse]:
        task_uuid = getUUID()
        base_fields = {
            "taskType": ETaskType.MODEL_UPLOAD.value,
            "taskUUID": task_uuid,
            "air": requestModel.air,
            "name": requestModel.name,
            "downloadUrl": requestModel.downloadUrl,
            "uniqueIdentifier": requestModel.uniqueIdentifier,
            "version": requestModel.version,
            "format": requestModel.format,
            "private": requestModel.private,
            "category": requestModel.category,
            "architecture": requestModel.architecture,
        }

        optional_fields = [
            "retry", "heroImageUrl", "tags", "shortDescription", "comment",
            "positiveTriggerWords", "type", "negativeTriggerWords",
            "defaultWeight", "defaultStrength", "defaultGuidanceScale",
            "defaultSteps", "defaultScheduler", "conditioning"
        ]

        request_object = {**base_fields, **{field: getattr(requestModel, field) for field in optional_fields if
                                            getattr(requestModel, field, None) is not None}}

        await self.send(
            [
                request_object
            ]
        )

        lis = self.globalListener(
            taskUUID=task_uuid,
        )

        def check(resolve: callable, reject: callable, *args: Any) -> bool:
            uploaded_model_list = self._globalMessages.get(task_uuid, [])
            unique_statuses = set()
            all_models = []

            for uploaded_model in uploaded_model_list:
                if uploaded_model.get("code"):
                    raise RunwareAPIError(uploaded_model)

                status = uploaded_model.get("status")
                if status not in unique_statuses:
                    all_models.append(uploaded_model)
                    unique_statuses.add(status)

                if status == "ready":
                    uploaded_model_list.remove(uploaded_model)
                    if not uploaded_model_list:
                        del self._globalMessages[task_uuid]
                    else:
                        self._globalMessages[task_uuid] = uploaded_model_list
                    resolve(all_models)
                    return True

            return False

        response = await getIntervalWithPromise(check, debugKey="upload-model")

        lis["destroy"]()

        if "code" in response:
            # This indicates an error response
            raise RunwareAPIError(response)

        if response:
            if not isinstance(response, list):
                response = [response]

            models = []
            for item in response:
                models.append({
                    'taskType': item.get('taskType'),
                    'taskUUID': item.get('taskUUID'),
                    'status': item.get('status'),
                    'message': item.get('message'),
                    'air': item.get('air')
                })
        else:
            models = None
        return models

    async def modelUpload(self, requestModel: IUploadModelBaseType) -> Optional[IUploadModelResponse]:
        try:
            await self.ensureConnection()
            return await asyncRetry(lambda: self._modelUpload(requestModel))
        except Exception as e:
            raise e

    def connected(self) -> bool:
        """
        Check if the current WebSocket connection is active and authenticated.

        :return: True if the connection is active and authenticated, False otherwise.
        """
        return self.isWebsocketReadyState() and self._connectionSessionUUID is not None
