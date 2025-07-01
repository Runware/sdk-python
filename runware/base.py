import inspect
import logging
import os
import re
import uuid
from asyncio import gather
from dataclasses import asdict
from typing import List, Optional, Union, Callable, Any, Dict

from websockets.protocol import State

from .async_retry import asyncRetry
from .types import (
    Environment,
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
    EPreProcessorGroup,
    File,
    ETaskType,
    IModelSearch,
    IModelSearchResponse,
    IControlNet,
    IVideo,
    IVideoInference,
    IGoogleProviderSettings,
    IKlingAIProviderSettings,
    IFrameImage,
)
from .types import IImage, IError, SdkType, ListenerType
from .utils import (
    BASE_RUNWARE_URLS,
    getUUID,
    fileToBase64,
    createImageFromResponse,
    createImageToTextFromResponse,
    createEnhancedPromptsFromResponse,
    instantiateDataclassList,
    RunwareAPIError,
    RunwareError,
    instantiateDataclass,
    TIMEOUT_DURATION,
    accessDeepObject,
    getIntervalWithPromise,
    removeListener,
    LISTEN_TO_IMAGES_KEY,
    isLocalFile,
    process_image, delay,
)

# Configure logging
# configure_logging(log_level=logging.CRITICAL)

logger = logging.getLogger(__name__)
MAX_POLLS_VIDEO_GENERATION = int(os.environ.get("RUNWARE_MAX_POLLS_VIDEO_GENERATION", 480))


class RunwareBase:
    def __init__(
            self,
            api_key: str,
            url: str = BASE_RUNWARE_URLS[Environment.PRODUCTION],
            timeout: int = TIMEOUT_DURATION,
    ):
        if timeout <= 0:
            raise ValueError("Timeout must be greater than 0 milliseconds")

        self._ws: Optional[ReconnectingWebsocketProps] = None
        self._listeners: List[ListenerType] = []
        self._apiKey: str = api_key
        self._url: Optional[str] = url
        self._timeout: int = timeout
        self._globalMessages: Dict[str, Any] = {}
        self._globalImages: List[IImage] = []
        self._globalError: Optional[IError] = None
        self._connectionSessionUUID: Optional[str] = None
        self._invalidAPIkey: Optional[str] = None
        self._sdkType: SdkType = SdkType.SERVER

    def isWebsocketReadyState(self) -> bool:
        return self._ws and self._ws.state is State.OPEN

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
                if isLocalFile(image) and not str(image).startswith("http"):
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
                **(
                    {"inputImages": requestPhotoMaker.inputImages}
                    if requestPhotoMaker.inputImages
                    else {}
                ),
                **(
                    {"steps": requestPhotoMaker.steps}
                    if requestPhotoMaker.steps
                    else {}
                ),
            }

            if requestPhotoMaker.outputFormat is not None:
                request_object["outputFormat"] = requestPhotoMaker.outputFormat
            if requestPhotoMaker.includeCost:
                request_object["includeCost"] = requestPhotoMaker.includeCost
            if requestPhotoMaker.outputType:
                request_object["outputType"] = requestPhotoMaker.outputType

            await self.send([request_object])

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

                if 0 < numberOfResults <= len(unique_results):
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
                self.logger.error(f"Error in photoMaker request:", exc_info=e)
                raise RunwareAPIError({"message": f"PhotoMaker failed after retries: {str(e)}"})
            else:
                raise e

    async def imageInference(
        self, requestImage: IImageInference
    ) -> Union[List[IImage], None]:
        let_lis: Optional[Any] = None
        request_object: Optional[Dict[str, Any]] = None
        task_uuids: List[str] = []
        retry_count = 0
        try:
            await self.ensureConnection()
            control_net_data: List[IControlNet] = []
            requestImage.maskImage = await process_image(requestImage.maskImage)
            requestImage.seedImage = await process_image(requestImage.seedImage)
            if requestImage.referenceImages:
                requestImage.referenceImages = await process_image(
                    requestImage.referenceImages
                )
            if requestImage.controlNet:
                for control_data in requestImage.controlNet:
                    image_uploaded = await self.uploadImage(control_data.guideImage)
                    if not image_uploaded:
                        return []
                    if hasattr(control_data, "preprocessor"):
                        control_data.preprocessor = control_data.preprocessor.value
                    control_data.guideImage = image_uploaded.imageUUID
                    control_net_data.append(control_data)
            prompt = f"{requestImage.positivePrompt}".strip()

            control_net_data_dicts = [asdict(item) for item in control_net_data]

            instant_id_data = {}
            if requestImage.instantID:
                instant_id_data = {
                    k: v
                    for k, v in vars(requestImage.instantID).items()
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

            ip_adapters_data = []
            if requestImage.ipAdapters:
                for ip_adapter in requestImage.ipAdapters:
                    ip_adapter_data = {
                        k: v for k, v in vars(ip_adapter).items() if v is not None
                    }
                    if "guideImage" in ip_adapter_data:
                        ip_adapter_data["guideImage"] = await process_image(
                            ip_adapter_data["guideImage"]
                        )

                    ip_adapters_data.append(ip_adapter_data)

            ace_plus_plus_data = {}
            if requestImage.acePlusPlus:
                ace_plus_plus_data = {
                    "inputImages": [],
                    "repaintingScale": requestImage.acePlusPlus.repaintingScale,
                    "type": requestImage.acePlusPlus.taskType,
                }
                if requestImage.acePlusPlus.inputImages:
                    ace_plus_plus_data["inputImages"] = await process_image(
                        requestImage.acePlusPlus.inputImages
                    )
                if requestImage.acePlusPlus.inputMasks:
                    ace_plus_plus_data["inputMasks"] = await process_image(
                        requestImage.acePlusPlus.inputMasks
                    )

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
                **(
                    {"controlNet": control_net_data_dicts}
                    if control_net_data_dicts
                    else {}
                ),
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
                        "lycoris": [
                            {"model": lycoris.model, "weight": lycoris.weight}
                            for lycoris in requestImage.lycoris
                        ]
                    }
                    if requestImage.lycoris
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
                                {
                                    "startStepPercentage": requestImage.refiner.startStepPercentage
                                }
                                if requestImage.refiner.startStepPercentage is not None
                                else {}
                            ),
                        }
                    }
                    if requestImage.refiner
                    else {}
                ),
                **({"instantID": instant_id_data} if instant_id_data else {}),
                **(
                    {
                        "outpaint": {
                            k: v
                            for k, v in vars(requestImage.outpaint).items()
                            if v is not None
                        }
                    }
                    if requestImage.outpaint
                    else {}
                ),
                **({"ipAdapters": ip_adapters_data} if ip_adapters_data else {}),
                **({"acePlusPlus": ace_plus_plus_data} if ace_plus_plus_data else {}),
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
            if requestImage.acceleratorOptions:
                pipeline_options_dict = {
                    k: v
                    for k, v in vars(requestImage.acceleratorOptions).items()
                    if v is not None
                }
                request_object.update({"acceleratorOptions": pipeline_options_dict})
            if requestImage.advancedFeatures:
                pipeline_options_dict = {
                    k: v.__dict__
                    for k, v in vars(requestImage.advancedFeatures).items()
                    if v is not None
                }
                request_object.update({"advancedFeatures": pipeline_options_dict})
            if requestImage.maskImage:
                request_object["maskImage"] = requestImage.maskImage
            if requestImage.referenceImages:
                request_object["referenceImages"] = requestImage.referenceImages
            if requestImage.strength:
                request_object["strength"] = requestImage.strength
            if requestImage.scheduler:
                request_object["scheduler"] = requestImage.scheduler
            if requestImage.vae:
                request_object["vae"] = requestImage.vae
            if requestImage.promptWeighting:
                request_object["promptWeighting"] = requestImage.promptWeighting
            if requestImage.maskMargin:
                request_object["maskMargin"] = requestImage.maskMargin
            if hasattr(requestImage, "extraArgs"):
                # if extraArgs is present, and a dictionary, we will add its attributes to the request.
                # these may contain options used for public beta testing.
                if isinstance(requestImage.extraArgs, dict):
                    request_object.update(requestImage.extraArgs)

            if requestImage.outputQuality:
                request_object["outputQuality"] = requestImage.outputQuality
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
                self.logger.error(f"Error in requestImages:", exc_info=e)
                raise RunwareAPIError({"message": f"Image inference failed after retries: {str(e)}"})
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
            shouldThrowError=True,
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

        response = await getIntervalWithPromise(
            check, debugKey="image-to-text", timeOutDuration=self._timeout
        )

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
        if removeImageBackgroundPayload.taskUUID is not None:
            taskUUID = removeImageBackgroundPayload.taskUUID
        else:
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
        if removeImageBackgroundPayload.includeCost:
            task_params["includeCost"] = removeImageBackgroundPayload.includeCost
        if removeImageBackgroundPayload.model:
            task_params["model"] = removeImageBackgroundPayload.model
        if removeImageBackgroundPayload.outputQuality:
            task_params["outputQuality"] = removeImageBackgroundPayload.outputQuality

        # Handle settings if provided - convert dataclass to dictionary and add non-None values
        if removeImageBackgroundPayload.settings:
            settings_dict = {
                k: v
                for k, v in vars(removeImageBackgroundPayload.settings).items()
                if v is not None
            }
            task_params.update(settings_dict)

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
            check, debugKey="remove-image-background", timeOutDuration=self._timeout
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

        response = await getIntervalWithPromise(
            check, debugKey="upscale-gan", timeOutDuration=self._timeout
        )

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
        promptMaxLength = getattr(promptEnhancer, "promptMaxLength", 380)

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

        response = await getIntervalWithPromise(
            check, debugKey="enhance-prompt", timeOutDuration=self._timeout
        )

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

    async def _uploadImage(self, file: Union[File, str]) -> Optional[UploadImageType]:
        task_uuid = getUUID()
        local_file = True
        if isinstance(file, str):
            if os.path.exists(file):
                local_file = True
            else:
                local_file = isLocalFile(file)

                # Check if it's a base64 string (with or without data URI prefix)
                if file.startswith("data:") or re.match(
                    r"^[A-Za-z0-9+/]+={0,2}$", file
                ):
                    # Assume it's a base64 string (with or without data URI prefix)
                    local_file = False
            if not local_file:
                return UploadImageType(
                    imageUUID=file,
                    imageURL=file,
                    taskUUID=task_uuid,
                )

            file = await fileToBase64(file)

        await self.send(
            [
                {
                    "taskType": ETaskType.IMAGE_UPLOAD.value,
                    "taskUUID": task_uuid,
                    "image": file,
                }
            ]
        )

        lis = self.globalListener(taskUUID=task_uuid)

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

        response = await getIntervalWithPromise(
            check, debugKey="upload-image", timeOutDuration=self._timeout
        )

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
            error_code_check = (
                True
                if any([error.get("code") for error in m.get("errors", [])])
                else False
            )
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
        isConnected = self.connected() and self._ws.state is State.OPEN
        # print(f"Is connected: {isConnected}")

        try:
            if self._invalidAPIkey:
                raise ConnectionError(self._invalidAPIkey)

            if not isConnected:
                await self.connect()
                # await asyncio.sleep(2)

        except Exception as e:
            raise ConnectionError(
                self._invalidAPIkey
                or "Could not connect to server. Ensure your API key is correct"
            )

    async def getSimililarImage(
        self,
        taskUUID: Union[str, List[str]],
        numberOfImages: int,
        shouldThrowError: bool = False,
        lis: Optional[ListenerType] = None,
        timeout: Optional[int] = None,
    ) -> Union[List[IImage], IError]:
        """
        Retrieve similar images based on the provided task UUID(s) and desired number of images.

        :param taskUUID: A single task UUID or a list of task UUIDs to filter images.
        :param numberOfImages: The desired number of images to retrieve.
        :param shouldThrowError: A flag indicating whether to throw an error if the desired number of images is not reached.
        :param lis: An optional listener to handle image updates.
        :param timeout: The timeout duration for the operation.
        :return: A list of retrieved images or an error object if the desired number of images is not reached.
        """
        taskUUIDs = taskUUID if isinstance(taskUUID, list) else [taskUUID]

        if timeout is None:
            timeout = self._timeout

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
            check,
            debugKey="getting images",
            shouldThrowError=shouldThrowError,
            timeOutDuration=timeout,
        )

    async def _modelUpload(
        self, requestModel: IUploadModelBaseType
    ) -> Optional[IUploadModelResponse]:
        task_uuid = getUUID()
        base_fields = {
            "taskType": ETaskType.MODEL_UPLOAD.value,
            "taskUUID": task_uuid,
            "air": requestModel.air,
            "name": requestModel.name,
            "downloadURL": requestModel.downloadURL,
            "uniqueIdentifier": requestModel.uniqueIdentifier,
            "version": requestModel.version,
            "format": requestModel.format,
            "private": requestModel.private,
            "category": requestModel.category,
            "architecture": requestModel.architecture,
        }

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
                field: getattr(requestModel, field)
                for field in optional_fields
                if getattr(requestModel, field, None) is not None
            },
        }

        await self.send([request_object])

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

                if status is not None and "error" in status:
                    raise RunwareAPIError(uploaded_model)

                if status == "ready":
                    uploaded_model_list.remove(uploaded_model)
                    if not uploaded_model_list:
                        del self._globalMessages[task_uuid]
                    else:
                        self._globalMessages[task_uuid] = uploaded_model_list
                    resolve(all_models)
                    return True

            return False

        response = await getIntervalWithPromise(
            check, debugKey="upload-model", timeOutDuration=self._timeout
        )

        lis["destroy"]()

        if "code" in response:
            # This indicates an error response
            raise RunwareAPIError(response)

        if response:
            if not isinstance(response, list):
                response = [response]

            models = []
            for item in response:
                models.append(
                    {
                        "taskType": item.get("taskType"),
                        "taskUUID": item.get("taskUUID"),
                        "status": item.get("status"),
                        "message": item.get("message"),
                        "air": item.get("air"),
                    }
                )
        else:
            models = None
        return models

    async def modelUpload(
        self, requestModel: IUploadModelBaseType
    ) -> Optional[IUploadModelResponse]:
        try:
            await self.ensureConnection()
            return await asyncRetry(lambda: self._modelUpload(requestModel))
        except Exception as e:
            raise e

    async def modelSearch(self, payload: IModelSearch) -> IModelSearchResponse:
        try:
            await self.ensureConnection()
            task_uuid = getUUID()

            request_object = {
                "taskUUID": task_uuid,
                "taskType": ETaskType.MODEL_SEARCH.value,
                **({"tags": payload.tags} if payload.tags else {}),
            }

            request_object.update(
                {
                    key: value
                    for key, value in vars(payload).items()
                    if value is not None and key != "additional_params"
                }
            )

            await self.send([request_object])

            listener = self.globalListener(taskUUID=task_uuid)

            def check(resolve: Callable, reject: Callable, *args: Any) -> bool:
                response = self._globalMessages.get(task_uuid)
                if response:
                    if response[0].get("error"):
                        reject(response[0])
                        return True
                    del self._globalMessages[task_uuid]
                    resolve(response[0])
                    return True
                return False

            response = await getIntervalWithPromise(
                check, debugKey="model-search", timeOutDuration=self._timeout
            )

            listener["destroy"]()

            if "code" in response:
                # This indicates an error response
                raise RunwareAPIError(response)

            return instantiateDataclass(IModelSearchResponse, response)

        except Exception as e:
            if isinstance(e, RunwareAPIError):
                raise

            raise RunwareAPIError({"message": str(e)})

    async def videoInference(self, requestVideo: IVideoInference) -> List[IVideo]:
        await self.ensureConnection()
        return await asyncRetry(lambda: self._requestVideo(requestVideo))

    async def _requestVideo(self, requestVideo: IVideoInference) -> List[IVideo]:
        await self._processVideoImages(requestVideo)
        requestVideo.taskUUID = requestVideo.taskUUID or getUUID()
        request_object = self._buildVideoRequest(requestVideo)
        await self.send([request_object])
        return await self._handleInitialVideoResponse(requestVideo.taskUUID, requestVideo.numberResults)

    async def _processVideoImages(self, requestVideo: IVideoInference) -> None:
        frame_tasks = []
        reference_tasks = []

        if requestVideo.frameImages:
            frame_tasks = [
                process_image(frame_item.inputImage)
                for frame_item in requestVideo.frameImages
                if isinstance(frame_item, IFrameImage)
            ]

        if requestVideo.referenceImages:
            reference_tasks = [
                process_image(reference_item)
                for reference_item in requestVideo.referenceImages
            ]

        frame_results = await gather(*frame_tasks) if frame_tasks else []
        reference_results = await gather(*reference_tasks) if reference_tasks else []

        if requestVideo.frameImages and frame_results:
            processed_frame_images = []
            result_index = 0
            for frame_item in requestVideo.frameImages:
                if isinstance(frame_item, IFrameImage):
                    frame_item.inputImages = frame_results[result_index]
                    result_index += 1
                processed_frame_images.append(frame_item)
            requestVideo.frameImages = processed_frame_images

        if requestVideo.referenceImages and reference_results:
            requestVideo.referenceImages = reference_results

    def _buildVideoRequest(self, requestVideo: IVideoInference) -> Dict[str, Any]:
        request_object = {
            "deliveryMethod": requestVideo.deliveryMethod,
            "taskType": ETaskType.VIDEO_INFERENCE.value,
            "taskUUID": requestVideo.taskUUID,
            "model": requestVideo.model,
            "positivePrompt": requestVideo.positivePrompt.strip(),
            "numberResults": requestVideo.numberResults,
        }

        self._addOptionalVideoFields(request_object, requestVideo)
        self._addVideoImages(request_object, requestVideo)
        self._addProviderSettings(request_object, requestVideo)
        return request_object

    def _addOptionalVideoFields(self, request_object: Dict[str, Any], requestVideo: IVideoInference) -> None:
        optional_fields = [
            "outputType", "outputFormat", "outputQuality", "uploadEndpoint",
            "includeCost", "negativePrompt", "fps", "steps", "seed",
            "CFGScale", "seedImage", "duration", "width", "height",
        ]

        for field in optional_fields:
            value = getattr(requestVideo, field, None)
            if value is not None:
                request_object[field] = value

    def _addVideoImages(self, request_object: Dict[str, Any], requestVideo: IVideoInference) -> None:
        if requestVideo.frameImages:
            frame_images_data = []
            for frame_item in requestVideo.frameImages:
                frame_images_data.append({k: v for k, v in asdict(frame_item).items() if v is not None})
            request_object["frameImages"] = frame_images_data

        if requestVideo.referenceImages:
            request_object["referenceImages"] = requestVideo.referenceImages

    def _addProviderSettings(self, request_object: Dict[str, Any], requestVideo: IVideoInference) -> None:
        if not requestVideo.providerSettings:
            return
        provider_dict = requestVideo.providerSettings.to_request_dict()
        if provider_dict:
            request_object["providerSettings"] = provider_dict

    async def _handleInitialVideoResponse(self, task_uuid: str, number_results: int) -> List[IVideo]:
        lis = self.globalListener(taskUUID=task_uuid)

        def check_initial_response(resolve: callable, reject: callable, *args: Any) -> bool:
            response_list = self._globalMessages.get(task_uuid, [])

            if not response_list:
                return False

            response = response_list[0]

            if response.get("code"):
                raise RunwareAPIError(response)

            if response.get("status") == "success":
                del self._globalMessages[task_uuid]
                resolve([response])
                return True

            del self._globalMessages[task_uuid]
            resolve("POLL_NEEDED")
            return True

        try:
            initial_response = await getIntervalWithPromise(
                check_initial_response,
                debugKey="video-inference-initial",
                timeOutDuration=30000
            )
        finally:
            lis["destroy"]()

        if initial_response == "POLL_NEEDED":
            return await self._pollVideoResults(task_uuid, number_results)
        else:
            return instantiateDataclassList(IVideo, initial_response)

    async def _pollVideoResults(self, task_uuid: str, number_results: int) -> List[IVideo]:
        for poll_count in range(MAX_POLLS_VIDEO_GENERATION):
            try:
                responses = await self._sendPollRequest(task_uuid, poll_count)
                completed_results = self._processVideoPollingResponse(responses)

                if len(completed_results) >= number_results:
                    return instantiateDataclassList(IVideo, completed_results[:number_results])

                if not self._hasPendingVideos(responses) and not completed_results:
                    raise RunwareAPIError({"message": f"Unexpected polling response at poll {poll_count}"})

            except Exception as e:
                if poll_count >= MAX_POLLS_VIDEO_GENERATION - 1:
                    raise e

            await delay(3)

        raise RunwareAPIError({"message": "Video generation timed out"})

    async def _sendPollRequest(self, task_uuid: str, poll_count: int) -> List[Dict[str, Any]]:
        await self.send([{
            "taskType": ETaskType.GET_RESPONSE.value,
            "taskUUID": task_uuid
        }])

        lis = self.globalListener(taskUUID=task_uuid)

        def check_poll_response(resolve: callable, reject: callable, *args: Any) -> bool:
            response_list = self._globalMessages.get(task_uuid, [])
            if response_list:
                del self._globalMessages[task_uuid]
                resolve(response_list)
                return True
            return False

        try:
            return await getIntervalWithPromise(
                check_poll_response,
                debugKey=f"video-poll-{poll_count}",
                timeOutDuration=10000
            )
        finally:
            lis["destroy"]()

    def _processVideoPollingResponse(self, responses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        completed_results = []

        for response in responses:
            if response.get("code"):
                raise RunwareAPIError(response)
            status = response.get("status")
            if status == "success":
                completed_results.append(response)

        return completed_results

    def _hasPendingVideos(self, responses: List[Dict[str, Any]]) -> bool:
        return any(response.get("status") == "pending" for response in responses)

    def connected(self) -> bool:
        """
        Check if the current WebSocket connection is active and authenticated.

        :return: True if the connection is active and authenticated, False otherwise.
        """
        return self.isWebsocketReadyState() and self._connectionSessionUUID is not None
