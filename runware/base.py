import asyncio
from doctest import debug
import json
import uuid
import inspect
from typing import List, Union, Optional, Callable, Any, Dict


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
    IRequestImage,
    IRequestImageToText,
    IImageToText,
    IRemoveImageBackground,
    IPromptEnhancer,
    IEnhancedPrompt,
    IUpscaleGan,
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

    async def requestImages(
        self, requestImage: IRequestImage
    ) -> Union[List[IImage], None]:
        let_lis: Optional[Any] = None
        request_object: Optional[Dict[str, Any]] = None
        task_uuids: List[str] = []
        retry_count = 0

        try:
            await self.ensureConnection()
            image_initiator_uuid: Optional[str] = None
            image_mask_initiator_uuid: Optional[str] = None
            control_net_data: List[IControlNetWithUUID] = []

            if requestImage.image_initiator:
                uploaded_image = await self.uploadImage(requestImage.image_initiator)
                if not uploaded_image:
                    return []
                image_initiator_uuid = uploaded_image.new_image_uuid

            if requestImage.image_mask_initiator:
                uploaded_mask_initiator = await self.uploadImage(
                    requestImage.image_mask_initiator
                )
                if not uploaded_mask_initiator:
                    return []
                image_mask_initiator_uuid = uploaded_mask_initiator.new_image_uuid

            if requestImage.control_net:
                for control_data in requestImage.control_net:
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

                    control_net_data.append(
                        IControlNetWithUUID(
                            guide_image_uuid=image_uploaded.new_image_uuid,
                            end_step=end_step,
                            preprocessor=preprocessor,
                            start_step=start_step,
                            weight=weight,
                            control_mode=control_mode or EControlMode.CONTROL_NET,
                            **get_canny_object(),
                        )
                    )

            prompt = f"{requestImage.positive_prompt} {'-no ' + requestImage.negative_prompt if requestImage.negative_prompt else ''}".strip()
            request_object = {
                "offset": 0,
                "modelId": requestImage.model_id,
                "promptText": prompt,
                "numberResults": requestImage.number_of_images,
                "sizeId": requestImage.image_size,
                "taskType": getTaskType(
                    prompt=prompt,
                    controlNet=requestImage.control_net,
                    imageMaskInitiator=requestImage.image_mask_initiator,
                    imageInitiator=requestImage.image_initiator,
                ),
                "useCache": requestImage.use_cache,
                "schedulerId": 22,
                "gScale": 7,
                **({"steps": requestImage.steps} if requestImage.steps else {}),
                **(
                    {"imageInitiatorUUID": image_initiator_uuid}
                    if image_initiator_uuid
                    else {}
                ),
                **(
                    {"imageMaskInitiatorUUID": image_mask_initiator_uuid}
                    if image_mask_initiator_uuid
                    else {}
                ),
                **({"controlNet": control_net_data} if control_net_data else {}),
                **({"lora": requestImage.lora} if requestImage.lora else {}),
                **({"seed": requestImage.seed} if requestImage.seed else {}),
            }
            # print(f"Request object: {request_object}")

            return await asyncRetry(
                lambda: self._requestImages(
                    request_object=request_object,
                    task_uuids=task_uuids,
                    let_lis=let_lis,
                    retry_count=retry_count,
                    number_of_images=requestImage.number_of_images,
                    on_partial_images=requestImage.on_partial_images,
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
            img for img in self._globalImages if img.task_uuid in task_uuids
        ]
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

        return [IImage(**image_data) for image_data in images]

        # return images

    async def requestImageToText(
        self, requestImageToText: IRequestImageToText
    ) -> IImageToText:
        try:
            await self.ensureConnection()
            return await asyncRetry(
                lambda: self._requestImageToText(requestImageToText)
            )
        except Exception as e:
            raise e

    async def _requestImageToText(
        self, requestImageToText: IRequestImageToText
    ) -> IImageToText:
        image_initiator = requestImageToText.image_initiator

        image_uploaded = await self.uploadImage(image_initiator)

        if not image_uploaded or not image_uploaded.newImageUUID:
            return None

        task_uuid = getUUID()

        await self.send(
            {
                "newReverseImageClip": {
                    "imageUUID": image_uploaded.newImageUUID,
                    "taskUUID": task_uuid,
                }
            }
        )

        lis = self.globalListener(
            responseKey="newReverseClip",
            taskKey="newReverseClip.texts",
            taskUUID=task_uuid,
        )

        def check(resolve: callable, reject: callable, *args: Any) -> bool:
            response = self._globalMessages.get(task_uuid)
            # TODO: Check why I need a conversion here?
            if response:
                image_to_text = response[0]
            else:
                image_to_text = response
            if image_to_text and image_to_text.get("error"):
                reject(image_to_text)
                return True

            if image_to_text:
                del self._globalMessages[task_uuid]
                resolve(image_to_text)
                return True

            return False

        response = await getIntervalWithPromise(check, debugKey="image-to-text")

        lis["destroy"]()

        if response:
            return IImageToText(task_uuid=response["taskUUID"], text=response["text"])
        else:
            return None

    async def removeImageBackground(
        self, removeImageBackgroundPayload: IRemoveImageBackground
    ) -> List[IImage]:
        try:
            await self.ensureConnection()
            return await asyncRetry(
                lambda: self._removeImageBackground(removeImageBackgroundPayload)
            )
        except Exception as e:
            raise e

    async def _removeImageBackground(
        self, removeImageBackgroundPayload: IRemoveImageBackground
    ) -> List[IImage]:
        image_initiator = removeImageBackgroundPayload.image_initiator

        image_uploaded = await self.uploadImage(image_initiator)

        if not image_uploaded or not image_uploaded.newImageUUID:
            return []

        taskUUID = getUUID()

        await self.send(
            {
                "newRemoveBackground": {
                    "imageUUID": image_uploaded.newImageUUID,
                    "taskUUID": taskUUID,
                    "taskType": 8,
                }
            }
        )

        lis = self.globalListener(
            responseKey="newRemoveBackground",
            taskKey="newRemoveBackground.images",
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

        image_list: List[IImage] = [
            IImage(
                imageSrc=response["imageSrc"],
                imageUUID=response["imageUUID"],
                taskUUID=response["taskUUID"],
                bNSFWContent=response["bNSFWContent"],
            )
        ]

        return image_list

    async def upscaleGan(self, upscaleGanPayload: IUpscaleGan) -> List[IImage]:
        try:
            await self.ensureConnection()
            return await asyncRetry(lambda: self._upscaleGan(upscaleGanPayload))
        except Exception as e:
            raise e

    async def _upscaleGan(self, upscaleGanPayload: IUpscaleGan) -> List[IImage]:
        image_initiator = upscaleGanPayload.image_initiator
        upscale_factor = upscaleGanPayload.upscale_factor

        image_uploaded = await self.uploadImage(image_initiator)

        if not image_uploaded or not image_uploaded.newImageUUID:
            return []

        taskUUID = getUUID()

        await self.send(
            {
                "newUpscaleGan": {
                    "imageUUID": image_uploaded.newImageUUID,
                    "taskUUID": taskUUID,
                    "upscaleFactor": upscale_factor,
                }
            }
        )

        lis = self.globalListener(
            responseKey="newUpscaleGan",
            taskKey="newUpscaleGan.images",
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

        # TODO: The respones has an upscaleImageUUID field, should I return it as well?
        image_list: List[IImage] = [
            IImage(
                imageSrc=response["imageSrc"],
                imageUUID=response["imageUUID"],
                taskUUID=response["taskUUID"],
                bNSFWContent=response["bNSFWContent"],
            )
        ]

        return image_list

    async def enhancePrompt(
        self, promptEnhancer: IPromptEnhancer
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
        self, promptEnhancer: IPromptEnhancer
    ) -> List[IEnhancedPrompt]:
        """
        Internal method to perform the actual prompt enhancement.

        :param promptEnhancer: An IPromptEnhancer object containing the prompt details.
        :return: A list of IEnhancedPrompt objects representing the enhanced versions of the prompt.
        """
        prompt = promptEnhancer.prompt
        promptMaxLength = promptEnhancer.prompt_max_length or 380
        promptLanguageId = promptEnhancer.prompt_language_id or 1
        promptVersions = promptEnhancer.prompt_versions or 1

        taskUUID = getUUID()

        await self.send(
            {
                "newPromptEnhance": {
                    "prompt": prompt,
                    "taskUUID": taskUUID,
                    "promptMaxLength": promptMaxLength,
                    "promptVersions": promptVersions,
                    "promptLanguageId": promptLanguageId,
                }
            }
        )

        lis = self.globalListener(
            responseKey="newPromptEnhancer",
            taskKey="newPromptEnhancer.texts",
            taskUUID=taskUUID,
        )

        def check(resolve: Any, reject: Any, *args: Any) -> bool:
            response = self._globalMessages.get(taskUUID)
            if isinstance(response, dict) and response.get("error"):
                reject(response)
                return True
            if response and len(response) >= promptVersions:
                del self._globalMessages[taskUUID]
                resolve(response)
                return True

            return False

        response = await getIntervalWithPromise(check, debugKey="enhance-prompt")

        lis["destroy"]()

        # Transform the response to a list of IEnhancedPrompt objects
        enhanced_prompts = [
            IEnhancedPrompt(task_uuid=prompt["taskUUID"], text=prompt["text"])
            for prompt in response
        ]

        return enhanced_prompts

    async def uploadImage(self, file: Union[File, str]) -> Optional[UploadImageType]:
        try:
            await self.ensureConnection()
            return await asyncRetry(lambda: self._uploadImage(file))
        except Exception as e:
            raise e

    async def _uploadImage(self, file: Union[File, str]) -> Optional[UploadImageType]:
        task_uuid = getUUID()

        if isinstance(file, str) and isValidUUID(file):
            return UploadImageType(
                new_image_uuid=file,
                new_image_src=file,
                task_uuid=task_uuid,
            )

        image_base64 = await fileToBase64(file) if isinstance(file, str) else file

        await self.send(
            {
                "newImageUpload": {
                    "imageBase64": image_base64,
                    "taskUUID": task_uuid,
                    "taskType": 7,
                }
            }
        )

        lis = self.globalListener(
            responseKey="newUploadedImageUUID",
            taskKey="newUploadedImageUUID",
            taskUUID=task_uuid,
        )

        def check(resolve: callable, reject: callable, *args: Any) -> bool:
            uploaded_image = self._globalMessages.get(task_uuid)

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

        if response:
            image = UploadImageType(
                newImageUUID=response["newImageUUID"],
                newImageSrc=response["newImageSrc"],
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
            new_image_uuid=str(uuid.uuid4()),
            new_image_src="https://example.com/uploaded_unprocessed_image.jpg",
            task_uuid=str(uuid.uuid4()),
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
            if m.get("newImages") and m["newImages"].get("images"):
                images = [
                    img
                    for img in m["newImages"]["images"]
                    if img["taskUUID"] == taskUUID
                ]

                if m.get("error"):
                    self._globalError = m
                else:
                    self._globalImages.extend(images)
                # print(f"Images listener: {images}")
                if len(images) > 0:
                    try:
                        partial_images = [IImage(**image_data) for image_data in images]
                        if onPartialImages:
                            onPartialImages(partial_images, m.get("error"))
                    except Exception as e:
                        print(
                            f"Error occurred in user on_partial_images callback function: {e}"
                        )
                    # print(f"Images listener: {images}")

        def listen_to_images_check(m):
            logger.debug("Images check message: %s", m)
            return m.get("newImages") and m["newImages"].get("images")

        temp_listener = self.addListener(
            check=listen_to_images_check, lis=listen_to_images_lis, groupKey=groupKey
        )

        logger.debug("Temp listener: %s", temp_listener)

        return temp_listener

    def globalListener(
        self, responseKey: str, taskKey: str, taskUUID: str
    ) -> Dict[str, Callable[[], None]]:
        """
        Set up a global listener to capture specific messages based on the provided keys.

        :param responseKey: The key to check for the presence of the desired response data.
        :param taskKey: The key to extract the relevant data from the received message.
        :param taskUUID: The unique identifier of the task associated with the listener.
        :return: A dictionary containing a 'destroy' function to remove the listener.
        """
        logger.debug("Setting up global listener for taskUUID: %s", taskUUID)

        def global_lis(m: Dict[str, Any]) -> None:
            logger.debug("Global listener message: %s", m)
            logger.debug("Global listener taskUUID: %s", taskUUID)
            logger.debug("Global listener taskKey: %s", taskKey)

            if m.get("error"):
                self._globalMessages[taskUUID] = m
                return

            value = accessDeepObject(taskKey, m)

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
            return accessDeepObject(responseKey, m)

        logger.debug("Global Listener responseKey: %s", responseKey)
        logger.debug("Global Listener taskUUID: %s", taskUUID)
        logger.debug("Global Listener taskKey: %s", taskKey)

        temp_listener = self.addListener(check=global_check, lis=global_lis)
        logger.debug("Temp listener: %s", temp_listener)

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
            # print(f"Task UUIDs: {taskUUIDs}")
            # print(f"Global images: {self._globalImages}")

            imagesWithSimilarTask = [
                img for img in self._globalImages if img["taskUUID"] in taskUUIDs
            ]

            # print(f"No images with similar task: {len(imagesWithSimilarTask)}")
            # print(f"numberOfImages: {numberOfImages}")

            if self._globalError:
                newData = self._globalError
                self._globalError = None
                reject(newData)
                return True
            elif len(imagesWithSimilarTask) >= numberOfImages:
                resolve(imagesWithSimilarTask[:numberOfImages])
                self._globalImages = [
                    img
                    for img in self._globalImages
                    if img["taskUUID"] not in taskUUIDs
                ]
                return True

        return await getIntervalWithPromise(
            check, debugKey="getting images", shouldThrowError=shouldThrowError
        )

    def connected(self) -> bool:
        """
        Check if the current WebSocket connection is active and authenticated.

        :return: True if the connection is active and authenticated, False otherwise.
        """
        return self.isWebsocketReadyState() and self._connectionSessionUUID is not None
