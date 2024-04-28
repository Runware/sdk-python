import asyncio
import json
import uuid
import inspect
from typing import List, Union, Optional, Callable, Any, Dict


from .utils import BASE_RUNWARE_URLS, delay, getUUID, removeListener, accessDeepObject
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
configure_logging(log_level=logging.CRITICAL)

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
        self._global_images: List[IImage] = []
        self._global_error: Optional[IError] = None
        self._connectionSessionUUID: Optional[str] = None
        self._invalid_api_key: Optional[str] = None
        self._sdkType: SdkType = SdkType.SERVER

    def isWebsocketReadyState(self) -> bool:
        return self._ws and self._ws.open

    def isAuthenticated(self):
        return self._connectionSessionUUID is not None

    async def addListener(
        self,
        lis: Callable[[Any], Any],
        check: Callable[[Any], Any],
        groupKey: Optional[str] = None,
    ) -> Dict[str, Callable[[], None]]:
        async def listener(msg: Any) -> None:
            if msg.get("error"):
                await lis(msg)
            elif await check(msg):
                await lis(msg)

        # Get the current frame
        current_frame = inspect.currentframe()

        # Get the caller's frame
        caller_frame = current_frame.f_back

        # Get the caller's function name
        caller_name = caller_frame.f_code.co_name

        # Get the caller's line number
        caller_line_number = caller_frame.f_lineno

        # Print the caller information
        logger.debug(
            f"Function {self.addListener.__name__} called by {caller_name} at line {caller_line_number}"
        )

        groupListener: ListenerType = ListenerType(
            key=getUUID(), listener=listener, group_key=groupKey
        )
        self._listeners.append(groupListener)

        def destroy() -> None:
            self._listeners = removeListener(self._listeners, groupListener)

        return {"destroy": destroy}

    def handle_connection_response(self, m):
        if m.get("error"):
            if m["errorId"] == 19:
                self._invalid_api_key = "Invalid API key"
            else:
                self._invalid_api_key = "Error connection"
            return
        self._connectionSessionUUID = m.get("newConnectionSessionUUID", {}).get(
            "connectionSessionUUID"
        )
        self._invalid_api_key = None

    async def requestImages(
        self,
        requestImage: IRequestImage,
    ) -> List[IImage]:
        # Create a list of dummy IImage objects
        images = [
            IImage(
                imageSrc=f"https://example.com/image_{i}.jpg",
                imageUUID=str(uuid.uuid4()),
                taskUUID=str(uuid.uuid4()),
                bNSFWContent=False,
            )
            for i in range(requestImage.number_of_images)
        ]

        # If onPartialImages callback is provided, invoke it with the dummy images
        if requestImage.on_partial_images:
            requestImage.on_partial_images(images, None)

        return images

    async def requestImageToText(
        self, requestImageToText: IRequestImageToText
    ) -> IImageToText:
        # Create a dummy IImageToText object
        image_to_text = IImageToText(
            task_uuid=str(uuid.uuid4()),
            text="Sample text",
        )

        return image_to_text

    async def removeImageBackground(
        self, removeImageBackgroundPayload: IRemoveImageBackground
    ) -> List[IImage]:
        # Create a list of dummy IImage objects
        images = [
            IImage(
                imageSrc=f"https://example.com/image_{i}.jpg",
                imageUUID=str(uuid.uuid4()),
                taskUUID=str(uuid.uuid4()),
                bNSFWContent=False,
            )
            for i in range(1)
        ]

        return images

    async def upscaleGan(self, upscaleGanPayload: IUpscaleGan) -> List[IImage]:
        # Create a list of dummy IImage objects
        images = [
            IImage(
                imageSrc=f"https://example.com/image_{i}.jpg",
                imageUUID=str(uuid.uuid4()),
                taskUUID=str(uuid.uuid4()),
                bNSFWContent=False,
            )
            for i in range(1)
        ]

        return images

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

        lis = await self.globalListener(
            responseKey="newPromptEnhancer",
            taskKey="newPromptEnhancer.texts",
            taskUUID=taskUUID,
        )

        def check(resolve: Any, reject: Any, *args: Any) -> bool:
            # print(f"Checking _globalMessages... {self._globalMessages}")
            # print(
            #     f"Checking task {taskUUID} for enhanced prompt... {self._globalMessages.get(taskUUID)}"
            # )

            # reducedPrompt: List[IEnhancedPrompt] = self._globalMessages.get(taskUUID)
            response = self._globalMessages.get(taskUUID)

            # print(f"Reduced prompt: {response}")

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
        # Create a dummy UploadImageType object
        uploaded_image = UploadImageType(
            new_image_uuid=str(uuid.uuid4()),
            new_image_src="https://example.com/uploaded_image.jpg",
            task_uuid=str(uuid.uuid4()),
        )

        return uploaded_image

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

    def listenToImages(
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

        def listener(m: Dict[str, Any]) -> None:
            if m.get("newImages") and m["newImages"].get("images"):
                images = [
                    img
                    for img in m["newImages"]["images"]
                    if img["taskUUID"] == taskUUID
                ]
                onPartialImages(images, m.get("error"))

                if m.get("error"):
                    self._globalError = m
                else:
                    if self._sdkType == SdkType.CLIENT:
                        self._globalImages.extend(m["newImages"]["images"])
                    else:
                        self._globalImages.extend(images)

        return self.addListener(
            listener,
            lambda m: m.get("newImages") and m["newImages"].get("images"),
            groupKey,
        )

    async def globalListener(
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

        async def listener(m: Dict[str, Any]) -> None:
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

        async def check(m):
            logger.debug("Global check message: %s", m)
            return accessDeepObject(responseKey, m)

        logger.debug("Global Listener responseKey: %s", responseKey)
        logger.debug("Global Listener taskUUID: %s", taskUUID)
        logger.debug("Global Listener taskKey: %s", taskKey)

        temp_listener = await self.addListener(check=check, lis=listener)
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
                    img
                    for img in self._globalImages
                    if img["taskUUID"] not in taskUUIDs
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
                self.connect()
                await asyncio.sleep(2)

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
            imagesWithSimilarTask = [
                img for img in self._globalImages if img["taskUUID"] in taskUUIDs
            ]

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
