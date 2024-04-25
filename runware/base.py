import asyncio
import json
import uuid
from typing import List, Union, Optional, Callable, Any, Dict


from .utils import BASE_RUNWARE_URLS, delay, getUUID

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


class RunwareBase:
    def __init__(self, api_key, url=BASE_RUNWARE_URLS[Environment.PRODUCTION]):
        self._ws = None
        self._listeners = []
        self._api_key = api_key
        self._url = url
        self._global_messages = {}
        self._global_images = []
        self._global_error = None
        self._connection_session_uuid = None
        self._invalid_api_key = None
        self._sdk_type = SdkType.SERVER

    def isWebsocketReadyState(self):
        return self._ws and self._ws.readyState == 1

    def addListener(self, lis, check, groupKey=None):
        async def listener(msg):
            if msg.get("error"):
                lis(msg)
            elif check(msg):
                lis(msg)

        groupListener = ListenerType(
            key=getUUID(), listener=listener, groupKey=groupKey
        )
        self._listeners.append(groupListener)

        def destroy():
            self._listeners = removeListener(self._listeners, groupListener)

        return {"destroy": destroy}

    def connect(self):
        async def on_open(e):
            if self._connection_session_uuid:
                self.send(
                    {
                        "newConnection": {
                            "apiKey": self._api_key,
                            "connectionSessionUUID": self._connection_session_uuid,
                        }
                    }
                )
            else:
                self.send({"newConnection": {"apiKey": self._api_key}})

            self.addListener(
                check=lambda m: m.get("newConnectionSessionUUID", {}).get(
                    "connectionSessionUUID"
                ),
                lis=lambda m: self.handle_connection_response(m),
            )

        async def on_message(e):
            data = json.loads(e.data)
            for lis in self._listeners:
                result = lis.listener(data)
                if result:
                    return

        async def on_close(e):
            if self._invalid_api_key:
                print(f"Error: {self._invalid_api_key}")

        self._ws.onopen = on_open
        self._ws.onmessage = on_message
        self._ws.onclose = on_close

    def handle_connection_response(self, m):
        if m.get("error"):
            if m["errorId"] == 19:
                self._invalid_api_key = "Invalid API key"
            else:
                self._invalid_api_key = "Error connection"
            return
        self._connection_session_uuid = m.get("newConnectionSessionUUID", {}).get(
            "connectionSessionUUID"
        )
        self._invalid_api_key = None

    def send(self, msg):
        self._ws.send(json.dumps(msg))

    # def is_websocket_ready_state(self):
    #     return self._ws and self._ws.open

    # async def connect(self):
    #     self.logger.info("Connecting to Runware server from base")
    #     # Implement the connection logic based on the WebSocket library you choose
    #     pass

    # async def send(self, msg):
    #     await self._ws.send(json.dumps(msg))

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
        # Create a list of dummy IEnhancedPrompt objects
        enhanced_prompts = [
            IEnhancedPrompt(task_uuid=str(uuid.uuid4()), text=f"Enhanced Prompt {i+1}")
            for i in range(promptEnhancer.prompt_versions)
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

    def listenToImages(self, onPartialImages, taskUUID, groupKey):
        # Placeholder for the listenToImages method
        pass

    def globalListener(self, responseKey, taskKey, taskUUID):
        # Placeholder for the globalListener method
        pass

    def handleIncompleteImages(self, taskUUIDs, error):
        # Placeholder for the handleIncompleteImages method
        pass

    async def ensureConnection(self):
        # Placeholder for the ensureConnection method
        pass

    async def getSimililarImage(self, taskUUID, numberOfImages, shouldThrowError, lis):
        # Placeholder for the getSimililarImage method
        pass

    def connected(self):
        # Placeholder for the connected method
        pass
