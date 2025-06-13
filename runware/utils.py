import asyncio
import base64
import os
import re
from urllib.parse import urlparse

import aiofiles
import datetime
import uuid
import json
import mimetypes
import inspect
from typing import Any, Dict, List, Union, Optional, TypeVar, Type, Coroutine
from dataclasses import fields
from .types import (
    Environment,
    EPreProcessor,
    EPreProcessorGroup,
    ListenerType,
    IControlNet,
    File,
    GetWithPromiseCallBackType,
    IImage,
    ETaskType,
    IImageToText,
    IEnhancedPrompt,
    IError,
    UploadImageType,
)
import logging

logger = logging.getLogger(__name__)

if not mimetypes.guess_type("test.webp")[0]:
    mimetypes.add_type("image/webp", ".webp")

BASE_RUNWARE_URLS = {
    Environment.PRODUCTION: "wss://ws-api.runware.ai/v1",
    Environment.TEST: "ws://localhost:8080",
}

RETRY_SDK_COUNTS = {
    "GLOBAL": 2,
    "REQUEST_IMAGES": 2,
}


PING_TIMEOUT_DURATION = 10000  # 10 seconds
PING_INTERVAL = 5000  # 5 seconds

TIMEOUT_DURATION = 240000  # 4 Minutes
POLLING_INTERVAL = 1000  # 1 seconds


class LISTEN_TO_IMAGES_KEY:
    REQUEST_IMAGES = "REQUEST_IMAGES"


class RunwareAPIError(Exception):
    def __init__(self, error_data: dict):
        self.code = error_data.get("code")
        self.error_data = error_data
        super().__init__(str(error_data))

    def __str__(self):
        return f"RunwareAPIError: {self.error_data}"


class RunwareError(Exception):
    def __init__(self, ierror: IError):
        self.ierror = ierror
        super().__init__(f"Runware Error: {ierror.error_message}")

    def format_error(self):
        return {
            "errors": [
                {
                    "code": self.ierror.error_code,
                    "message": self.ierror.error_message,
                    "parameter": self.ierror.parameter,
                    "type": self.ierror.error_type,
                    "documentation": self.ierror.documentation,
                    "taskUUID": self.ierror.task_uuid,
                }
            ]
        }

    def __str__(self):
        return f"Runware Error: {self.format_error()}"


class Blob:
    """
    A Python equivalent of the Blob class to simulate file-like behavior in an immutable manner.

    This class is designed to closely align with the TypeScript implementation of the Blob class.
    It provides a way to represent and manipulate immutable binary data, similar to how files are handled.

    :param blob_parts: List[bytes], content parts of the blob, typically bytes.
    :param options: Dict[str, Any], containing options such as type (MIME type).

    Example:
        content = b"Hello, world!"
        blob = Blob([content], {"type": "text/plain"})
        print(len(blob))  # Output: 13
        print(str(blob))  # Output: "Hello, world!"
        print(blob.size())  # Output: 13
    """

    def __init__(
        self,
        blob_parts: Optional[List[bytes]] = None,
        options: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the Blob object.
        :param blob_parts: List[bytes], content parts of the blob, typically bytes.
        :param options: Dict[str, Any], containing options such as type (MIME type).
        """
        self._content = b"".join(blob_parts) if blob_parts else b""
        self.type = options.get("type", "") if options else ""

    def __len__(self) -> int:
        return len(self._content)

    def __str__(self) -> str:
        return self._content.decode("utf-8")

    def size(self) -> int:
        return len(self)


class MockFile:
    """
    A class that provides a method to create mock file objects for testing purposes.

    The `create` method generates a Blob object that simulates a file with specified attributes
    such as name, size, and MIME type. This is useful when you need to work with file-like objects
    in tests or when actual files are not available.

    Example:
        mock_file = MockFile()
        file_obj = mock_file.create(name="example.txt", size=2048, mime_type="text/plain")
        print(file_obj.name)  # Output: "example.txt"
        print(file_obj.size())  # Output: 2048
        print(file_obj.type)  # Output: "text/plain"
        print(file_obj.lastModifiedDate)  # Output: current datetime
    """

    def create(
        self, name: str = "mock.txt", size: int = 1024, mime_type: str = "plain/txt"
    ) -> Blob:
        """
        Create a mock file object with specified attributes.

        This method generates a Blob object that simulates a file. The content of the file is
        created as a sequence of 'a' characters repeated 'size' times. The Blob object is then
        enhanced with additional attributes to mimic a real file, such as name and lastModifiedDate.

        :param name: str, the name of the file (default: "mock.txt")
        :param size: int, the size of the file in bytes (default: 1024)
        :param mime_type: str, the MIME type of the file (default: "plain/txt")
        :return: Blob, a Blob object simulating a file with the specified attributes
        """
        content = ("a" * size).encode()  # Create content as bytes
        blob = Blob(blob_parts=[content], options={"type": mime_type})

        # Simulate File attributes by adding properties to the Blob object
        setattr(blob, "name", name)
        setattr(blob, "lastModifiedDate", datetime.datetime.now())
        return blob


T = TypeVar("T")


def removeFromAray(col: Optional[List[T]], targetElem: T) -> None:
    """
    Remove the first occurrence of an element from an array.

    :param col: Optional[List[T]], the collection from which to remove the element. None is safely handled.
    :param target_elem: T, the element to remove from the collection.
    """
    if col is None:
        return
    try:
        col.remove(targetElem)
    except ValueError:
        # If targetElem is not in col, do nothing
        return


def getUUID() -> str:
    """
    Generate and return a new UUID as a string.
    """
    return str(uuid.uuid4())


def isValidUUID(uuid_str: str) -> bool:
    """
    Check if a given string is a valid UUID.

    :param uuid_str: str, the UUID string to validate.
    :return: bool, True if the string is a valid UUID, otherwise False.
    """
    try:
        uuid.UUID(uuid_str)
        return True
    except ValueError:
        return False


def evaluateToBoolean(*args: Any) -> bool:
    """
    Evaluate to boolean by checking if all arguments are truthy.

    :param args: Variable length argument list of any type.
    :return: Returns True if all arguments are truthy, otherwise False.
    """
    return all(args)


def compact(key: Any, data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Returns a dictionary containing the data if the key is truthy, otherwise returns an empty dictionary.

    :param key: Any, the key to check for truthiness.
    :param data: Dict[str, Any], the dictionary to return if the key is truthy.
    :return: A dictionary containing the data if key is truthy; otherwise, an empty dictionary.

    Example:
        lowThresholdCanny = 5  # or None if it should be omitted
        highThresholdCanny = 10  # or None if it should be omitted
        send_data = {
            "newPreProcessControlNet": {
                "taskUUID": "some-uuid",
                "preProcessorType": "some-type",
                "guideImageUUID": "another-uuid",
                "includeHandsAndFaceOpenPose": True,
                **compact(lowThresholdCanny, {"lowThresholdCanny": lowThresholdCanny}),
                **compact(highThresholdCanny, {"highThresholdCanny": highThresholdCanny}),
            },
        }
    """
    return data if key else {}


#  originally range() in Typescipt library, renamed nu avoid conflict with Python's built-in range function
# TODO: only used in tests/test_utils.py, consider removing
def generateString(count: int) -> str:
    return "a" * count


# TODO: function it's not used in the code anywhere else, consider removing
def remove1Mutate(col: List[Any], targetElem: Any) -> None:
    if col is None:
        return
    try:
        col.remove(targetElem)
    except ValueError:
        return


async def fileToBase64(file_path: str) -> str:
    """
    Asynchronously convert a file at a given path to a Base64-encoded string.

    :param file_path: str, the path to the file.
    :return: str, Base64-encoded content of the file.
    :raises FileNotFoundError: if the file does not exist.
    :raises IOError: if the file cannot be read.

    Example:
        try:
            if isinstance(file, str) and file.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                image_base64 = await fileToBase64(file)
            else:
                # Otherwise, use the string directly as it might be a Base64 string
                image_base64 = file

            await send({
                "newImageUpload": {
                    "imageBase64": image_base64,
                    "taskUUID": task_uuid,
                    "taskType": task_type,
                }
            })
        except FileNotFoundError as e:
            print(f"Error: {e}")
        except IOError as e:
            print(f"Error: {e}")

    """
    try:
        async with aiofiles.open(file_path, "rb") as file:
            file_contents = await file.read()
            mime_type, _ = mimetypes.guess_type(file_path)

            if mime_type is None:
                raise ValueError(
                    f"Unable to determine the MIME type for file: {file_path}"
                )

            base64_content = base64.b64encode(file_contents).decode("utf-8")
            return f"data:{mime_type};base64,{base64_content}"
    except FileNotFoundError:
        raise FileNotFoundError(f"The file at {file_path} does not exist.")
    except IOError:
        raise IOError(f"The file at {file_path} could not be read.")


def removeListener(
    listeners: List[ListenerType], listener: ListenerType
) -> List[ListenerType]:
    """
    Remove a specified listener from a list of listeners.

    This function filters out the listener whose `key` attribute matches that of the given `listener` object.
    It returns a new list without altering the original list of listeners.

    :param listeners: List[ListenerType], the list from which to remove the listener.
    :param listener: ListenerType, the listener to be removed based on its 'key'.
    :return: List[ListenerType], a new list with the specified listener removed.
    """

    return [lis for lis in listeners if lis.key != listener.key]


def removeAllKeyListener(listeners: List[ListenerType], key: str) -> List[ListenerType]:
    """
    Remove all listeners from the list that have a specific key.

    This function filters out all listeners whose `key` attribute matches the specified `key`.
    It creates a new list of listeners without those that have the matching key, without altering the original list.

    :param listeners: List[ListenerType], the list from which to remove listeners.
    :param key: str, the key associated with listeners to be removed.
    :return: List[ListenerType], a new list with all matching key listeners removed.
    """
    return [lis for lis in listeners if lis.key != key]


async def delay(time: float, milliseconds: int = 1000) -> None:
    """
    Asynchronously wait for a specified amount of time.

    :param time: float, the number of time units to wait.
    :param milliseconds: int, number of milliseconds each time unit represents.
    """
    await asyncio.sleep(time * milliseconds / 1000)


def getTaskType(
    prompt: str,
    controlNet: Optional[List[IControlNet]],
    imageMaskInitiator: Optional[Union[File, str]],
    imageInitiator: Optional[Union[File, str]],
) -> int:
    """
    Determine the task type based on the presence or absence of various parameters.

    :param prompt: str, the prompt text.
    :param control_net: Optional[List[IControlNet]], a list of settings for controlling the network, which can be None.
    :param image_initiator: Optional[Union[File, str]], a File object or a string path indicating the image initiator.
    :param image_mask_initiator: Optional[Union[File, str]], a File object or a string path indicating the image mask initiator.
    :return: int, the task type determined by the conditions.
    """
    if evaluateToBoolean(
        prompt, not controlNet, not imageMaskInitiator, not imageInitiator
    ):
        return 1
    if evaluateToBoolean(
        prompt, not controlNet, not imageMaskInitiator, imageInitiator
    ):
        return 2
    if evaluateToBoolean(prompt, not controlNet, imageMaskInitiator, imageInitiator):
        return 3
    if evaluateToBoolean(
        prompt, controlNet, not imageMaskInitiator, not imageInitiator
    ):
        return 9
    if evaluateToBoolean(prompt, controlNet, not imageMaskInitiator, imageInitiator):
        return 10
    if evaluateToBoolean(prompt, controlNet, imageMaskInitiator, imageInitiator):
        return 10
    # TODO: Better handling of invalid task types, e.g. raise an exception
    return -1


def getPreprocessorType(processor: EPreProcessor) -> EPreProcessorGroup:
    """
    Determine the preprocessor group based on the given preprocessor.

    :param processor: EPreProcessor, the preprocessor for which to determine the group.
    :return: EPreProcessorGroup, the corresponding preprocessor group for the given preprocessor.

    This function maps an EPreProcessor enum value to its corresponding EPreProcessorGroup enum value.
    It helps in identifying the group or category to which a specific preprocessor belongs.

    Example:
        preprocessor = EPreProcessor.canny
        group = getPreprocessorType(preprocessor)
        print(group)  # Output: EPreProcessorGroup.canny
    """
    if processor == EPreProcessor.canny:
        return EPreProcessorGroup.canny
    elif processor in [
        EPreProcessor.depth_leres,
        EPreProcessor.depth_midas,
        EPreProcessor.depth_zoe,
    ]:
        return EPreProcessorGroup.depth
    elif processor == EPreProcessor.inpaint_global_harmonious:
        return EPreProcessorGroup.depth
    elif processor == EPreProcessor.lineart_anime:
        return EPreProcessorGroup.lineart_anime
    elif processor in [
        EPreProcessor.lineart_coarse,
        EPreProcessor.lineart_realistic,
        EPreProcessor.lineart_standard,
    ]:
        return EPreProcessorGroup.lineart
    elif processor == EPreProcessor.mlsd:
        return EPreProcessorGroup.mlsd
    elif processor == EPreProcessor.normal_bae:
        return EPreProcessorGroup.normalbae
    elif processor in [
        EPreProcessor.openpose_face,
        EPreProcessor.openpose_faceonly,
        EPreProcessor.openpose_full,
        EPreProcessor.openpose_hand,
        EPreProcessor.openpose,
    ]:
        return EPreProcessorGroup.openpose
    elif processor in [EPreProcessor.scribble_hed, EPreProcessor.scribble_pidinet]:
        return EPreProcessorGroup.scribble
    elif processor in [
        EPreProcessor.seg_ofade20k,
        EPreProcessor.seg_ofcoco,
        EPreProcessor.seg_ufade20k,
    ]:
        return EPreProcessorGroup.seg
    elif processor == EPreProcessor.shuffle:
        return EPreProcessorGroup.shuffle
    elif processor in [
        EPreProcessor.softedge_hed,
        EPreProcessor.softedge_hedsafe,
        EPreProcessor.softedge_pidinet,
        EPreProcessor.softedge_pidisafe,
    ]:
        return EPreProcessorGroup.softedge
    elif processor == EPreProcessor.tile_gaussian:
        return EPreProcessorGroup.tile
    else:
        return EPreProcessorGroup.canny


def accessDeepObject(
    key: str,
    data: Dict[str, Any],
    useZero: bool = True,
    shouldReturnString: bool = False,
) -> Any:
    """
    Navigate deeply nested data structures based on a dot/bracket-notated key.

    :param key: str, the key path (e.g., "person.address[0].street").
    :param data: dict, the data to navigate.
    :param useZero: bool, return 0 instead of None for non-existent values.
    :param shouldReturnString: bool, return a JSON string of the object if True.
    :return: The value found at the key path or a default value.
    """

    # Get the current frame
    current_frame = inspect.currentframe()

    # Get the caller's frame
    caller_frame = current_frame.f_back

    # Get the caller's function name
    caller_name = caller_frame.f_code.co_name

    # Get the caller's line number
    caller_line_number = caller_frame.f_lineno

    logger.debug(
        f"Function {accessDeepObject.__name__} called by {caller_name} at line {caller_line_number}"
    )
    logger.debug(f"accessDeepObject key: {key}")
    logger.debug(f"accessDeepObject data: {data}")

    default_value = 0 if useZero else None

    # if "data" in data and isinstance(data["data"], list):
    #     # Iterate through each item in the data list
    #     for item in data["data"]:
    #         # Check if 'taskType' is in the item and matches the target_task_type
    #         if "taskType" in item and item["taskType"] == key:
    #             # Return the entire item if there's a match
    #             matching_tasks.append(item)
    matching_tasks = []

    for field in ["data", "errors"]:
        if field in data and isinstance(data[field], list):
            for item in data[field]:
                if "taskUUID" in item and item["taskUUID"] == key:
                    matching_tasks.append(item)

    # Check for successful messages
    if "data" in data and isinstance(data["data"], list):
        for item in data["data"]:
            if "taskUUID" in item and item["taskUUID"] == key:
                matching_tasks.append(item)

    # Check for error messages
    if "errors" in data and isinstance(data["errors"], list):
        for error in data["errors"]:
            if "taskUUID" in error and error["taskUUID"] == key:
                matching_tasks.append(error)

    if len(matching_tasks) == 0:
        return default_value

    logger.debug(f"accessDeepObject matching_tasks: {matching_tasks}")

    if shouldReturnString and isinstance(matching_tasks, (dict, list)):
        return json.dumps(matching_tasks)

    return matching_tasks

    # keys = re.split(r"\.|\[", key)
    # keys = [k.replace("]", "") for k in keys]

    # logger.debug(f"accessDeepObject keys: {keys}")

    # current_value = data
    # for k in keys:
    #     logger.debug(f"accessDeepObject key: {k}")
    #     # logger.debug(
    #     #     "isinstance(current_value, (dict, list))",
    #     #     str(isinstance(current_value, (dict, list))),
    #     # )
    #     if isinstance(current_value, (dict, list)):
    #         logger.debug(f"accessDeepObject current_value: {current_value}")
    #         logger.debug(f"k in current_value: {k in current_value}")
    #         if k.isdigit() and isinstance(current_value, list):
    #             index = int(k)
    #             if 0 <= index < len(current_value):
    #                 current_value = current_value[index]
    #             else:
    #                 return default_value
    #         elif k in current_value:
    #             current_value = current_value[k]
    #         else:
    #             return default_value
    #     else:
    #         return default_value

    # logger.debug(f"accessDeepObject current_value: {current_value}")

    # if shouldReturnString and isinstance(current_value, (dict, list)):
    #     return json.dumps(current_value)

    # return current_value


def createEnhancedPromptsFromResponse(response: List[dict]) -> List[IEnhancedPrompt]:
    def process_single_prompt(prompt_data: dict) -> IEnhancedPrompt:
        processed_fields = {}

        for field in fields(IEnhancedPrompt):
            if field.name in prompt_data:
                if field.name == "taskType":
                    processed_fields[field.name] = ETaskType(prompt_data[field.name])
                elif field.type == float or field.type == Optional[float]:
                    processed_fields[field.name] = float(prompt_data[field.name])
                else:
                    processed_fields[field.name] = prompt_data[field.name]

        return instantiateDataclass(IEnhancedPrompt, processed_fields)

    return [process_single_prompt(prompt) for prompt in response]


def createImageFromResponse(response: dict) -> IImage:
    processed_fields = {}

    for field in fields(IImage):
        if field.name in response:
            if field.type == bool or field.type == Optional[bool]:
                processed_fields[field.name] = bool(response[field.name])
            elif field.type == float or field.type == Optional[float]:
                processed_fields[field.name] = float(response[field.name])
            else:
                processed_fields[field.name] = response[field.name]

    return instantiateDataclass(IImage, processed_fields)


def createImageToTextFromResponse(response: dict) -> IImageToText:
    processed_fields = {}

    for field in fields(IImageToText):
        if field.name in response:
            if field.name == "taskType":
                # Convert string to ETaskType enum
                processed_fields[field.name] = ETaskType(response[field.name])
            elif field.type == float or field.type == Optional[float]:
                processed_fields[field.name] = float(response[field.name])
            else:
                processed_fields[field.name] = response[field.name]

    return instantiateDataclass(IImageToText, processed_fields)


async def getIntervalWithPromise(
    callback: GetWithPromiseCallBackType,
    debugKey: str = "debugKey",
    timeOutDuration: int = TIMEOUT_DURATION,  # in milliseconds
    shouldThrowError: bool = True,
    pollingInterval: int = 100,  # in milliseconds
) -> Any:
    """
    Set up an interval to repeatedly call a callback function until a condition is met or a timeout occurs.

    :param callback: A function that is called repeatedly within the interval. It receives an object with
                     `resolve`, `reject`, and `intervalId` properties, allowing the callback to control
                     the promise's resolution or rejection and clear the interval if needed.
                     The callback should return a boolean value indicating whether to clear the interval.
    :param debugKey: A string used for debugging purposes. Default is "debugKey".
    :param timeOutDuration: The duration in milliseconds after which the promise will be rejected if the
                            callback hasn't resolved or rejected it. Default is TIMEOUT_DURATION.
    :param shouldThrowError: A boolean indicating whether to reject the promise with an error message if
                             the timeout is reached. Default is True.
    :param pollingInterval: The interval in milliseconds at which the callback is invoked. Default is 100.
    :return: The result of the callback function or the rejection reason if the timeout is reached.

    Example:
        async def upload_image(task_uuid):
            image = await getIntervalWithPromise(
                lambda params: params["resolve"]("uploadedImage") if "uploadedImage" in globals() else None,
                debugKey="upload-image",
                pollingInterval=200,
            )
            return image

        uploaded_image = await upload_image("task123")
        print(uploaded_image)  # Output: "uploadedImage"
    """
    logger = logging.getLogger(__name__)

    loop = asyncio.get_running_loop()
    future = loop.create_future()
    intervalHandle = None

    async def check_callback():
        nonlocal intervalHandle, future
        try:
            if not future.done():
                # logger.debug(f"Checking callback for {debugKey}")
                # logger.debug(f"Future done: {future.done()}")
                # logger.debug(f"Future callback: {callback}")
                # logger.debug(f"Future result: {future.result}")

                # logger.debug(f"Future exception: {future.exception}")
                # logger.debug(f"callback: {callback}")

                result = callback(
                    future.set_result, future.set_exception, intervalHandle
                )

                if result:
                    if intervalHandle:
                        intervalHandle.cancel()
                        logger.debug(f"Interval cleared for {debugKey}")
                else:
                    # TODO: Find a better way than polling, as it's not very efficient.
                    # Consider using asyncio.Event or asyncio.Condition triggered by an incoming message
                    # as the state won't change unless I have a new message from the service
                    intervalHandle = loop.call_later(
                        pollingInterval / 1000,
                        lambda: (
                            # logger.debug("Creating task for check_callback"),
                            asyncio.create_task(check_callback()),
                        )[-1],
                    )
            else:
                logger.debug(
                    f"Future already done for {debugKey}, interval not rescheduled"
                )
        except Exception as e:
            future.set_exception(e)
            logger.exception(f"Error in check_callback 2 for {debugKey}: {str(e)}")

    await check_callback()
    # intervalHandle = loop.call_later(
    #    pollingInterval / 1000,
    #    lambda: (
    #        logger.debug("Creating task for check_callback"),
    #        asyncio.create_task(check_callback()),
    #    )[
    #        -1
    #    ],  # Return the task itself)
    # )

    async def timeout_handler():
        nonlocal future, intervalHandle
        try:
            if not future.done():
                if shouldThrowError:
                    future.set_exception(
                        Exception(f"Message could not be received for {debugKey}")
                    )
                    logger.error(f"Error: Message could not be received for {debugKey}")
                else:
                    future.set_result(None)
                if intervalHandle:
                    intervalHandle.cancel()
                    logger.debug(f"Interval cleared due to timeout for {debugKey}")
        except Exception as e:
            future.set_exception(e)
            logger.exception(f"Error in timeout_handler for {debugKey}: {str(e)}")

    # Schedule the timeout handler
    timeoutHandle = loop.call_later(
        timeOutDuration / 1000,
        lambda: (
            logger.debug("Creating task for timeout_handler"),
            asyncio.create_task(timeout_handler()),
        )[-1],
    )

    try:
        await future
    finally:
        if intervalHandle and not intervalHandle.cancelled():
            intervalHandle.cancel()
            logger.debug(f"Interval canceled for {debugKey}")
        if timeoutHandle and not timeoutHandle.cancelled():
            timeoutHandle.cancel()
            logger.debug(f"Timeout canceled for {debugKey}")

    return await future


def instantiateDataclass(dataclass_type: Type[Any], data: dict) -> Any:
    """
    Instantiates a dataclass object from a dictionary, filtering out any unknown attributes.

    :param dataclass_type: The dataclass type to instantiate.
    :param data: A dictionary with data.
    :return: An instantiated dataclass object.
    """
    # Get the set of valid field names for the dataclass
    valid_fields = {f.name for f in fields(dataclass_type)}
    # Filter the data to include only valid fields
    filtered_data = {k: v for k, v in data.items() if k in valid_fields}
    return dataclass_type(**filtered_data)


def instantiateDataclassList(
    dataclass_type: Type[Any], data_list: List[dict]
) -> List[Any]:
    """
    Instantiates a list of dataclass objects from a list of dictionaries,
    filtering out any unknown attributes.

    :param dataclass_type: The dataclass type to instantiate.
    :param data_list: A list of dictionaries with data.
    :return: A list of instantiated dataclass objects.
    """
    # Get the set of valid field names for the dataclass
    instances = []
    for data in data_list:
        instances.append(instantiateDataclass(dataclass_type, data))
    return instances


def isLocalFile(file):
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
        if (
            not parsed_url.scheme
            and not parsed_url.netloc
            or parsed_url.scheme == "data"
        ):
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


async def process_image(
    image: Optional[Union[str, list, UploadImageType | None | File]],
) -> None | list[Any] | str:
    if image is None:
        return None
    elif isinstance(image, list):
        images = []
        for image in image:
            images.append(await process_image(image))
        return images
    elif isinstance(image, UploadImageType):
        return image.imageUUID
    if isLocalFile(image) and not image.startswith("http"):
        return await fileToBase64(image)
    return image
