import base64
import mimetypes
import os
import re
import uuid
from dataclasses import fields
from typing import Any, Optional, Type, Union
from urllib.parse import urlparse

import aiofiles

from .logging_config import get_logger
from .types import Environment, File, UploadImageType

logger = get_logger(__name__)

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


async def fileToBase64(file_path: str) -> str:
    """
    Asynchronously convert a file at a given path to a Base64-encoded string.

    :param file_path: str, the path to the file.
    :return: str, Base64-encoded content of the file.
    :raises FileNotFoundError: if the file does not exist.
    :raises IOError: if the file cannot be read.
    """
    try:
        logger.debug(f"Converting file to base64: {file_path}")
        async with aiofiles.open(file_path, "rb") as file:
            file_contents = await file.read()
            mime_type, _ = mimetypes.guess_type(file_path)

            if mime_type is None:
                logger.warning(f"Unable to determine MIME type for file: {file_path}")
                raise ValueError(
                    f"Unable to determine the MIME type for file: {file_path}"
                )

            base64_content = base64.b64encode(file_contents).decode("utf-8")
            logger.debug(
                f"Successfully converted file to base64, size: {len(base64_content)} chars"
            )
            return f"data:{mime_type};base64,{base64_content}"
    except FileNotFoundError:
        logger.error(f"File not found: {file_path}")
        raise FileNotFoundError(f"The file at {file_path} does not exist.")
    except IOError as e:
        logger.error(f"Failed to read file: {file_path}", exc_info=e)
        raise IOError(f"The file at {file_path} could not be read.")


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


def isLocalFile(file):
    logger.debug(f"Checking if file is local: {file}")

    if os.path.isfile(file):
        logger.debug(f"File exists locally: {file}")
        return True

    # Check if the string is a valid UUID
    if isValidUUID(file):
        logger.debug(f"File is a valid UUID: {file}")
        return False

    # Check if the string is a valid URL
    parsed_url = urlparse(file)
    if parsed_url.scheme and parsed_url.netloc:
        logger.debug(f"File is a valid URL: {file}")
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
                logger.debug(f"File is a base64 string: {file}")
                return False

            # Assume it's a URL without scheme (e.g., 'example.com/some/path')
            # Add 'https://' in front and treat it as a valid URL
            file = f"https://{file}"
            parsed_url = urlparse(file)
            if parsed_url.netloc:  # Now it should have a valid netloc
                logger.debug(f"File is a URL without scheme: {file}")
                return False
            else:
                logger.error(f"File or URL '{file}' not found or invalid")
                raise FileNotFoundError(f"File or URL '{file}' not found.")

    logger.error(f"File or URL '{file}' not valid or not found")
    raise FileNotFoundError(f"File or URL '{file}' not valid or not found.")


async def process_image(
    image: Optional[Union[str, list, UploadImageType | None | File]],
) -> None | list[Any] | str:
    if image is None:
        return None
    elif isinstance(image, list):
        logger.debug(f"Processing {len(image)} images in list")
        images = []
        for img in image:
            images.append(await process_image(img))
        return images
    elif isinstance(image, UploadImageType):
        logger.debug(f"Using uploaded image UUID: {image.imageUUID}")
        return image.imageUUID

    if isLocalFile(image) and not image.startswith("http"):
        logger.debug(f"Converting local file to base64: {image}")
        return await fileToBase64(image)

    logger.debug(f"Using image as-is: {image}")
    return image
