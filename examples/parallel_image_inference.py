import asyncio
import os
import logging
from typing import List, Optional
from dotenv import load_dotenv
from runware import Runware, IImage, IError, IImageInference, RunwareAPIError
from runware.types import ILora

# Load environment variables from .env file
load_dotenv()

RUNWARE_API_KEY = os.environ.get("RUNWARE_API_KEY")


# By providing the `onPartialImages` callback function, you can receive and handle the generated images incrementally,
# allowing for more responsive and interactive processing of the images.
# Note that the `onPartialImages` callback function is optional. If you don't provide it, the `requestImages`
# method will still return the complete list of generated images once the entire async request is finished.
def on_partial_images(images: List[IImage], error: Optional[IError]) -> None:
    if error:
        print(f"API Error: {error}")
    else:
        print(f"Received {len(images)} partial images")
        for image in images:
            print(f"Partial Image URL: {image.imageURL}")
        # Process or save the partial image as needed


# This function provides a safe way to make image requests, handling potential exceptions
# without disrupting other concurrent requests. It's particularly useful when making multiple
# image requests simultaneously.
#
# Usage:
# - Use this function with asyncio.gather for parallel processing of multiple requests.
# - Check the return value: None indicates an error occurred (details will be printed),
#   while a successful result will contain the list of generated images.
async def safe_request_images(runware: Runware, request_image: IImageInference):
    try:
        return await runware.imageInference(requestImage=request_image)
    except RunwareAPIError as e:
        print(f"API Error: {e}")
        print(f"Error Code: {e.code}")
        return None
    except Exception as e:
        print(f"Unexpected Error: {e}")
        return None


async def main() -> None:
    # Create an instance of RunwareServer
    runware = Runware(api_key=RUNWARE_API_KEY)

    # Connect to the Runware service
    await runware.connect()

    lora_1 = [
        ILora(model="civitai:58390@62833", weight=0.4),
        ILora(model="civitai:42903@232848", weight=0.3),
        ILora(model="civitai:42903@222732", weight=0.3),
    ]

    request_image1 = IImageInference(
        positivePrompt="a beautiful sunset over the mountains",
        model="civitai:36520@76907",
        numberResults=2,
        negativePrompt="cloudy, rainy",
        onPartialImages=on_partial_images,
        height=512,
        width=512,
        outputFormat="PNG",
    )

    request_image2 = IImageInference(
        positivePrompt="a cozy hut in the woods",
        model="civitai:30240@102996",
        numberResults=1,
        negativePrompt="modern, city",
        lora=lora_1,
        height=1024,
        width=1024,
        outputType="base64Data",
    )

    request_image3 = IImageInference(
        positivePrompt="a wood workshop with tools and sawdust on the floor",
        model="civitai:4384@128713",
        numberResults=3,
        height=1024,
        width=1024,
        includeCost=True,
    )

    first_images_request, second_images_request, third_image_request = (
        await asyncio.gather(
            safe_request_images(runware, request_image1),
            safe_request_images(runware, request_image2),
            safe_request_images(runware, request_image3),
            return_exceptions=True,
        )
    )
    if first_images_request:
        print("\nFirst Image Request Results:")
        for image in first_images_request:
            print(f"Image URL: {image.imageURL}")
    else:
        print("First Image Request Failed")

    if second_images_request:
        print("\nSecond Image Request Results:")
        for image in second_images_request:
            print(f"imageBase64Data: {image.imageBase64Data[:100]}...")
    else:
        print("Second Image Request Failed")

    if third_image_request:
        print("\nThird Image Request Results:")
        for image in third_image_request:
            print(f"Image URL: {image.imageURL}")
    else:
        print("Third Image Request Failed")


if __name__ == "__main__":
    asyncio.run(main())
