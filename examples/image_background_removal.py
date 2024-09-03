import asyncio
from doctest import debug
import os
import logging
from typing import List, Optional
from dotenv import load_dotenv
from runware import Runware, IImage, IImageBackgroundRemoval, RunwareAPIError

# Load environment variables from .env file
load_dotenv()

RUNWARE_API_KEY = os.environ.get("RUNWARE_API_KEY")


async def main() -> None:
    # Initialize the Runware client
    runware = Runware(api_key=os.environ.get("RUNWARE_API_KEY"))
    # Specifies the input image to be processed https://docs.runware.ai/en/image-editing/background-removal#inputimage

    inputImage = "retriever.jpg"
    inputImage = "ca9ef3d0-9863-4c55-a07c-34079f9f6608"
    # inputImage = "dalmatian.jpg"

    # With only mandatory parameters
    remove_image_background_payload = IImageBackgroundRemoval(inputImage=inputImage)
    # With all parameters
    remove_image_background_payload = IImageBackgroundRemoval(
        inputImage=inputImage,
        outputType="URL",
        outputFormat="PNG",
        rgba=[255, 255, 255, 0],
        postProcessMask=True,
        returnOnlyMask=False,
        alphaMatting=True,
        alphaMattingForegroundThreshold=200,
        alphaMattingBackgroundThreshold=50,
        alphaMattingErodeSize=10,
        includeCost=True,
    )

    try:
        processed_images: List[IImage] = await runware.imageBackgroundRemoval(
            removeImageBackgroundPayload=remove_image_background_payload
        )
    except RunwareAPIError as e:
        print(f"API Error: {e}")
        print(f"Error Code: {e.code}")
    except Exception as e:
        print(f"Unexpected Error: {e}")
    else:
        print("Processed Image with the background removed:")
        for image in processed_images:
            print(image.imageURL)


if __name__ == "__main__":
    asyncio.run(main())
