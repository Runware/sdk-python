import asyncio
import os
import logging
from typing import List, Optional
from dotenv import load_dotenv
from runware import Runware, IImage, IImageBackgroundRemoval, RunwareAPIError

# Load environment variables from .env file
load_dotenv()

RUNWARE_API_KEY = os.environ.get("RUNWARE_API_KEY")


async def main() -> None:
    # Create an instance of RunwareServer
    runware = Runware(api_key=RUNWARE_API_KEY)

    # Connect to the Runware service
    await runware.connect()

    image_path = "retriever.jpg"
    image_path = "dalmatian.jpg"
    # image_path = "cb969e10-ef6f-449c-9f1a-cc3c2778a951"

    # With only mandatory parameters
    remove_image_background_payload = IImageBackgroundRemoval(inputImage=image_path)
    # With all parameters
    remove_image_background_payload = IImageBackgroundRemoval(
        inputImage=image_path,
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
