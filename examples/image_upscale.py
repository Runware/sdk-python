import asyncio
import os
import logging
from typing import List, Optional
from dotenv import load_dotenv
from runware import Runware, IImage, IImageUpscale, RunwareAPIError


# Load environment variables from .env file
load_dotenv()

RUNWARE_API_KEY = os.environ.get("RUNWARE_API_KEY")


async def main() -> None:
    # Initialize the Runware client
    runware = Runware(api_key=os.environ.get("RUNWARE_API_KEY"))
    # Specifies the input image to be processed https://docs.runware.ai/en/image-editing/upscaling#inputimage
    inputImage = "dalmatian.jpg"
    inputImage = "https://img.freepik.com/free-photo/macro-picture-red-leaf-lights-against-black-background_181624-32636.jpg"
    upscale_factor = 4

    # With only mandatory parameters
    upscale_gan_payload = IImageUpscale(
        inputImage=inputImage, upscaleFactor=upscale_factor
    )

    # With all parameters
    upscale_gan_payload = IImageUpscale(
        inputImage=inputImage,
        upscaleFactor=upscale_factor,
        outputType="URL",
        outputFormat="PNG",
        includeCost=True,
    )
    try:
        upscaled_images: List[IImage] = await runware.imageUpscale(
            upscaleGanPayload=upscale_gan_payload
        )
    except RunwareAPIError as e:
        print(f"API Error: {e}")
        print(f"Error Code: {e.code}")
    except Exception as e:
        print(f"Unexpected Error: {e}")
    else:
        print(f"Upscaled Images ({upscale_factor}x):")
        for inputImage in upscaled_images:
            print(inputImage.imageURL)


if __name__ == "__main__":
    asyncio.run(main())
