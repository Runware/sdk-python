import asyncio
import os
from typing import List, Optional
from dotenv import load_dotenv
from runware import Runware, IImage, IUpscaleGan

# Load environment variables from .env file
load_dotenv()


async def main() -> None:
    # Initialize the Runware client
    runware = Runware(api_key=os.environ.get("RUNWARE_API_KEY"))

    # The image requires for the seed image. It can be the UUID of previously generated image or an a file image.
    image_path = "path/to/image.jpg"
    upscale_factor = 4

    upscale_gan_payload = IUpscaleGan(
        image_initiator=image_path, upscale_factor=upscale_factor
    )

    upscaled_images: List[IImage] = await runware.upscaleGan(
        upscaleGanPayload=upscale_gan_payload
    )

    # Print the URLs of the upscaled images
    # TODO: Does it really return a list of IImage objects or just one IImage object?
    print(f"Upscaled Images ({upscale_factor}x):")
    for image in upscaled_images:
        print(image.imageSrc)


if __name__ == "__main__":
    asyncio.run(main())
