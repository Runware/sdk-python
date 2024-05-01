import asyncio
import os
import logging
from typing import List, Optional
from dotenv import load_dotenv
from runware import Runware, IImage, IRemoveImageBackground

# Load environment variables from .env file
load_dotenv()

RUNWARE_API_KEY = os.environ.get("RUNWARE_API_KEY")


async def main() -> None:
    # Create an instance of RunwareServer
    runware = Runware(api_key=RUNWARE_API_KEY)

    # Connect to the Runware service
    await runware.connect()

    image_path = "retriever.jpg"

    remove_image_background_payload = IRemoveImageBackground(image_initiator=image_path)
    try:
        processed_images: List[IImage] = await runware.removeImageBackground(
            removeImageBackgroundPayload=remove_image_background_payload
        )
    except Exception as e:
        print(f"Error: {e}")
        return

    print("Processed Image with the background removed:")
    # TODO: Does it really return List[IImage] or just one IImage object?
    for image in processed_images:
        print(image.imageSrc)


if __name__ == "__main__":
    asyncio.run(main())
