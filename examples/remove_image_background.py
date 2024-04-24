import asyncio
import os
from typing import List, Optional
from dotenv import load_dotenv
from runware import Runware, IImage, IRemoveImageBackground

# Load environment variables from .env file
load_dotenv()

RUNWARE_API_KEY = os.environ.get("RUNWARE_API_KEY")


async def main() -> None:
    # Initialize the Runware client
    runware = Runware(api_key=RUNWARE_API_KEY)

    image_path = "path/to/image.jpg"

    remove_image_background_payload = IRemoveImageBackground(image_initiator=image_path)

    processed_images: List[IImage] = await runware.removeImageBackground(
        removeImageBackgroundPayload=remove_image_background_payload
    )

    print("Processed Images:")
    # TODO: Does it really return List[IImage] or just one IImage object?
    for image in processed_images:
        print(image.imageSrc)


if __name__ == "__main__":
    asyncio.run(main())
