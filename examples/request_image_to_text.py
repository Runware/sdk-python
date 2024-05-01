import asyncio
import os
import logging
from dotenv import load_dotenv
from runware import Runware, IImageToText, IRequestImageToText

# Load environment variables from .env file
load_dotenv()

RUNWARE_API_KEY = os.environ.get("RUNWARE_API_KEY")


async def main() -> None:
    # Create an instance of RunwareServer
    runware = Runware(api_key=RUNWARE_API_KEY)

    # Connect to the Runware service
    await runware.connect()

    # The image requires for the seed image. It can be the UUID of previously generated image or an a file image.
    image_path = "retriever.jpg"

    request_image_to_text_payload = IRequestImageToText(image_initiator=image_path)

    try:
        image_to_text: IImageToText = await runware.requestImageToText(
            requestImageToText=request_image_to_text_payload
        )
    except Exception as e:
        print(f"Error: {e}")
        return

    print("Description of the image:")
    print(image_to_text.text)


if __name__ == "__main__":
    asyncio.run(main())
