import asyncio
import os
from dotenv import load_dotenv
from runware import Runware, IImageToText, IRequestImageToText

# Load environment variables from .env file
load_dotenv()


async def main() -> None:
    # Initialize the Runware client
    runware = Runware(api_key=os.environ.get("RUNWARE_API_KEY"))

    # The image requires for the seed image. It can be the UUID of previously generated image or an a file image.
    image_path = "path/to/image.jpg"

    request_image_to_text_payload = IRequestImageToText(image_initiator=image_path)

    image_to_text: IImageToText = await runware.requestImageToText(
        requestImageToText=request_image_to_text_payload
    )

    print("Description of the image:")
    print(image_to_text.text)


if __name__ == "__main__":
    asyncio.run(main())
