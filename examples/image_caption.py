import asyncio
import os
import logging
from dotenv import load_dotenv
from runware import Runware, IImageToText, IImageCaption, RunwareAPIError

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

    # With only mandatory parameters
    request_image_to_text_payload = IImageCaption(inputImage=image_path)
    # With all parameters
    request_image_to_text_payload = IImageCaption(
        inputImage=image_path,
        includeCost=True,
    )

    try:
        image_to_text: IImageToText = await runware.imageCaption(
            requestImageToText=request_image_to_text_payload
        )
    except RunwareAPIError as e:
        print(f"API Error: {e}")
        print(f"Error Code: {e.code}")
    except Exception as e:
        print(f"Unexpected Error: {e}")
    else:
        print("Description of the image:")
        print(image_to_text.text)


if __name__ == "__main__":
    asyncio.run(main())
