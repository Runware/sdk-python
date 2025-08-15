import asyncio
import os
import logging
from dotenv import load_dotenv
from runware import Runware, IImageToText, RunwareAPIError
from runware.types import IImageCaption

# Load environment variables from .env file
load_dotenv()

RUNWARE_API_KEY = os.environ.get("RUNWARE_API_KEY")


async def main() -> None:
    # Create an instance of RunwareServer
    runware = Runware(api_key=RUNWARE_API_KEY)

    # Connect to the Runware service
    await runware.connect()

    # The images for captioning. Can be UUIDs, URLs, base64, or file paths
    image_path = "retriever.jpg"

    # Example 1: Using new inputImages parameter with multiple images and custom prompts
    # request_image_to_text_payload = IImageCaption(
    #     inputImages=[image_path, "dalmatian.jpg"],  # Multiple images
    #     prompts=["Describe this image in detail", "What breed is this dog?"],  # Custom prompts
    #     includeCost=True,
    #     model="runware:150@1",  # AIR ID for image captioning model version 1
    # )
    
    # Example 2: Using new inputImages parameter with single image and default prompt
    # request_image_to_text_payload = IImageCaption(
    #     inputImages=[image_path],  # Single image in array
    #     includeCost=True,
    #     model="runware:150@2",  # AIR ID for image captioning model version 2
    # )
    
    # Example 3: Backward compatibility - using old inputImage parameter (mapped to inputImages[0])
    request_image_to_text_payload = IImageCaption(
        inputImage=image_path,  # Old parameter for backward compatibility
        includeCost=True,
        model="runware:150@1",
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
