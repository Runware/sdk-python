from runware import Runware, RunwareAPIError,IImage, IImageBackgroundRemoval
import asyncio
import os
from dotenv import load_dotenv

load_dotenv(override=True)


async def main() -> None:
    runware = Runware(
        api_key=os.environ.get("RUNWARE_API_KEY"),
    )
    await runware.connect()

    request_image = IImageBackgroundRemoval(
        taskUUID="abcdbb9c-3bd3-4d75-9234-bffeef994772",
        model="runware:110@1",
        inputImage="https://raw.githubusercontent.com/adilentiq/test-images/refs/heads/main/common/headphones.jpeg"
    )

    print(f"Payload: {request_image}")
    try:
        processed_images: List[IImage] = await runware.imageBackgroundRemoval(
            removeImageBackgroundPayload=request_image
        )
    except RunwareAPIError as e:
        print(f"API Error: {e}")
        print(f"Error Code: {e.code}")
    except Exception as e:
        print(f"Unexpected Error: {e}")
    else:
        print("Processed Image with the background removed:")
        print(processed_images)
        for image in processed_images:
            print(image.imageURL)


asyncio.run(main())