import asyncio
import os
from typing import List, Optional
from dotenv import load_dotenv
from runware import Runware, IImage, IError, IRequestImage

# Load environment variables from .env file
load_dotenv()

RUNWARE_API_KEY = os.environ.get("RUNWARE_API_KEY")


# By providing the `onPartialImages` callback function, you can receive and handle the generated images incrementally,
# allowing for more responsive and interactive processing of the images.
# Note that the `onPartialImages` callback function is optional. If you don't provide it, the `requestImages`
# method will still return the complete list of generated images once the entire async request is finished.
def on_partial_images(images: List[IImage], error: Optional[IError]) -> None:
    if error:
        print(f"Error: {error.error_message}")
    else:
        print(f"Received {len(images)} partial images")
        for image in images:
            print(f"Partial Image URL: {image.imageSrc}")
            # Process or save the partial image as needed


async def main() -> None:
    # Initialize the Runware client
    runware = Runware(api_key=RUNWARE_API_KEY)

    # Define the parameters for the first image request
    request_image1 = IRequestImage(
        positive_prompt="a beautiful sunset over the ocean",
        image_size=512,
        model_id=1,
        number_of_images=2,
        negative_prompt="cloudy, rainy",
        use_cache=True,
        on_partial_images=on_partial_images,
    )

    request_image2 = IRequestImage(
        positive_prompt="a cozy cabin in the woods",
        image_size=512,
        model_id=1,
        number_of_images=1,
        negative_prompt="modern, city",
        use_cache=True,
        on_partial_images=on_partial_images,
    )

    first_images_request: List[IImage]
    second_images_request: List[IImage]
    first_images_request, second_images_request = await asyncio.gather(
        runware.requestImages(requestImage=request_image1),
        runware.requestImages(requestImage=request_image2),
    )

    print("First Image Request Results:")
    for image in first_images_request:
        print(f"Image URL: {image.imageSrc}")

    print("\nSecond Image Request Results:")
    for image in second_images_request:
        print(f"Image URL: {image.imageSrc}")


if __name__ == "__main__":
    asyncio.run(main())
