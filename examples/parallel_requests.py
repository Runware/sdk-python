import asyncio
import os
import logging
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
    # Create an instance of RunwareServer
    runware = Runware(api_key=RUNWARE_API_KEY)

    # Connect to the Runware service
    await runware.connect()

    # Define the parameters for the first image request
    # TODO: For model_id=1, the server will not return a response
    # e.g. {'offset': 0, 'modelId': 1, 'promptText': 'a beautiful sunset over the ocean -no cloudy, rainy', 'numberResults': 2, 'sizeId': 1, 'taskType': 1, 'useCache': True, 'schedulerId': 22, 'gScale': 7}
    # TODO:  image_size actually takes an ID ("sizeId":int), not resolution, so maybe rename?

    request_image1 = IRequestImage(
        positive_prompt="a beautiful sunset over the mountains",
        image_size=1,
        model_id=13,
        number_of_images=5,
        negative_prompt="cloudy, rainy",
        use_cache=False,
        on_partial_images=on_partial_images,
    )

    request_image2 = IRequestImage(
        positive_prompt="a cozy hut in the woods",
        image_size=7,
        model_id=25,
        number_of_images=1,
        negative_prompt="modern, city",
        use_cache=True,
        on_partial_images=on_partial_images,
    )

    request_image3 = IRequestImage(
        positive_prompt="a wood workshop with tools and sawdust on the floor",
        image_size=7,
        model_id=25,
        number_of_images=4,
        use_cache=False,
    )

    first_images_request, second_images_request, third_image_request = (
        await asyncio.gather(
            runware.requestImages(requestImage=request_image1),
            runware.requestImages(requestImage=request_image2),
            runware.requestImages(requestImage=request_image3),
        )
    )

    print("\nFirst Image Request Results:")
    for image in first_images_request:
        print(f"Image URL: {image.imageSrc}")

    print("\nSecond Image Request Results:")
    for image in second_images_request:
        print(f"Image URL: {image.imageSrc}")

    print("\nThird Image Request Results:")
    for image in third_image_request:
        print(f"Image URL: {image.imageSrc}")


if __name__ == "__main__":
    asyncio.run(main())
