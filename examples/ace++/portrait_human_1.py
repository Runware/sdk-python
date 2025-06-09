import os

from runware import Runware, IImageInference, IAcePlusPlus


async def main() -> None:
    runware = Runware(
        api_key=os.getenv("RUNWARE_API_KEY"),
    )
    await runware.connect()

    reference_image = "https://raw.githubusercontent.com/ali-vilab/ACE_plus/refs/heads/main/assets/samples/portrait/human_1.jpg"
    request_image = IImageInference(
        positivePrompt="Maintain the facial features, A girl is wearing a neat police uniform and sporting a badge. She is smiling with a friendly and confident demeanor. The background is blurred, featuring a cartoon logo.",
        model="runware:102@1",
        height=1024,
        width=1024,
        seed=4194866942,
        numberResults=1,
        steps=28,
        CFGScale=50.0,
        referenceImages=[reference_image],
        acePlusPlus=IAcePlusPlus(
            repaintingScale=0.5,
            taskType="portrait"
        ),
    )
    images = await runware.imageInference(requestImage=request_image)
    for image in images:
        print(f"Image URL: {image.imageURL}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
