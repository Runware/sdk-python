import os

from runware import Runware, IImageInference, IAcePlusPlus


async def main() -> None:
    runware = Runware(
        api_key=os.getenv("RUNWARE_API_KEY"),
    )
    await runware.connect()

    reference_image = "https://raw.githubusercontent.com/ali-vilab/ACE_plus/refs/heads/main/assets/samples/subject/subject_1.jpg"
    request_image = IImageInference(
        positivePrompt="Display the logo in a minimalist style printed in white on a matte black ceramic coffee mug, alongside a steaming cup of coffee on a cozy cafe table.",
        model="runware:102@1",
        height=1024,
        width=1024,
        seed=2935362780,
        numberResults=1,
        steps=28,
        CFGScale=50.0,
        referenceImages=[reference_image],
        acePlusPlus=IAcePlusPlus(
            repaintingScale=1,
            taskType="subject"
        ),
    )
    images = await runware.imageInference(requestImage=request_image)
    for image in images:
        print(f"Image URL: {image.imageURL}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
