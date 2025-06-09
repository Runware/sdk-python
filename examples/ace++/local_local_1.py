import os

from runware import Runware, IImageInference, IAcePlusPlus


async def main() -> None:
    runware = Runware(
        api_key=os.getenv("RUNWARE_API_KEY"),
    )
    await runware.connect()
    mask_image = "https://raw.githubusercontent.com/ali-vilab/ACE_plus/refs/heads/main/assets/samples/local/local_1_m.webp"
    init_image = "https://raw.githubusercontent.com/ali-vilab/ACE_plus/refs/heads/main/assets/samples/local/local_1.webp"
    request_image = IImageInference(
        positivePrompt="By referencing the mask, restore a partial image from the doodle {image} that aligns with the textual explanation: \"1 white old owl\".",
        model="runware:102@1",
        height=1024,
        width=1024,
        numberResults=1,
        steps=28,
        CFGScale=50.0,
        acePlusPlus=IAcePlusPlus(
            inputImages=[init_image],
            inputMasks=[mask_image],
            repaintingScale=0.5,
            taskType="local_editing"
        ),
    )
    images = await runware.imageInference(requestImage=request_image)
    for image in images:
        print(f"Image URL: {image.imageURL}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
