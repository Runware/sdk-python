import os

from runware import Runware, IImageInference, IAcePlusPlus


async def main() -> None:
    runware = Runware(
        api_key=os.getenv("RUNWARE_API_KEY"),
    )
    await runware.connect()

    reference_image = "https://raw.githubusercontent.com/ali-vilab/ACE_plus/refs/heads/main/assets/samples/application/logo_paste/1_ref.png"
    mask_image = "https://raw.githubusercontent.com/ali-vilab/ACE_plus/refs/heads/main/assets/samples/application/logo_paste/1_1_m.png"
    init_image = "https://raw.githubusercontent.com/ali-vilab/ACE_plus/refs/heads/main/assets/samples/application/logo_paste/1_1_edit.png"
    request_image = IImageInference(
        positivePrompt="The logo is printed on the headphones.",
        model="runware:102@1",
        taskUUID="68020b8f-bbcf-4779-ba51-4f3bb00aef6a",
        height=1024,
        width=1024,
        numberResults=1,
        steps=28,
        CFGScale=50.0,
        referenceImages=[reference_image],
        acePlusPlus=IAcePlusPlus(
            inputImages=[init_image],
            inputMasks=[mask_image],
            repaintingScale=1.0,
            taskType="subject"
        ),
    )
    images = await runware.imageInference(requestImage=request_image)
    for image in images:
        print(f"Image URL: {image.imageURL}")


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
