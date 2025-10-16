import asyncio
import os

from runware import Runware, IVideoCaption, IVideoCaptionInputs


async def main():
    runware = Runware(
        api_key=os.getenv("RUNWARE_API_KEY"),
    )
    await runware.connect()

    request = IVideoCaption(
        model="memories:1@1",
        inputs=IVideoCaptionInputs(
            video="https://vm.runware.dev/video/ws/5/vi/b395d4bb-364c-44b5-91ba-f768e5ff3e98.mp4"
        ),
        deliveryMethod="async",
        includeCost=True
    )

    caption = await runware.videoCaption(requestVideoCaption=request)
    print(f"Caption: {caption.text}")
    print(f"Cost: {caption.cost}")
    print(f"Task UUID: {caption.taskUUID}")
    print(f"Status: {caption.status}")


if __name__ == "__main__":
    asyncio.run(main())
