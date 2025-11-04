import asyncio
import os

from runware import Runware, IVideoBackgroundRemoval, IVideoBackgroundRemovalInputs, IVideoBackgroundRemovalSettings


async def main():
    runware = Runware(
        api_key=os.getenv("RUNWARE_API_KEY"),
    )
    await runware.connect()

    request = IVideoBackgroundRemoval(
        model="bria:51@1",
        inputs=IVideoBackgroundRemovalInputs(
            video="https://example.com/video.mp4"
        ),
        outputFormat="WEBM",
        includeCost=True,
        settings=IVideoBackgroundRemovalSettings(
            rgba=[255, 255, 255, 0]
        )
    )

    videos = await runware.videoBackgroundRemoval(requestVideoBackgroundRemoval=request)
    for video in videos:
        print(f"Video URL: {video.videoURL}")
        print(f"Cost: {video.cost}")
        print(f"Task UUID: {video.taskUUID}")
        print(f"Status: {video.status}")


if __name__ == "__main__":
    asyncio.run(main())
