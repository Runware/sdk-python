import asyncio
import os

from runware import Runware, IVideoInference, IFrameImage, IKlingAIProviderSettings, IKlingCameraControl, IKlingCameraConfig


async def main():
    runware = Runware(
        api_key=os.getenv("RUNWARE_API_KEY"),
    )
    await runware.connect()

    request = IVideoInference(
        positivePrompt="A majestic eagle soaring through mountain peaks at golden hour, cinematic view",
        model="klingai:1@1",
        width=1280,
        height=720,
        duration=5,
        numberResults=1,
        includeCost=True,
        CFGScale=1,
    )

    videos = await runware.videoInference(requestVideo=request)
    for video in videos:
        print(f"Video URL: {video.videoURL}")
        print(f"Cost: {video.cost}")
        print(f"Seed: {video.seed}")
        print(f"Status: {video.status}")


if __name__ == "__main__":
    asyncio.run(main())