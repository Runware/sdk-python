import asyncio
import os

from runware import Runware, IVideoInference, IFrameImage, IPixverseProviderSettings


async def main():
    runware = Runware(
        api_key=os.getenv("RUNWARE_API_KEY"),
    )
    await runware.connect()

    request = IVideoInference(
        positivePrompt="realistic video, slow motion, cinematic, high quality, 4k, 60fps",
        negativePrompt="blurry",
        model="pixverse:1@1",
        width=1280,
        height=720,
        duration=5,
        fps=24,
        numberResults=1,
        seed=10,
        includeCost=True,
        frameImages=[
            IFrameImage(
                inputImage="https://raw.githubusercontent.com/adilentiq/test-images/refs/heads/main/common/man_beard.jpg",
            ),
            IFrameImage(
                inputImage="https://raw.githubusercontent.com/adilentiq/test-images/refs/heads/main/common/abraham_lincon.png",
            ),
        ],
        providerSettings=IPixverseProviderSettings(
            effect="boom drop",
            style="anime",
            motionMode="normal",
        )
    )

    videos = await runware.videoInference(requestVideo=request)
    for video in videos:
        print(f"Video URL: {video.videoURL}")
        print(f"Cost: {video.cost}")
        print(f"Seed: {video.seed}")
        print(f"Status: {video.status}")


if __name__ == "__main__":
    asyncio.run(main())