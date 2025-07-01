import asyncio
import os

from runware import Runware, IVideoInference, IGoogleProviderSettings, IFrameImage


async def main():
    runware = Runware(
        api_key=os.getenv("RUNWARE_API_KEY"),
    )
    await runware.connect()

    request = IVideoInference(
        positivePrompt="spinning galaxy",
        model="google:3@0",
        width=1280,
        height=720,
        numberResults=1,
        seed=10,
        includeCost=True,
        frameImages=[ # Comment this to use t2v
            IFrameImage(
                inputImage="https://github.com/adilentiq/test-images/blob/main/common/image_15_mb.jpg?raw=true",
            ),
        ],
        providerSettings=IGoogleProviderSettings(  # Needs only for veo3
            generateAudio=True,
            enhancePrompt=True
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
