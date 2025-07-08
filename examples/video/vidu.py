import asyncio
import os

from runware import Runware, IVideoInference, IFrameImage, IViduProviderSettings


async def main():
    runware = Runware(
        api_key=os.getenv("RUNWARE_API_KEY"),
    )
    await runware.connect()

    request = IVideoInference(
        positivePrompt="A red fox moves stealthily through autumn woods, hunting for prey.",
        model="vidu:1@1",
        width=1920,
        height=1080,
        duration=5,
        numberResults=1,
        seed=10,
        includeCost=True,
        providerSettings=IViduProviderSettings(
            style="anime",
            movementAmplitude="auto",
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