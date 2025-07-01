import asyncio
import os

from runware import Runware, IVideoInference, IFrameImage, IBytedanceProviderSettings


async def main():
    runware = Runware(
        api_key=os.getenv("RUNWARE_API_KEY"),
    )
    await runware.connect()

    request = IVideoInference(
        positivePrompt=" couple in formal evening attire is caught in heavy rain on their way home, holding a black umbrella. In the flat shot, the man is wearing a black suit and the woman is wearing a white long dress. They walk slowly in the rain, and the rain drips down the umbrella. The camera moves smoothly with their steps, showing their elegant posture in the rain.",
        model="bytedance:1@1",
        height=1504,  # Comment this to use i2v
        width=640,  # Comment this to use i2v
        duration=5,
        numberResults=1,
        seed=10,
        includeCost=True,
        # frameImages=[  # Uncomment this to use i2v
        #     IFrameImage(
        #         inputImage="https://raw.githubusercontent.com/adilentiq/test-images/refs/heads/main/common/background.jpg",
        #     ),
        # ],
        providerSettings=IBytedanceProviderSettings(
            cameraFixed=False
        )
    )

    videos = await runware.videoInference(requestVideo=request)
    for video in videos:
        print(f"Video URL: {video.videoURL}")
        print(f"Cost: {video.cost}")
        print(f"Seed: {video.seed}")


if __name__ == "__main__":
    asyncio.run(main())