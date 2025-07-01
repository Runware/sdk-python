import asyncio
import os

from runware import Runware, IVideoInference, IFrameImage, IMinimaxProviderSettings


async def main():
    runware = Runware(
        api_key=os.getenv("RUNWARE_API_KEY"),
    )
    await runware.connect()

    request = IVideoInference(
        positivePrompt="[Push in, Follow]A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage. She wears a black leather jacket, a long red dress, and black boots, and carries a black purse.[Pan left] The street opens into a small plaza where street vendors sell steaming food under colorful awnings.",
        model="minimax:1@1",
        width=1366,  # Comment this to use i2v
        height=768,  # Comment this to use i2v
        duration=6,
        numberResults=1,
        seed=10,
        includeCost=True,
        # frameImages=[  # Uncomment this to use t2v
        #     IFrameImage(
        #         inputImage= "https://raw.githubusercontent.com/adilentiq/test-images/refs/heads/main/video_inference/woman_city.png",
        #     ),
        # ]
        providerSettings=IMinimaxProviderSettings(
            promptOptimizer=True
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