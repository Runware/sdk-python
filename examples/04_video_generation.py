"""
Video Generation Example

This example demonstrates video generation using different AI providers:
- Google Veo for high-quality cinematic videos
- Kling AI for creative video generation
- Minimax for text-to-video and image-to-video
- Bytedance for professional video content
- Pixverse for stylized video generation
- Vidu for anime-style videos
"""

import asyncio
import os
from typing import List

from runware import (
    IBytedanceProviderSettings,
    IFrameImage,
    IGoogleProviderSettings,
    IMinimaxProviderSettings,
    IPixverseProviderSettings,
    IVideo,
    IVideoInference,
    IViduProviderSettings,
    Runware,
    RunwareError,
)


async def google_veo_generation():
    """Generate videos using Google's Veo model with advanced features."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Text-to-video generation
        print("Generating text-to-video with Google Veo...")
        text_to_video_request = IVideoInference(
            positivePrompt="A majestic eagle soaring through mountain peaks at golden hour, cinematic cinematography",
            model="google:3@0",
            width=1280,
            height=720,
            duration=8,
            numberResults=1,
            seed=42,
            includeCost=True,
            providerSettings=IGoogleProviderSettings(
                generateAudio=True, enhancePrompt=True
            ),
        )

        videos: List[IVideo] = await runware.videoInference(
            requestVideo=text_to_video_request
        )

        for video in videos:
            print(f"  Generated video: {video.videoURL}")
            if video.cost:
                print(f"  Cost: ${video.cost}")
            print(f"  Seed used: {video.seed}")

        # Image-to-video generation
        print("\nGenerating image-to-video with Google Veo...")
        image_to_video_request = IVideoInference(
            positivePrompt="The galaxy slowly rotates with sparkling stars",
            model="google:2@0",
            width=1280,
            height=720,
            duration=5,
            numberResults=1,
            frameImages=[
                IFrameImage(
                    inputImage="https://github.com/adilentiq/test-images/blob/main/common/image_15_mb.jpg?raw=true"
                )
            ],
            includeCost=True,
        )

        videos = await runware.videoInference(requestVideo=image_to_video_request)

        for video in videos:
            print(f"  I2V result: {video.videoURL}")
            if video.cost:
                print(f"  Cost: ${video.cost}")

    except RunwareError as e:
        print(f"Error with Google Veo: {e}")
    finally:
        await runware.disconnect()


async def kling_ai_generation():
    """Generate videos using Kling AI"""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        request = IVideoInference(
            positivePrompt="A beautiful woman portrait",
            model="klingai:1@2",
            width=1280,
            height=720,
            duration=5,
            numberResults=1,
            CFGScale=1.0,
            includeCost=True,
            frameImages=[
                IFrameImage(
                    inputImage="https://huggingface.co/ntc-ai/SDXL-LoRA-slider.Studio-Ghibli-style/resolve/main/images/Studio%20Ghibli%20style_17_-1.5.png",
                    frame="first",
                )
            ],
        )

        videos: List[IVideo] = await runware.videoInference(requestVideo=request)

        print("Kling AI video:")
        for video in videos:
            print(f"  Video URL: {video.videoURL}")
            print(f"  Status: {video.status}")
            if video.cost:
                print(f"  Cost: ${video.cost}")

    except RunwareError as e:
        print(f"Error with Kling AI: {e}")
    finally:
        await runware.disconnect()


async def minimax_generation():
    """Generate videos using Minimax with prompt optimization."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        request = IVideoInference(
            positivePrompt="A stylish woman walks down a Tokyo street filled with warm glowing neon and animated city signage",
            model="minimax:1@1",
            width=1366,
            height=768,
            duration=6,
            numberResults=1,
            seed=12345,
            includeCost=True,
            providerSettings=IMinimaxProviderSettings(
                promptOptimizer=True  # Enhance prompt automatically
            ),
        )

        videos: List[IVideo] = await runware.videoInference(requestVideo=request)

        print("Minimax video generation:")
        for video in videos:
            print(f"  Video URL: {video.videoURL}")
            if video.cost:
                print(f"  Cost: ${video.cost}")
            print(f"  Generated with seed: {video.seed}")

    except RunwareError as e:
        print(f"Error with Minimax: {e}")
    finally:
        await runware.disconnect()


async def bytedance_generation():
    """Generate videos using Bytedance with professional settings."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        request = IVideoInference(
            positivePrompt="A couple in formal evening attire walking in heavy rain with an umbrella, cinematic lighting",
            model="bytedance:1@1",
            height=1504,
            width=640,
            duration=5,
            numberResults=1,
            seed=98765,
            includeCost=True,
            providerSettings=IBytedanceProviderSettings(
                cameraFixed=False,  # Allow camera movement
            ),
        )

        videos: List[IVideo] = await runware.videoInference(requestVideo=request)

        print("Seedance 1.0 Lite video:")
        for video in videos:
            print(f"  Video URL: {video.videoURL}")
            if video.cost:
                print(f"  Cost: ${video.cost}")

    except RunwareError as e:
        print(f"Error with Bytedance: {e}")
    finally:
        await runware.disconnect()


async def pixverse_stylized_generation():
    """Generate stylized videos using Pixverse with effects and styles."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        request = IVideoInference(
            positivePrompt="A magical transformation sequence with sparkles and light effects",
            negativePrompt="blurry, low quality",
            model="pixverse:1@2",
            width=1280,
            height=720,
            duration=5,
            fps=24,
            numberResults=1,
            seed=55555,
            includeCost=True,
            frameImages=[
                IFrameImage(
                    inputImage="https://raw.githubusercontent.com/adilentiq/test-images/refs/heads/main/common/man_beard.jpg"
                )
            ],
            providerSettings=IPixverseProviderSettings(
                effect="boom drop",  # Special effect
                style="anime",  # Anime art style
                motionMode="normal",  # Motion intensity
            ),
        )

        videos: List[IVideo] = await runware.videoInference(requestVideo=request)

        print("Pixverse stylized video:")
        for video in videos:
            print(f"  Anime-style video: {video.videoURL}")
            print(f"  Status: {video.status}")
            if video.cost:
                print(f"  Cost: ${video.cost}")

    except RunwareError as e:
        print(f"Error with Pixverse: {e}")
    finally:
        await runware.disconnect()


async def vidu_anime_generation():
    """Generate anime-style videos using Vidu."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        request = IVideoInference(
            positivePrompt="A red fox moves stealthily through autumn woods, hunting for prey",
            model="vidu:1@5",
            width=1920,
            height=1080,
            duration=4,
            numberResults=1,
            seed=77777,
            includeCost=True,
            providerSettings=IViduProviderSettings(
                style="anime",  # Anime art style
                movementAmplitude="auto",  # Automatic movement detection
                bgm=True,  # Add background music
            ),
        )

        videos: List[IVideo] = await runware.videoInference(requestVideo=request)

        print("Vidu anime-style video:")
        for video in videos:
            print(f"  Anime video with BGM: {video.videoURL}")
            print(f"  Status: {video.status}")
            if video.cost:
                print(f"  Cost: ${video.cost}")

    except RunwareError as e:
        print(f"Error with Vidu: {e}")
    finally:
        await runware.disconnect()


async def main():
    """Run video generation examples for different providers."""
    print("=== Google Veo Video Generation ===")
    await google_veo_generation()

    print("\n=== Kling AI ===")
    await kling_ai_generation()

    print("\n=== Minimax with Prompt Optimization ===")
    await minimax_generation()

    print("\n=== Bytedance Video ===")
    await bytedance_generation()

    print("\n=== Pixverse Stylized Video ===")
    await pixverse_stylized_generation()
    #
    print("\n=== Vidu Anime-Style Video ===")
    await vidu_anime_generation()


if __name__ == "__main__":
    asyncio.run(main())
