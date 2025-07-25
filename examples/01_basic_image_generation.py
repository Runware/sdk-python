"""
Basic Image Generation Example

This example demonstrates how to generate images using the Runware SDK
with various configuration options and proper error handling.
"""

import asyncio
import os
from typing import List

from runware import IImage, IImageInference, Runware, RunwareError


async def basic_image_generation():
    """Generate a simple image with basic parameters."""

    # Initialize client with API key from environment
    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        # Connect to the Runware service
        await runware.connect()

        # Create image generation request
        request = IImageInference(
            positivePrompt="A majestic mountain landscape at sunset with vibrant colors",
            model="civitai:4384@128713",
            numberResults=1,
            height=1024,
            width=1024,
            negativePrompt="blurry, low quality, distorted",
            steps=30,
            CFGScale=7.5,
            seed=42,
        )

        # Generate images
        images: List[IImage] = await runware.imageInference(requestImage=request)

        # Process results
        for i, image in enumerate(images):
            print(f"Generated image {i + 1}:")
            print(f"  URL: {image.imageURL}")
            print(f"  UUID: {image.imageUUID}")
            if image.seed:
                print(f"  Seed: {image.seed}")
            if image.cost:
                print(f"  Cost: ${image.cost}")

    except RunwareError as e:
        print(f"Runware API Error: {e}")
        if hasattr(e, "code"):
            print(f"Error Code: {e.code}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        # Always disconnect when done
        await runware.disconnect()


async def batch_image_generation():
    """Generate multiple images with different configurations."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Generate multiple images with different aspect ratios
        requests = [
            IImageInference(
                positivePrompt="Portrait of a wise old wizard with a long beard",
                model="civitai:4384@128713",
                numberResults=1,
                height=1024,
                width=768,  # Portrait orientation
                negativePrompt="young, modern clothing",
            ),
            IImageInference(
                positivePrompt="Futuristic cityscape with flying cars and neon lights",
                model="civitai:4384@128713",
                numberResults=1,
                height=768,
                width=1024,  # Landscape orientation
                negativePrompt="old, vintage, medieval",
            ),
            IImageInference(
                positivePrompt="Cute cartoon animal in a magical forest",
                model="civitai:4384@128713",
                numberResults=2,
                height=1024,
                width=1024,  # Square format
                negativePrompt="realistic, dark, scary",
            ),
        ]

        # Process all requests concurrently
        tasks = [runware.imageInference(requestImage=req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"Request {i + 1} failed: {result}")
            else:
                images: List[IImage] = result
                print(f"Request {i + 1} completed with {len(images)} images:")
                for image in images:
                    print(f"  Image URL: {image.imageURL}")

    except RunwareError as e:
        print(f"Runware API Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
    finally:
        await runware.disconnect()


async def main():
    """Run all basic image generation examples."""
    print("=== Basic Image Generation ===")
    await basic_image_generation()

    print("\n=== Batch Image Generation ===")
    await batch_image_generation()


if __name__ == "__main__":
    asyncio.run(main())
