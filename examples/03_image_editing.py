"""
Image Editing Operations Example

This example demonstrates various image editing capabilities:
- Image captioning for description generation
- Background removal with custom settings
- Image upscaling for resolution enhancement
- File upload and processing workflows
"""

import asyncio
import os
from typing import List

from runware import (
    IBackgroundRemovalSettings,
    IImage,
    IImageBackgroundRemoval,
    IImageCaption,
    IImageToText,
    IImageUpscale,
    Runware,
    RunwareError,
)


async def image_captioning():
    """Generate descriptive captions for images."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Test image URL
        image_url = "https://img.freepik.com/free-photo/macro-picture-red-leaf-lights-against-black-background_181624-32636.jpg"

        # Create captioning request
        caption_request = IImageCaption(inputImage=image_url, includeCost=True)

        # Generate caption
        result: IImageToText = await runware.imageCaption(
            requestImageToText=caption_request
        )

        print("Image Caption Analysis:")
        print(f"  Image: {image_url}")
        print(f"  Description: {result.text}")
        if result.cost:
            print(f"  Processing cost: ${result.cost}")

    except RunwareError as e:
        print(f"Error in image captioning: {e}")
    finally:
        await runware.disconnect()


async def background_removal_basic():
    """Remove background from image using default settings."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Image with clear subject for background removal
        image_url = "https://raw.githubusercontent.com/adilentiq/test-images/refs/heads/main/common/headphones.jpeg"

        # Basic background removal request
        removal_request = IImageBackgroundRemoval(
            inputImage=image_url, outputType="URL", outputFormat="PNG", includeCost=True
        )

        # Process image
        processed_images: List[IImage] = await runware.imageBackgroundRemoval(
            removeImageBackgroundPayload=removal_request
        )

        print("Basic Background Removal:")
        for i, image in enumerate(processed_images):
            print(f"  Result {i + 1}: {image.imageURL}")
            if image.cost:
                print(f"    Cost: ${image.cost}")

    except RunwareError as e:
        print(f"Error in background removal: {e}")
    finally:
        await runware.disconnect()


async def background_removal_advanced():
    """Remove background with advanced settings and custom model."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        image_url = "https://raw.githubusercontent.com/adilentiq/test-images/refs/heads/main/common/headphones.jpeg"

        # Advanced background removal settings
        advanced_settings = IBackgroundRemovalSettings(
            rgba=[255, 255, 255, 0],  # White transparent background
            alphaMatting=True,  # Better edge quality
            postProcessMask=True,  # Refine mask edges
            returnOnlyMask=False,  # Return processed image, not just mask
            alphaMattingErodeSize=10,
            alphaMattingForegroundThreshold=240,
            alphaMattingBackgroundThreshold=10,
        )

        # Request with custom settings
        removal_request = IImageBackgroundRemoval(
            inputImage=image_url,
            settings=advanced_settings,
            outputType="URL",
            outputFormat="PNG",
            outputQuality=95,
            includeCost=True,
        )

        processed_images: List[IImage] = await runware.imageBackgroundRemoval(
            removeImageBackgroundPayload=removal_request
        )

        print("Advanced Background Removal:")
        for image in processed_images:
            print(f"  High-quality result: {image.imageURL}")
            if image.cost:
                print(f"  Cost: ${image.cost}")

    except RunwareError as e:
        print(f"Error in advanced background removal: {e}")
    finally:
        await runware.disconnect()


async def image_upscaling():
    """Enhance image resolution using upscaling."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Lower resolution image for upscaling demonstration
        image_url = "https://img.freepik.com/free-photo/macro-picture-red-leaf-lights-against-black-background_181624-32636.jpg"

        # Test different upscale factors
        upscale_factors = [2, 4]

        for factor in upscale_factors:
            upscale_request = IImageUpscale(
                inputImage=image_url,
                upscaleFactor=factor,
                outputType="URL",
                outputFormat="PNG",
                includeCost=True,
            )

            upscaled_images: List[IImage] = await runware.imageUpscale(
                upscaleGanPayload=upscale_request
            )

            print(f"Upscaling {factor}x:")
            for image in upscaled_images:
                print(f"  Enhanced image: {image.imageURL}")
                if image.cost:
                    print(f"  Processing cost: ${image.cost}")

    except RunwareError as e:
        print(f"Error in image upscaling: {e}")
    finally:
        await runware.disconnect()


async def complete_editing_workflow():
    """Demonstrate a complete image editing workflow."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        original_image = "https://raw.githubusercontent.com/adilentiq/test-images/refs/heads/main/common/headphones.jpeg"

        print("Starting complete editing workflow...")

        # Step 1: Generate caption for the original image
        print("Step 1: Analyzing image content...")
        caption_request = IImageCaption(inputImage=original_image)
        caption_result: IImageToText = await runware.imageCaption(
            requestImageToText=caption_request
        )
        print(f"  Original image description: {caption_result.text}")

        # Step 2: Remove background
        print("Step 2: Removing background...")
        bg_removal_request = IImageBackgroundRemoval(
            inputImage=original_image, outputType="URL", outputFormat="PNG"
        )
        bg_removed_images: List[IImage] = await runware.imageBackgroundRemoval(
            removeImageBackgroundPayload=bg_removal_request
        )
        background_removed_url = bg_removed_images[0].imageURL
        print(f"  Background removed: {background_removed_url}")

        # Step 3: Upscale the result
        print("Step 3: Enhancing resolution...")
        upscale_request = IImageUpscale(
            inputImage=background_removed_url,
            upscaleFactor=2,
            outputType="URL",
            outputFormat="PNG",
        )
        upscaled_images: List[IImage] = await runware.imageUpscale(
            upscaleGanPayload=upscale_request
        )
        final_image_url = upscaled_images[0].imageURL
        print(f"  Final enhanced image: {final_image_url}")

        # Step 4: Generate caption for final result
        print("Step 4: Analyzing final result...")
        final_caption_request = IImageCaption(inputImage=final_image_url)
        final_caption: IImageToText = await runware.imageCaption(
            requestImageToText=final_caption_request
        )
        print(f"  Final image description: {final_caption.text}")

        print("\nWorkflow completed successfully!")
        print(f"Original: {original_image}")
        print(f"Final: {final_image_url}")

    except RunwareError as e:
        print(f"Error in editing workflow: {e}")
    finally:
        await runware.disconnect()


async def main():
    """Run all image editing examples."""
    print("=== Image Captioning ===")
    await image_captioning()

    print("\n=== Basic Background Removal ===")
    await background_removal_basic()

    print("\n=== Advanced Background Removal ===")
    await background_removal_advanced()

    print("\n=== Image Upscaling ===")
    await image_upscaling()

    print("\n=== Complete Editing Workflow ===")
    await complete_editing_workflow()


if __name__ == "__main__":
    asyncio.run(main())
