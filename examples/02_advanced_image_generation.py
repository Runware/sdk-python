"""
Advanced Image Generation Example

This example demonstrates advanced image generation features including:
- LoRA models for style enhancement
- ControlNet for guided generation
- Refiner models for quality improvement
- Accelerator options for faster inference
- Progress callbacks for real-time updates
"""

import asyncio
import os
from typing import List

from runware import (
    EControlMode,
    IAcceleratorOptions,
    IControlNetGeneral,
    IImage,
    IImageInference,
    ILora,
    IRefiner,
    ProgressUpdate,
    Runware,
    RunwareError,
)
import time


def progress_callback(progress: ProgressUpdate):
    """Handle progress updates during image generation."""
    print(f"Operation {progress.operation_id}: {progress.progress:.1%} complete")
    if progress.message:
        print(f"  Status: {progress.message}")
    if progress.partial_results:
        print(f"  Received {len(progress.partial_results)} partial results")


async def lora_enhanced_generation():
    """Generate images using LoRA models for enhanced styling."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Define LoRA models for style enhancement
        lora_models = [
            ILora(model="civitai:58390@62833", weight=0.8),
            ILora(model="civitai:42903@232848", weight=0.6),
        ]

        request = IImageInference(
            positivePrompt="masterpiece, best quality, 1girl, elegant dress, garden background, soft lighting",
            model="civitai:36520@76907",
            lora=lora_models,
            numberResults=3,
            height=1024,
            width=768,
            negativePrompt="worst quality, blurry, nsfw",
            steps=35,
            CFGScale=8.0,
            outputFormat="PNG",
            includeCost=True,
        )

        images: List[IImage] = await runware.imageInference(
            requestImage=request, progress_callback=progress_callback
        )

        print("LoRA-enhanced images generated:")
        for i, image in enumerate(images):
            print(f"  Image {i + 1}: {image.imageURL}")
            if image.cost:
                print(f"    Cost: ${image.cost}")

    except RunwareError as e:
        print(f"Error in LoRA generation: {e}")
    finally:
        await runware.disconnect()


async def controlnet_guided_generation():
    """Generate images using ControlNet for precise control."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Define ControlNet configuration
        controlnet = IControlNetGeneral(
            model="civitai:38784@44716",
            guideImage="https://huggingface.co/datasets/mishig/sample_images/resolve/main/canny-edge.jpg",
            weight=0.8,
            startStep=0,
            endStep=15,
            controlMode=EControlMode.BALANCED,
        )

        request = IImageInference(
            positivePrompt="beautiful anime character, detailed eyes, colorful hair",
            model="civitai:4384@128713",
            controlNet=[controlnet],
            numberResults=1,
            height=768,
            width=768,
            steps=30,
            CFGScale=7.0,
            seed=12345,
        )

        images: List[IImage] = await runware.imageInference(requestImage=request)

        print("ControlNet-guided images:")
        for image in images:
            print(f"  URL: {image.imageURL}")
            print(f"  Generated with seed: {image.seed}")

    except RunwareError as e:
        print(f"Error in ControlNet generation: {e}")
    finally:
        await runware.disconnect()


async def refiner_enhanced_generation():
    """Generate images with refiner model for quality enhancement."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Configure refiner model
        refiner = IRefiner(
            model="civitai:101055@128080",  # SDXL refiner
            startStep=25,  # Start refining after 25 steps
        )

        request = IImageInference(
            positivePrompt="hyperrealistic portrait of a astronaut in space, detailed helmet reflection",
            model="civitai:101055@128078",
            refiner=refiner,
            numberResults=1,
            height=1024,
            width=1024,
            steps=40,  # More steps for better quality with refiner
            CFGScale=7.5,
            outputFormat="PNG",
        )

        images: List[IImage] = await runware.imageInference(requestImage=request)

        print("Refiner-enhanced images:")
        for image in images:
            print(f"  High-quality URL: {image.imageURL}")

    except RunwareError as e:
        print(f"Error in refiner generation: {e}")
    finally:
        await runware.disconnect()


async def fast_generation_with_accelerators():
    """Generate images quickly using accelerator options."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Configure accelerator options for speed
        accelerator_options = IAcceleratorOptions(
            teaCache=True,
            teaCacheDistance=0.4,
            cacheStartStep=5,
            cacheStopStep=25,
        )

        request = IImageInference(
            positivePrompt="vibrant fantasy landscape with magical creatures",
            model="runware:100@1",  # Flux model that supports acceleration
            acceleratorOptions=accelerator_options,
            numberResults=1,
            height=1024,
            width=1024,
            steps=28,
            CFGScale=3.5,
        )

        start_time = time.time()

        images: List[IImage] = await runware.imageInference(
            requestImage=request, progress_callback=progress_callback
        )

        generation_time = time.time() - start_time

        print(f"Fast generation completed in {generation_time:.2f} seconds:")
        for image in images:
            print(f"  URL: {image.imageURL}")

    except RunwareError as e:
        print(f"Error in fast generation: {e}")
    finally:
        await runware.disconnect()


async def main():
    """Run all advanced image generation examples."""
    print("=== LoRA Enhanced Generation ===")
    await lora_enhanced_generation()

    print("\n=== ControlNet Guided Generation ===")
    await controlnet_guided_generation()

    print("\n=== Refiner Enhanced Generation ===")
    await refiner_enhanced_generation()

    print("\n=== Fast Generation with Accelerators ===")
    await fast_generation_with_accelerators()


if __name__ == "__main__":
    asyncio.run(main())
