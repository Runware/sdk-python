"""
ACE++ Advanced Character Editing Example

ACE++ (Advanced Character Edit) enables character-consistent image generation
and editing while preserving identity. This example demonstrates:
- Portrait editing with identity preservation
- Subject integration and replacement
- Local editing with masks
- Logo and object placement
- Movie poster style editing
"""

import asyncio
import os
from typing import List

from runware import IAcePlusPlus, IImage, IImageInference, Runware, RunwareError


async def logo_placement():
    """Place logos and branding elements on products using masks."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Assets for logo placement
        reference_image = "https://raw.githubusercontent.com/ali-vilab/ACE_plus/refs/heads/main/assets/samples/application/logo_paste/1_ref.png"
        mask_image = "https://raw.githubusercontent.com/ali-vilab/ACE_plus/refs/heads/main/assets/samples/application/logo_paste/1_1_m.png"
        init_image = "https://raw.githubusercontent.com/ali-vilab/ACE_plus/refs/heads/main/assets/samples/application/logo_paste/1_1_edit.png"

        request = IImageInference(
            positivePrompt="The logo is printed on the headphones with high quality and proper lighting.",
            model="runware:102@1",
            height=1024,
            width=1024,
            numberResults=1,
            steps=28,
            CFGScale=50.0,
            referenceImages=[reference_image],  # Logo reference
            acePlusPlus=IAcePlusPlus(
                inputImages=[init_image],  # Product image
                inputMasks=[mask_image],  # Mask for logo placement area
                repaintingScale=1.0,  # Full prompt adherence for placement
                taskType="subject",  # Subject placement task
            ),
        )

        images: List[IImage] = await runware.imageInference(requestImage=request)

        print("Logo Placement:")
        for image in images:
            print(f"  Product with logo: {image.imageURL}")
            print(f"  Logo professionally integrated using mask guidance")

    except RunwareError as e:
        print(f"Error in logo placement: {e}")
    finally:
        await runware.disconnect()


async def local_region_editing():
    """Edit specific regions of images using local editing masks."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Assets for local editing
        mask_image = "https://raw.githubusercontent.com/ali-vilab/ACE_plus/refs/heads/main/assets/samples/local/local_1_m.webp"
        init_image = "https://raw.githubusercontent.com/ali-vilab/ACE_plus/refs/heads/main/assets/samples/local/local_1.webp"

        request = IImageInference(
            positivePrompt='By referencing the mask, restore a partial image from the doodle that aligns with the textual explanation: "1 white old owl".',
            model="runware:102@1",
            height=1024,
            width=1024,
            numberResults=1,
            steps=28,
            CFGScale=50.0,
            acePlusPlus=IAcePlusPlus(
                inputImages=[init_image],  # Image to edit
                inputMasks=[mask_image],  # Local region mask
                repaintingScale=0.5,  # Balanced editing
                taskType="local_editing",  # Local editing mode
            ),
        )

        images: List[IImage] = await runware.imageInference(requestImage=request)

        print("Local Region Editing:")
        for image in images:
            print(f"  Locally edited image: {image.imageURL}")
            print(f"  Specific region refined while preserving surrounding areas")

    except RunwareError as e:
        print(f"Error in local editing: {e}")
    finally:
        await runware.disconnect()


async def movie_poster_editing():
    """Create movie poster style edits with character replacement."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Movie poster editing assets
        reference_image = "https://raw.githubusercontent.com/ali-vilab/ACE_plus/refs/heads/main/assets/samples/application/movie_poster/1_ref.png"
        mask_image = "https://raw.githubusercontent.com/ali-vilab/ACE_plus/refs/heads/main/assets/samples/application/movie_poster/1_1_m.png"
        init_image = "https://raw.githubusercontent.com/ali-vilab/ACE_plus/refs/heads/main/assets/samples/application/movie_poster/1_1_edit.png"

        request = IImageInference(
            positivePrompt="The man is facing the camera and is smiling with confidence and charisma, perfect for a movie poster.",
            model="runware:102@1",
            height=768,
            width=1024,
            numberResults=1,
            steps=28,
            CFGScale=50.0,
            referenceImages=[reference_image],  # Character reference
            acePlusPlus=IAcePlusPlus(
                inputImages=[init_image],  # Poster template
                inputMasks=[mask_image],  # Character replacement area
                repaintingScale=1.0,  # Full creative freedom in masked area
                taskType="portrait",  # Portrait-aware processing
            ),
        )

        images: List[IImage] = await runware.imageInference(requestImage=request)

        print("Movie Poster Editing:")
        for image in images:
            print(f"  Movie poster: {image.imageURL}")
            print(f"  Character seamlessly integrated into poster design")

    except RunwareError as e:
        print(f"Error in movie poster editing: {e}")
    finally:
        await runware.disconnect()


async def photo_editing_workflow():
    """Demonstrate a complex photo editing workflow using ACE++."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Professional photo editing assets
        init_image = "https://raw.githubusercontent.com/ali-vilab/ACE_plus/refs/heads/main/assets/samples/application/photo_editing/1_1_edit.png"
        mask_image = "https://raw.githubusercontent.com/ali-vilab/ACE_plus/refs/heads/main/assets/samples/application/photo_editing/1_1_m.png"
        reference_image = "https://raw.githubusercontent.com/ali-vilab/ACE_plus/refs/heads/main/assets/samples/application/photo_editing/1_ref.png"

        request = IImageInference(
            positivePrompt="The item is put on the ground with proper lighting and realistic shadows, professional product photography.",
            model="runware:102@1",
            height=1024,
            width=1024,
            numberResults=1,
            steps=28,
            CFGScale=50.0,
            referenceImages=[reference_image],  # Product reference
            acePlusPlus=IAcePlusPlus(
                inputImages=[init_image],  # Scene to edit
                inputMasks=[mask_image],  # Product placement area
                repaintingScale=1.0,  # Full control over placement
                taskType="subject",  # Subject placement
            ),
        )

        images: List[IImage] = await runware.imageInference(requestImage=request)

        print("Professional Photo Editing:")
        for image in images:
            print(f"  Edited photo: {image.imageURL}")
            print(f"  Product professionally integrated with realistic lighting")

    except RunwareError as e:
        print(f"Error in photo editing: {e}")
    finally:
        await runware.disconnect()


async def main():
    """Run all ACE++ advanced editing examples."""
    print("\n=== Logo Placement ===")
    await logo_placement()

    print("\n=== Local Region Editing ===")
    await local_region_editing()

    print("\n=== Movie Poster Editing ===")
    await movie_poster_editing()

    print("\n=== Professional Photo Editing ===")
    await photo_editing_workflow()


if __name__ == "__main__":
    asyncio.run(main())
