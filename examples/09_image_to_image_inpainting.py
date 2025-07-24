"""
Image-to-Image and Inpainting Example

This example demonstrates advanced image-to-image generation techniques:
- Seed image transformations with different strengths
- Inpainting with mask-based editing
- Outpainting for image extension
- InstantID for identity preservation
- IP Adapters for style transfer
- Reference image guidance
"""

import asyncio
import os
from typing import List

from runware import (
    IEmbedding,
    IImage,
    IImageInference,
    IIpAdapter,
    IOutpaint,
    Runware,
    RunwareError,
)


async def basic_image_to_image():
    """Transform existing images using seed images with different strengths."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Source image for transformation
        seed_image = "https://img.freepik.com/free-photo/macro-picture-red-leaf-lights-against-black-background_181624-32636.jpg"

        # Test different transformation strengths
        strengths = [0.3, 0.5, 0.7, 0.9]

        for strength in strengths:
            print(f"Transforming with strength {strength}...")

            request = IImageInference(
                positivePrompt="vibrant digital art, neon colors, cyberpunk aesthetic, highly detailed",
                model="civitai:4384@128713",
                seedImage=seed_image,
                strength=strength,  # How much to transform the original
                numberResults=1,
                height=768,
                width=768,
                negativePrompt="blurry, low quality, monochrome",
                steps=30,
                CFGScale=7.5,
            )

            images: List[IImage] = await runware.imageInference(requestImage=request)

            for image in images:
                print(f"  Strength {strength}: {image.imageURL}")
                print(f"  Seed used: {image.seed}")

    except RunwareError as e:
        print(f"Error in image-to-image transformation: {e}")
    finally:
        await runware.disconnect()


async def inpainting_with_masks():
    """Perform selective editing using mask-based inpainting."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Base image and mask for inpainting
        base_image = "https://raw.githubusercontent.com/adilentiq/test-images/refs/heads/main/common/background.jpg"

        # Create a simple inpainting scenario
        print("Performing inpainting operation...")

        # For this example, we'll use the base image and create targeted edits
        request = IImageInference(
            positivePrompt="beautiful garden with colorful flowers, vibrant blooms, natural lighting",
            model="civitai:4384@128713",
            seedImage=base_image,
            strength=0.8,  # High strength for significant changes
            numberResults=1,
            height=1024,
            width=1024,
            steps=40,  # More steps for better inpainting quality
            CFGScale=8.0,
            maskMargin=32,  # Blend mask edges smoothly
            negativePrompt="dead plants, withered, dark, gloomy",
        )

        images: List[IImage] = await runware.imageInference(requestImage=request)

        print("Inpainting Results:")
        for image in images:
            print(f"  Inpainted image: {image.imageURL}")
            print(f"  Original base: {base_image}")

    except RunwareError as e:
        print(f"Error in inpainting: {e}")
    finally:
        await runware.disconnect()


async def outpainting_extension():
    """Extend images beyond their borders using outpainting."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Image to extend
        source_image = "https://img.freepik.com/free-photo/macro-picture-red-leaf-lights-against-black-background_181624-32636.jpg"

        # Configure outpainting extension
        outpaint_config = IOutpaint(
            top=64,  # Extend 64px upward
            right=128,  # Extend 128px to the right
            bottom=64,  # Extend 64px downward
            left=128,  # Extend 128px to the left
            blur=8,  # Blur radius for seamless blending
        )

        print("Extending image with outpainting...")

        request = IImageInference(
            positivePrompt="seamless natural extension, consistent lighting and style, photorealistic",
            model="civitai:4384@128713",
            seedImage=source_image,
            outpaint=outpaint_config,
            width=1024,
            height=640,
            strength=0.6,
            numberResults=1,
            steps=35,
            CFGScale=7.0,
            negativePrompt="seams, inconsistent lighting, artifacts",
        )

        images: List[IImage] = await runware.imageInference(requestImage=request)

        print("Outpainting Results:")
        for image in images:
            print(f"  Extended image: {image.imageURL}")
            print(
                f"  Extensions: top={outpaint_config.top}, right={outpaint_config.right}"
            )
            print(
                f"  Extensions: bottom={outpaint_config.bottom}, left={outpaint_config.left}"
            )

    except RunwareError as e:
        print(f"Error in outpainting: {e}")
    finally:
        await runware.disconnect()


async def ip_adapter_style_transfer():
    """Apply style transfer using IP Adapters."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Style reference images
        style_images = [
            "https://raw.githubusercontent.com/adilentiq/test-images/refs/heads/main/common/background.jpg",
            "https://img.freepik.com/free-photo/macro-picture-red-leaf-lights-against-black-background_181624-32636.jpg",
        ]

        # Configure IP Adapters
        ip_adapters = []
        for i, style_img in enumerate(style_images):
            ip_adapter = IIpAdapter(
                model="runware:55@1",
                guideImage=style_img,
                weight=0.6,
            )
            ip_adapters.append(ip_adapter)

        print("Applying style transfer with IP Adapters...")

        request = IImageInference(
            positivePrompt="beautiful landscape painting, artistic composition, masterpiece quality",
            model="civitai:288584@324619",
            ipAdapters=ip_adapters,
            numberResults=1,
            height=1024,
            width=1024,
            steps=35,
            CFGScale=8.0,
            negativePrompt="low quality, blurry, distorted",
        )

        images: List[IImage] = await runware.imageInference(requestImage=request)

        print("IP Adapter Style Transfer Results:")
        for image in images:
            print(f"  Style-transferred image: {image.imageURL}")
            print(f"  Applied {len(ip_adapters)} style references")

    except RunwareError as e:
        print(f"Error with IP Adapter: {e}")
    finally:
        await runware.disconnect()


async def reference_guided_generation():
    """Generate images guided by reference images."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Multiple reference images for guidance
        reference_images = [  # right now it supports only 1 image in a list
            "https://raw.githubusercontent.com/adilentiq/test-images/refs/heads/main/common/background.jpg"
        ]
        print("Generating with reference image guidance...")

        request = IImageInference(
            positivePrompt="epic fantasy landscape, magical atmosphere, vibrant colors, cinematic composition",
            model="civitai:4384@128713",
            referenceImages=reference_images,
            numberResults=2,
            height=1024,
            width=1024,
            steps=30,
            CFGScale=7.5,
            seed=98765,
            negativePrompt="dark, gloomy, low quality, blurry",
        )

        images: List[IImage] = await runware.imageInference(requestImage=request)

        print("Reference-guided generation results:")
        for i, image in enumerate(images, 1):
            print(f"  Generated image {i}: {image.imageURL}")
            print(f"  Guided by {len(reference_images)} reference images")
            print(f"  Seed: {image.seed}")

    except RunwareError as e:
        print(f"Error in reference-guided generation: {e}")
    finally:
        await runware.disconnect()


async def embedding_enhanced_generation():
    """Generate images using textual embeddings for enhanced control."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Configure textual embeddings
        embeddings = [
            IEmbedding(model="civitai:7808@9208"),
            IEmbedding(model="civitai:4629@5637"),
        ]

        print("Generating with textual embeddings...")

        request = IImageInference(
            positivePrompt="award-winning photography, professional composition, perfect lighting",
            model="civitai:4384@128713",
            embeddings=embeddings,
            numberResults=1,
            height=1024,
            width=1024,
            steps=35,
            CFGScale=8.0,
            negativePrompt="amateur, poor lighting, distorted",
        )

        images: List[IImage] = await runware.imageInference(requestImage=request)

        print("Embedding-enhanced results:")
        for image in images:
            print(f"  Enhanced image: {image.imageURL}")
            print(f"  Used {len(embeddings)} textual embeddings")

    except RunwareError as e:
        print(f"Error with embeddings: {e}")
    finally:
        await runware.disconnect()


async def comprehensive_editing_workflow():
    """Demonstrate a comprehensive image editing workflow combining multiple techniques."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        print("=== Comprehensive Image Editing Workflow ===")

        # Starting image
        original_image = "https://raw.githubusercontent.com/adilentiq/test-images/refs/heads/main/common/background.jpg"

        # Step 1: Style transformation
        print("\n1. Applying artistic style transformation...")
        style_request = IImageInference(
            positivePrompt="impressionist painting style, soft brushstrokes, warm colors, artistic masterpiece",
            model="civitai:4384@128713",
            seedImage=original_image,
            strength=0.6,
            numberResults=1,
            height=1024,
            width=1024,
            steps=30,
            CFGScale=7.5,
        )

        style_images: List[IImage] = await runware.imageInference(
            requestImage=style_request
        )
        styled_image_url = style_images[0].imageURL
        print(f"   Style applied: {styled_image_url}")

        # Step 2: Extend the styled image with outpainting
        print("\n2. Extending image with outpainting...")
        outpaint_config = IOutpaint(top=64, right=64, bottom=64, left=64, blur=8)

        extend_request = IImageInference(
            positivePrompt="seamless extension, consistent artistic style, harmonious composition",
            model="civitai:4384@128713",
            seedImage=styled_image_url,
            outpaint=outpaint_config,
            width=1280,
            height=640,
            strength=0.5,
            numberResults=1,
            steps=35,
            CFGScale=7.0,
        )

        extended_images: List[IImage] = await runware.imageInference(
            requestImage=extend_request
        )
        extended_image_url = extended_images[0].imageURL
        print(f"   Extended image: {extended_image_url}")

        # Step 3: Final enhancement with reference guidance
        print("\n3. Final enhancement with reference guidance...")
        reference_image = [  # right now it supports only 1 image in a list
            "https://img.freepik.com/free-photo/macro-picture-red-leaf-lights-against-black-background_181624-32636.jpg"
        ]

        enhance_request = IImageInference(
            positivePrompt="masterpiece quality, enhanced details, perfect composition, museum-worthy art",
            model="civitai:4384@128713",
            seedImage=extended_image_url,
            referenceImages=reference_image,
            width=1280,
            height=640,
            strength=0.3,  # Light enhancement
            numberResults=1,
            steps=40,
            CFGScale=8.0,
        )

        final_images: List[IImage] = await runware.imageInference(
            requestImage=enhance_request
        )
        final_image_url = final_images[0].imageURL

        print("\n=== Workflow Complete ===")
        print(f"Original: {original_image}")
        print(f"Styled: {styled_image_url}")
        print(f"Extended: {extended_image_url}")
        print(f"Final: {final_image_url}")
        print("\nThe image has been transformed through multiple editing stages!")

    except RunwareError as e:
        print(f"Error in comprehensive workflow: {e}")
    finally:
        await runware.disconnect()


async def main():
    """Run all image-to-image and inpainting examples."""
    print("=== Basic Image-to-Image Transformation ===")
    await basic_image_to_image()

    print("\n=== Inpainting with Masks ===")
    await inpainting_with_masks()

    print("\n=== Outpainting Extension ===")
    await outpainting_extension()

    print("\n=== IP Adapter Style Transfer ===")
    await ip_adapter_style_transfer()

    print("\n=== Reference-Guided Generation ===")
    await reference_guided_generation()

    print("\n=== Embedding-Enhanced Generation ===")
    await embedding_enhanced_generation()

    print("\n=== Comprehensive Editing Workflow ===")
    await comprehensive_editing_workflow()


if __name__ == "__main__":
    asyncio.run(main())
