"""
PhotoMaker Example

PhotoMaker enables identity-consistent photo generation by combining
multiple input photos to create new images while preserving identity.
This example demonstrates various PhotoMaker use cases and styles.
"""

import asyncio
import os
from typing import List

from runware import IImage, IPhotoMaker, Runware, RunwareError, UploadImageType


async def basic_photo_maker():
    """Generate photos using PhotoMaker with multiple input images."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Sample input images for identity reference
        input_images = [
            "https://im.runware.ai/image/ws/0.5/ii/74723926-22f6-417c-befb-f2058fc88c13.webp",
            "https://im.runware.ai/image/ws/0.5/ii/64acee31-100d-4aa1-a47e-6f8b432e7188.webp",
            "https://im.runware.ai/image/ws/0.5/ii/1b39b0e0-6bf7-4c9a-8134-c0251b5ede01.webp",
            "https://im.runware.ai/image/ws/0.5/ii/f4b4cec3-66d9-4c02-97c5-506b8813182a.webp",
        ]

        request = IPhotoMaker(
            model="civitai:139562@344487",  # PhotoMaker compatible model
            positivePrompt="img of a beautiful lady in a peaceful forest setting, natural lighting",
            steps=35,
            numberResults=2,
            height=768,
            width=512,
            style="No style",  # Natural style
            strength=40,
            outputFormat="WEBP",
            includeCost=True,
            inputImages=input_images,
        )

        photos: List[IImage] = await runware.photoMaker(requestPhotoMaker=request)

        print("Basic PhotoMaker Results:")
        for i, photo in enumerate(photos):
            print(f"  Photo {i + 1}: {photo.imageURL}")
            if photo.cost:
                print(f"    Cost: ${photo.cost}")
            if photo.seed:
                print(f"    Seed: {photo.seed}")

    except RunwareError as e:
        print(f"Error in basic PhotoMaker: {e}")
    finally:
        await runware.disconnect()


async def styled_photo_generation():
    """Generate photos with different artistic styles."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        input_images = [
            "https://im.runware.ai/image/ws/0.5/ii/74723926-22f6-417c-befb-f2058fc88c13.webp",
            "https://im.runware.ai/image/ws/0.5/ii/64acee31-100d-4aa1-a47e-6f8b432e7188.webp",
        ]

        # Test different artistic styles
        styles = [
            (
                "Cinematic",
                "img of a person in a dramatic movie scene with cinematic lighting",
            ),
            (
                "Digital Art",
                "img of a person as a digital art character with vibrant colors",
            ),
            (
                "Fantasy art",
                "img of a person as a fantasy character in a magical realm",
            ),
            ("Comic book", "img of a person in comic book art style with bold lines"),
        ]

        for style_name, prompt in styles:
            print(f"Generating {style_name} style...")

            request = IPhotoMaker(
                model="civitai:139562@344487",
                positivePrompt=prompt,
                steps=30,
                numberResults=1,
                height=1024,
                width=768,
                style=style_name,
                strength=50,
                outputFormat="PNG",
                inputImages=input_images,
            )

            photos: List[IImage] = await runware.photoMaker(requestPhotoMaker=request)

            for photo in photos:
                print(f"  {style_name} result: {photo.imageURL}")

    except RunwareError as e:
        print(f"Error in styled photo generation: {e}")
    finally:
        await runware.disconnect()


async def professional_portraits():
    """Generate professional portrait photos with various settings."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        input_images = [
            "https://im.runware.ai/image/ws/0.5/ii/74723926-22f6-417c-befb-f2058fc88c13.webp",
            "https://im.runware.ai/image/ws/0.5/ii/64acee31-100d-4aa1-a47e-6f8b432e7188.webp",
            "https://im.runware.ai/image/ws/0.5/ii/1b39b0e0-6bf7-4c9a-8134-c0251b5ede01.webp",
        ]

        # Professional portrait scenarios
        scenarios = [
            {
                "name": "Business Portrait",
                "prompt": (
                    "img of a professional person in business attire, office background, confident expression"
                ),
                "strength": 30,
            },
            {
                "name": "Casual Portrait",
                "prompt": (
                    "img of a person in casual clothing, natural outdoor setting, relaxed smile"
                ),
                "strength": 35,
            },
            {
                "name": "Artistic Portrait",
                "prompt": (
                    "img of a person with dramatic lighting, artistic composition, professional photography"
                ),
                "strength": 45,
            },
        ]

        for scenario in scenarios:
            print(f"Creating {scenario['name']}...")

            request = IPhotoMaker(
                model="civitai:139562@344487",
                positivePrompt=scenario["prompt"],
                steps=40,  # Higher steps for quality
                numberResults=1,
                height=1024,
                width=768,
                style="Photographic",  # Realistic style
                strength=scenario["strength"],
                outputFormat="PNG",
                includeCost=True,
                inputImages=input_images,
            )

            photos: List[IImage] = await runware.photoMaker(requestPhotoMaker=request)

            for photo in photos:
                print(f"  {scenario['name']}: {photo.imageURL}")
                if photo.cost:
                    print(f"    Processing cost: ${photo.cost}")

    except RunwareError as e:
        print(f"Error in professional portraits: {e}")
    finally:
        await runware.disconnect()


async def creative_photo_scenarios():
    """Generate creative photo scenarios with thematic elements."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        input_images = [
            "https://im.runware.ai/image/ws/0.5/ii/74723926-22f6-417c-befb-f2058fc88c13.webp",
            "https://im.runware.ai/image/ws/0.5/ii/64acee31-100d-4aa1-a47e-6f8b432e7188.webp",
        ]

        # Creative scenarios with specific themes
        creative_themes = [
            {
                "theme": "Vintage Portrait",
                "prompt": (
                    "img of a person in vintage 1920s clothing, sepia tones, classic photography style"
                ),
                "style": "Enhance",
                "dimensions": (768, 1024),
            },
            {
                "theme": "Superhero Style",
                "prompt": (
                    "img of a person as a superhero character, dynamic pose, comic book style"
                ),
                "style": "Comic book",
                "dimensions": (512, 768),
            },
            {
                "theme": "Fantasy Character",
                "prompt": (
                    "img of a person as an elegant elf character in a mystical forest"
                ),
                "style": "Fantasy art",
                "dimensions": (768, 1024),
            },
            {
                "theme": "Futuristic Portrait",
                "prompt": (
                    "img of a person in futuristic sci-fi setting with neon lighting"
                ),
                "style": "Digital Art",
                "dimensions": (1024, 768),
            },
        ]

        for theme_config in creative_themes:
            print(f"Creating {theme_config['theme']}...")

            width, height = theme_config["dimensions"]

            request = IPhotoMaker(
                model="civitai:139562@344487",
                positivePrompt=theme_config["prompt"],
                steps=35,
                numberResults=1,
                height=height,
                width=width,
                style=theme_config["style"],
                strength=50,  # Higher strength for creative themes
                outputFormat="WEBP",
                inputImages=input_images,
            )

            photos: List[IImage] = await runware.photoMaker(requestPhotoMaker=request)

            for photo in photos:
                print(f"  {theme_config['theme']}: {photo.imageURL}")
                print(f"    Style: {theme_config['style']}")
                print(f"    Dimensions: {width}x{height}")

    except RunwareError as e:
        print(f"Error in creative scenarios: {e}")
    finally:
        await runware.disconnect()


async def upload_and_generate():
    """Upload custom images and use them with PhotoMaker."""

    runware = Runware(api_key=os.getenv("RUNWARE_API_KEY"))

    try:
        await runware.connect()

        # Example of uploading images (URLs in this case, but could be local files)
        image_urls = [
            "https://im.runware.ai/image/ws/0.5/ii/74723926-22f6-417c-befb-f2058fc88c13.webp",
            "https://im.runware.ai/image/ws/0.5/ii/64acee31-100d-4aa1-a47e-6f8b432e7188.webp",
        ]

        # Upload images and get UUIDs
        uploaded_images = []
        for i, url in enumerate(image_urls):
            print(f"Uploading image {i + 1}...")
            uploaded: UploadImageType = await runware.uploadImage(url)
            if uploaded and uploaded.imageUUID:
                uploaded_images.append(uploaded.imageUUID)
                print(f"  Uploaded successfully: {uploaded.imageUUID}")
            else:
                print(f"  Failed to upload image {i + 1}")

        if len(uploaded_images) >= 2:
            # Use uploaded images for PhotoMaker
            request = IPhotoMaker(
                model="civitai:139562@344487",
                positivePrompt="img of a person in a beautiful garden setting, golden hour lighting, professional photography",
                steps=32,
                numberResults=1,
                height=1024,
                width=768,
                style="Photographic",
                strength=40,
                outputFormat="PNG",
                includeCost=True,
                inputImages=uploaded_images,  # Use uploaded UUIDs
            )

            photos: List[IImage] = await runware.photoMaker(requestPhotoMaker=request)

            print("PhotoMaker with uploaded images:")
            for photo in photos:
                print(f"  Generated photo: {photo.imageURL}")
                if photo.cost:
                    print(f"  Cost: ${photo.cost}")
        else:
            print("Not enough images uploaded successfully")

    except RunwareError as e:
        print(f"Error with upload and generate: {e}")
    finally:
        await runware.disconnect()


async def main():
    """Run all PhotoMaker examples."""
    print("=== Basic PhotoMaker ===")
    await basic_photo_maker()

    print("\n=== Styled Photo Generation ===")
    await styled_photo_generation()

    print("\n=== Professional Portraits ===")
    await professional_portraits()

    print("\n=== Creative Photo Scenarios ===")
    await creative_photo_scenarios()

    print("\n=== Upload and Generate ===")
    await upload_and_generate()


if __name__ == "__main__":
    asyncio.run(main())
