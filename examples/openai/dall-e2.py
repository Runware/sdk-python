import asyncio
import os
from dotenv import load_dotenv
from runware import Runware, IImageInference, IOpenAIProviderSettings, RunwareAPIError

load_dotenv()

RUNWARE_API_KEY = os.environ.get("RUNWARE_API_KEY")


async def main() -> None:
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()

    # DALL-E 2 with standard settings
    provider_settings = IOpenAIProviderSettings(
        quality="high"
    )

    request_image = IImageInference(
        positivePrompt="A serene landscape with mountains and a crystal clear lake at sunset",
        model="openai:1@1",
        width=1024,
        height=1024,
        numberResults=1,
        outputFormat="PNG",
        outputQuality=95,
        includeCost=True,
        providerSettings=provider_settings
    )

    try:
        images = await runware.imageInference(requestImage=request_image)
        for image in images:
            print(f"Image URL: {image.imageURL}")
            if image.cost:
                print(f"Cost: ${image.cost}")
    except RunwareAPIError as e:
        print(f"API Error: {e}")
        print(f"Error Code: {e.code}")
    except Exception as e:
        print(f"Unexpected Error: {e}")

    # DALL-E 2 with transparent background
    provider_settings_transparent = IOpenAIProviderSettings(
        quality="high",
        background="transparent"
    )

    request_image_transparent = IImageInference(
        positivePrompt="A cute cartoon robot character with big eyes, isolated on transparent background",
        model="openai:1@1",
        width=1024,
        height=1024,
        numberResults=1,
        outputFormat="PNG",
        outputQuality=95,
        includeCost=True,
        providerSettings=provider_settings_transparent
    )

    try:
        images = await runware.imageInference(requestImage=request_image_transparent)
        for image in images:
            print(f"Image URL (transparent): {image.imageURL}")
            if image.cost:
                print(f"Cost: ${image.cost}")
    except RunwareAPIError as e:
        print(f"API Error: {e}")
        print(f"Error Code: {e.code}")
    except Exception as e:
        print(f"Unexpected Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())