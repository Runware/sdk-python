import asyncio
import os

from click import style
from dotenv import load_dotenv
from runware import Runware, IImageInference, IOpenAIProviderSettings, RunwareAPIError

load_dotenv()

RUNWARE_API_KEY = os.environ.get("RUNWARE_API_KEY")


async def main() -> None:
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()

    # DALL-E 3 with HD quality
    provider_settings_hd = IOpenAIProviderSettings(
        quality="hd"
    )

    request_image_hd = IImageInference(
        positivePrompt="A futuristic city with flying cars and neon lights, cyberpunk aesthetic, highly detailed",
        model="openai:2@3",
        width=1024,
        height=1024,
        numberResults=1,
        outputFormat="PNG",
        outputQuality=95,
        includeCost=True,
        providerSettings=provider_settings_hd
    )

    try:
        images = await runware.imageInference(requestImage=request_image_hd)
        for image in images:
            print(f"Image URL (HD): {image.imageURL}")
            if image.cost:
                print(f"Cost: ${image.cost}")
    except RunwareAPIError as e:
        print(f"API Error: {e}")
        print(f"Error Code: {e.code}")
    except Exception as e:
        print(f"Unexpected Error: {e}")

    # DALL-E 3 with standard quality
    provider_settings_standard = IOpenAIProviderSettings(
        quality="standard",
        style="ghibli style"
    )

    request_image_standard = IImageInference(
        positivePrompt="A whimsical tea party in a magical forest with talking animals",
        model="openai:2@3",
        width=1024,
        height=1024,
        numberResults=1,
        outputFormat="PNG",
        outputQuality=95,
        includeCost=True,
        providerSettings=provider_settings_standard
    )

    try:
        images = await runware.imageInference(requestImage=request_image_standard)
        for image in images:
            print(f"Image URL (Standard): {image.imageURL}")
            if image.cost:
                print(f"Cost: ${image.cost}")
    except RunwareAPIError as e:
        print(f"API Error: {e}")
        print(f"Error Code: {e.code}")
    except Exception as e:
        print(f"Unexpected Error: {e}")


if __name__ == "__main__":
    asyncio.run(main())