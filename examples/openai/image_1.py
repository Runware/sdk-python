import asyncio
import os

from dotenv import load_dotenv
from runware import Runware, IImageInference, IOpenAIProviderSettings, RunwareAPIError

load_dotenv()

RUNWARE_API_KEY = os.environ.get("RUNWARE_API_KEY")


async def main() -> None:
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()

    provider_settings_hd = IOpenAIProviderSettings(
        quality="high",
        style="cyberpunk"
    )

    request_image_hd = IImageInference(
        positivePrompt="A futuristic city with flying cars and neon lights, cyberpunk aesthetic, highly detailed",
        model="openai:1@1",
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

if __name__ == "__main__":
    asyncio.run(main())