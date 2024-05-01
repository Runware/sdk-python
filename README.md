# Python Runware SDK

The Python Runware SDK is used to run image inference with the Runware API, powered by the RunWare inference platform. It can be used to generate imaged with text-to-image and image-to-image. It also allows the use of an existing gallery of models or selecting any model or LoRA from the CivitAI gallery. The API also supports upscaling, background removal, inpainting and outpainting, and a series of other ControlNet models.

## Get API Access

To use the Python Runware SDK, you need to obtain an API key. Follow these steps to get API access:

1. [Create a free account](https://my.runware.ai/) with [Runware](https://runware.ai/).
2. Once you have created an account, you will receive an API key and trial credits.

**Important**: Please keep your API key private and do not share it with anyone. Treat it as a sensitive credential.

## Documentation

For detailed documentation and API reference, please visit the [Runware Documentation](https://docs.runware.ai/) or refer to the [docs](docs) folder in the repository. The documentation provides comprehensive information about the available classes, methods, and parameters, along with code examples to help you get started with the Runware SDK Python.

## Installation

To install the Python Runware SDK, use the following command:

```bash
pip install runware
```

## Usage

Before using the Python Runware SDK, make sure to set your Runware API key in the environment variable `RUNWARE_API_KEY`. You can do this by creating a `.env` file in your project root and adding the following line:

```bash
RUNWARE_API_KEY = "your_api_key_here"
```

### Generating Images

To generate images using the Runware API, you can use the `requestImages` method of the `Runware` class. Here's an example:

```python
from runware import Runware, IRequestImage

async def main() -> None:
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()

    request_image = IRequestImage(
        positive_prompt="A beautiful sunset over the mountains",
        image_size=1,
        model_id=13,
        number_of_images=5,
        negative_prompt="cloudy, rainy",
        use_cache=False,
    )

    images = await runware.requestImages(requestImage=request_image)
    for image in images:
        print(f"Image URL: {image.imageSrc}")
```

### Enhancing Prompts

To enhance prompts using the Runware API, you can use the `enhancePrompt` method of the `Runware` class. Here's an example:

```python
from runware import Runware, IPromptEnhancer

async def main() -> None:
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()

    prompt = "A beautiful sunset over the mountains"
    prompt_enhancer = IPromptEnhancer(
        prompt=prompt,
        prompt_versions=3,
    )

    enhanced_prompts = await runware.enhancePrompt(promptEnhancer=prompt_enhancer)
    for enhanced_prompt in enhanced_prompts:
        print(enhanced_prompt.text)
```

### Removing Image Background

To remove the background from an image using the Runware API, you can use the `removeImageBackground` method of the `Runware` class. Here's an example:

```python
from runware import Runware, IRemoveImageBackground

async def main() -> None:
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()

    image_path = "image.jpg"
    remove_image_background_payload = IRemoveImageBackground(image_initiator=image_path)

    processed_images = await runware.removeImageBackground(
        removeImageBackgroundPayload=remove_image_background_payload
    )
    for image in processed_images:
        print(image.imageSrc)
```

### Image-to-Text Conversion

To convert an image to text using the Runware API, you can use the `requestImageToText` method of the `Runware` class. Here's an example:

```python
from runware import Runware, IRequestImageToText

async def main() -> None:
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()

    image_path = "image.jpg"
    request_image_to_text_payload = IRequestImageToText(image_initiator=image_path)

    image_to_text = await runware.requestImageToText(
        requestImageToText=request_image_to_text_payload
    )
    print(image_to_text.text)
```

### Upscaling Images

To upscale an image using the Runware API, you can use the `upscaleGan` method of the `Runware` class. Here's an example:

```python
from runware import Runware, IUpscaleGan

async def main() -> None:
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()

    image_path = "image.jpg"
    upscale_factor = 4

    upscale_gan_payload = IUpscaleGan(
        image_initiator=image_path, upscale_factor=upscale_factor
    )
    upscaled_images = await runware.upscaleGan(upscaleGanPayload=upscale_gan_payload)
    for image in upscaled_images:
        print(image.imageSrc)
```

For more detailed usage and additional examples, please refer to the examples directory.
