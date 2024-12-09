# Python Runware SDK

The Python Runware SDK is used to run image inference with the Runware API, powered by the Runware inference platform. It can be used to generate images with text-to-image and image-to-image. It also allows the use of an existing gallery of models or selecting any model or LoRA from the CivitAI gallery. The API also supports upscaling, background removal, inpainting and outpainting, and a series of other ControlNet models.

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

To generate images using the Runware API, you can use the `imageInference` method of the `Runware` class. Here's an example:

```python
from runware import Runware, IImageInference

async def main() -> None:
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()

    request_image = IImageInference(
        positivePrompt="a beautiful sunset over the mountains",
        model="civitai:36520@76907",  
        numberResults=4,  
        negativePrompt="cloudy, rainy",
        height=512,  
        width=512, 
    )

    images = await runware.imageInference(requestImage=request_image)
    for image in images:
        print(f"Image URL: {image.imageURL}")
```

### Enhancing Prompts

To enhance prompts using the Runware API, you can use the `promptEnhance` method of the `Runware` class. Here's an example:

```python
from runware import Runware, IPromptEnhance

async def main() -> None:
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()

    prompt = "A beautiful sunset over the mountains"
    prompt_enhancer = IPromptEnhance(
        prompt=prompt,
        promptVersions=3,
        promptMaxLength=64,
    )

    enhanced_prompts = await runware.promptEnhance(promptEnhancer=prompt_enhancer)
    for enhanced_prompt in enhanced_prompts:
        print(enhanced_prompt.text)
```

### Removing Image Background

To remove the background from an image using the Runware API, you can use the `imageBackgroundRemoval` method of the `Runware` class. Here's an example:

```python
from runware import Runware, IImageBackgroundRemoval

async def main() -> None:
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()

    image_path = "image.jpg"
    remove_image_background_payload = IImageBackgroundRemoval(image_initiator=image_path)

    processed_images = await runware.imageBackgroundRemoval(
        removeImageBackgroundPayload=remove_image_background_payload
    )
    for image in processed_images:
        print(image.imageURL)
```

### Image-to-Text Conversion

To convert an image to text using the Runware API, you can use the `imageCaption` method of the `Runware` class. Here's an example:

```python
from runware import Runware, IRequestImageToText

async def main() -> None:
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()

    image_path = "image.jpg"
    request_image_to_text_payload = IImageCaption(image_initiator=image_path)

    image_to_text = await runware.imageCaption(
        requestImageToText=request_image_to_text_payload
    )
    print(image_to_text.text)
```

### Upscaling Images

To upscale an image using the Runware API, you can use the `imageUpscale` method of the `Runware` class. Here's an example:

```python
from runware import Runware, IImageUpscale

async def main() -> None:
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()

    image_path = "image.jpg"
    upscale_factor = 4

    upscale_gan_payload = IImageUpscale(
        inputImage=image_path, upscaleFactor=upscale_factor
    )
    upscaled_images = await runware.imageUpscale(upscaleGanPayload=upscale_gan_payload)
    for image in upscaled_images:
        print(image.imageSrc)
```

### Photo Maker

Use the `photoMaker` method of the `Runware` class. Here's an example:

```python
from runware import Runware, IPhotoMaker
import uuid

async def main() -> None:
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()

    request_image = IPhotoMaker(
        model="civitai:101055@128080",
        positivePrompt="img of a beautiful lady in a forest",
        steps=35,
        numberResults=1,
        height=512,
        width=512,
        style="No style",
        strength=40,
        outputFormat="WEBP",
        includeCost=True,
        taskUUID=str(uuid.uuid4()),
        inputImages=[
            "https://im.runware.ai/image/ws/0.5/ii/74723926-22f6-417c-befb-f2058fc88c13.webp",
            "https://im.runware.ai/image/ws/0.5/ii/64acee31-100d-4aa1-a47e-6f8b432e7188.webp",
            "https://im.runware.ai/image/ws/0.5/ii/1b39b0e0-6bf7-4c9a-8134-c0251b5ede01.webp",
            "https://im.runware.ai/image/ws/0.5/ii/f4b4cec3-66d9-4c02-97c5-506b8813182a.webp"
        ],
    )
    
    
     photos = await runware.photoMaker(requestPhotoMaker=request_image)
     for photo in photos:
         print(f"Image URL: {photo.imageURL}")
```

### Generating Images with refiner

To generate images using the Runware API with refiner support, you can use the `imageInference` method of the `Runware` class. Here's an example:

```python
from runware import Runware, IImageInference, IRefiner

async def main() -> None:
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()
    
    refiner = IRefiner(
        model="civitai:101055@128080",
        startStep=a,
        startStepPercentage=None,
    )

    request_image = IImageInference(
        positivePrompt="a beautiful sunset over the mountains",
        model="civitai:36520@76907",  
        numberResults=4,  
        negativePrompt="cloudy, rainy",
        height=512,  
        width=512, 
        refiner=refiner
    )

    images = await runware.imageInference(requestImage=request_image)
    for image in images:
        print(f"Image URL: {image.imageURL}")
```

### Model Upload

To upload model using the Runware API, you can use the `uploadModel` method of the `Runware` class. Here are examples:

```python
from runware import Runware, IImageInference, IRefiner, IUploadModelCheckPoint


async def main() -> None:
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()

    payload = IUploadModelCheckPoint(
        air='qatests:68487@08629',
        name='yWO8IaKwez',
        heroImageUrl='https://raw.githubusercontent.com/adilentiq/test-images/refs/heads/main/image.jpg',
        downloadUrl='https://repo-controlnets-r2.runware.ai/controlnet-zoe-depth-sdxl-1.0.safetensors'
                    '/controlnet-zoe-depth-sdxl-1.0.safetensors.part-001-1',
        uniqueIdentifier='aq2w3e4r5t6y7u8i9o0p1q2w3e4r5t6y7u8i9o0p1q2w3e4r5t6y7u8i9o0p1234',
        version='1.0',
        tags=['tag1', 'tag2', 'tag2'],
        architecture='flux1d',
        type='base',
        defaultWeight=0.8,
        format='safetensors',
        positiveTriggerWords='my trigger word',
        shortDescription='a model description',
        private=False,
        defaultScheduler='Default',
        comment='some comments if you want to add for internal use',
    )

    uploaded = await runware.modelUpload(payload)
    print(f"Response : {uploaded}")
```

```python
from runware import Runware, IImageInference, IRefiner, IUploadModelLora


async def main() -> None:
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()

    payload = IUploadModelLora(
        air='qatests:68487@08629',
        name='yWO8IaKwez',
        heroImageUrl='https://raw.githubusercontent.com/adilentiq/test-images/refs/heads/main/image.jpg',
        downloadUrl='https://repo-controlnets-r2.runware.ai/controlnet-zoe-depth-sdxl-1.0.safetensors'
                    '/controlnet-zoe-depth-sdxl-1.0.safetensors.part-001-1',
        uniqueIdentifier='aq2w3e4r5t6y7u8i9o0p1q2w3e4r5t6y7u8i9o0p1q2w3e4r5t6y7u8i9o0p1234',
        version='1.0',
        tags=['tag1', 'tag2', 'tag2'],
        architecture='flux1d',
        type='base',
        defaultWeight=0.8,
        format='safetensors',
        positiveTriggerWords='my trigger word',
        shortDescription='a model description',
        private=False,
        comment='some comments if you want to add for internal use',
    )

    uploaded = await runware.modelUpload(payload)
    print(f"Response : {uploaded}")
```

```python
from runware import Runware, IImageInference, IRefiner, IUploadModelControlNet


async def main() -> None:
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()

    payload = IUploadModelControlNet(
        air='qatests:68487@08629',
        name='yWO8IaKwez',
        heroImageUrl='https://raw.githubusercontent.com/adilentiq/test-images/refs/heads/main/image.jpg',
        downloadUrl='https://repo-controlnets-r2.runware.ai/controlnet-zoe-depth-sdxl-1.0.safetensors'
                    '/controlnet-zoe-depth-sdxl-1.0.safetensors.part-001-1',
        uniqueIdentifier='aq2w3e4r5t6y7u8i9o0p1q2w3e4r5t6y7u8i9o0p1q2w3e4r5t6y7u8i9o0p1234',
        version='1.0',
        tags=['tag1', 'tag2', 'tag2'],
        architecture='flux1d',
        type='base',
        format='safetensors',
        positiveTriggerWords='my trigger word',
        shortDescription='a model description',
        private=False,
        comment='some comments if you want to add for internal use',
    )


uploaded = await runware.modelUpload(payload)
print(f"Response : {uploaded}")
```

For more detailed usage and additional examples, please refer to the examples directory.
