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

#### Enabling teaCache/deepCache for faster inference

Flux and SDXL models support teaCache and deepCache for faster inference, with the trade-off of quality loss with more aggressive settings.

```python
from runware import Runware, IImageInference, IAcceleratorOptions

async def main() -> None:
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()

    request_image = IImageInference(
        positivePrompt="a beautiful sunset over the mountains",
        model="civitai:943001@1055701", # using Shuttle v3 for this test, to showcase the power on 3rd party Flux finetunes.
        numberResults=1,
        negativePrompt="cloudy, rainy",
        height=1024,
        width=1024,
        acceleratorOptions=IAcceleratorOptions(
            teaCache=True,
            teaCacheDistance=0.6, # 0.6 is at the more moderate-to-extreme end, and 0.1 is at the more conservative end.
        ),
    )

    images = await runware.imageInference(requestImage=request_image)
    for image in images:
        print(f"Image URL: {image.imageURL}")
```

##### teaCache

- `teaCache` is a boolean that enables or disables the teaCache feature. If set to `True`, it will use teaCache for faster inference.
  - It is specific to transformer models, Flux and SD3. `teaCache` does not work for UNet models like SDXL or SD1x.
- `teaCacheDistance` is a float between 0.0 and 1.0, where 0.0 is the most conservative and 1.0 is the most aggressive.
- `cacheStartStep` and `cacheStopStep` are integers that represent the start and end steps of the teaCache and DeepCache process.
  - `cacheStartStep` is the step at which the generator starts to skip blocks and reduce quality; `cacheStopStep` is the step at which the teaCache process ends, returning to full fidelity steps.
  - If not specified, teaCache (or DeepCache) will be enabled throughout the entire image generation process, which may be undesirable for preserving quality.

##### deepCache

- `deepCache` is a boolean that enables or disables the deepCache feature. If set to `True`, it will use deepCache for faster inference.
- `deepCacheInterval` represents the frequency of feature caching, specified as the number of steps between each cache operation.
  - A larger cache interval makes inference faster, and costs more quality.
  - The default value is `3`
- `deepCacheBranchId` represents which branch of the network (ordered from the shallowest to the deepest layer) is responsible for executing the caching processes.
  - Opting for a lower branch ID will result in a more aggressive caching process, while a higher branch ID will yield a more conservative approach.
  - The default value is `0`

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
    remove_image_background_payload = IImageBackgroundRemoval(inputImage=image_path)

    processed_images = await runware.imageBackgroundRemoval(
        removeImageBackgroundPayload=remove_image_background_payload
    )
    for image in processed_images:
        print(image.imageURL)
```

### Image-to-Text Conversion

To convert an image to text using the Runware API, you can use the `imageCaption` method of the `Runware` class. Here's an example:

```python
from runware import Runware, IImageCaption

async def main() -> None:
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()

    image_path = "image.jpg"
    request_image_to_text_payload = IImageCaption(inputImage=image_path)

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
        print(image.imageURL)
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
        model="civitai:139562@344487",
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

### ACE++

ACE++ (**A**dvanced **C**haracter **E**dit) is an advanced framework for character-consistent image generation and editing. It allows you to create new images from a single reference image while preserving identity, and edit existing images without retraining the model.

**Note:** When using ACE++, you must set the model parameter to `runware:102@1`.

#### Character-Consistent Generation

To generate new images while preserving character identity from a reference image:

```python
from runware import Runware, IImageInference, IAcePlusPlus

async def main() -> None:
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()

    # Upload your reference image first
    reference_image = await runware.uploadImage("path/to/reference_image.jpg")

    request_image = IImageInference(
        positivePrompt="photo of man wearing a business suit in a modern office",
        model="runware:102@1",                        # Required model for ACE++
        height=1024,
        width=1024,
        numberResults=1,
        acePlusPlus=IAcePlusPlus(
            inputImages=[reference_image.imageUUID],  # Reference image for character identity
            repaintingScale=0.3                       # Lower values (0.0-0.5) preserve more identity
        )
    )

    images = await runware.imageInference(requestImage=request_image)
    for image in images:
        print(f"Image URL: {image.imageURL}")
```

#### Character-Consistent Editing

To edit existing images while preserving character identity using masks:

```python
from runware import Runware, IImageInference, IAcePlusPlus

async def main() -> None:
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()

    # Upload your reference image and mask
    reference_image = await runware.uploadImage("path/to/reference_image.jpg")
    mask_image = await runware.uploadImage("path/to/mask_image.png")

    request_image = IImageInference(
        positivePrompt="photo of woman wearing a red dress",
        model="runware:102@1",  # Required model for ACE++
        height=1024,
        width=1024,
        numberResults=1,
        acePlusPlus=IAcePlusPlus(
            inputImages=[reference_image.imageUUID],  # Reference image
            inputMasks=[mask_image.imageUUID],  # Mask for selective editing
            repaintingScale=0.7  # Higher values (0.5-1.0) follow prompt more in edited areas
        )
    )

    images = await runware.imageInference(requestImage=request_image)
    for image in images:
        print(f"Image URL: {image.imageURL}")
```

**ACE++ Parameters:**
- `inputImages`: Array containing exactly one reference image (required)
- `inputMasks`: Array containing at most one mask image (optional, for editing)
- `repaintingScale`: Float between 0.0 and 1.0
  - 0.0: Maximum character identity preservation
  - 1.0: Maximum adherence to prompt instructions
  - For generation: Use 0.0-0.5 for strong resemblance
  - For editing: Use 0.5-1.0 for more creative freedom in edited areas

**Mask Requirements:**
- The mask should be a black and white image
- White (255) represents areas to be edited
- Black (0) represents areas to be preserved
- Supported formats: PNG, JPG, WEBP

### Generating Images with refiner

To generate images using the Runware API with refiner support, you can use the `imageInference` method of the `Runware` class. Here's an example:

```python
from runware import Runware, IImageInference, IRefiner

async def main() -> None:
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()
    
    refiner = IRefiner(
        model="civitai:101055@128080",
        startStep=2,
        startStepPercentage=None,
    )

    request_image = IImageInference(
        positivePrompt="a beautiful sunset over the mountains",
        model="civitai:101055@128078",
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


### Using ControlNet with Image Inference

To use ControlNet for image inference in the Runware SDK, you can use a class `IControlNetGeneral`. Here's an example of how to set up and use this feature:

```python
from runware import Runware, IImageInference, IControlNetGeneral, EControlMode

async def main() -> None:
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()

    controlNet = IControlNetGeneral(
        startStep=1,
        endStep=30,
        weight=0.5,
        controlMode=EControlMode.BALANCED.value,
        guideImage="https://huggingface.co/datasets/mishig/sample_images/resolve/main/canny-edge.jpg",
        model='civitai:38784@44716'
    )

    request_image = IImageInference(
        positivePrompt="a beautiful sunset",
        model='civitai:4384@128713',
        controlNet=[controlNet],
        numberResults=1,
        height=512,
        width=512,
        outputType="URL",
        seed=1568,
        steps=40
    )

    images = await runware.imageInference(requestImage=request_image)

    for image in images:
        print(f"Image URL: {image.imageURL}")

```
This example demonstrates how to configure and use a ControlNet to enhance the image inference process.


### Inferencing Ace++ Pipeline

To use Ace++ in the Runware SDK, you can use a class `IAcePlusPlus`. Here's an example of how to set up and use this feature:
Much more examples are in examples/ace++

```python
from runware import Runware, IImageInference, IAcePlusPlus

async def main() -> None:
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()

    # Upload your reference image and mask
    reference_image = "https://raw.githubusercontent.com/ali-vilab/ACE_plus/refs/heads/main/assets/samples/application/logo_paste/1_ref.png"
    mask_image = "https://raw.githubusercontent.com/ali-vilab/ACE_plus/refs/heads/main/assets/samples/application/logo_paste/1_1_m.png"
    init_image = "https://raw.githubusercontent.com/ali-vilab/ACE_plus/refs/heads/main/assets/samples/application/logo_paste/1_1_edit.png"
    request_image = IImageInference(
        positivePrompt="The logo is printed on the headphones.",
        model="runware:102@1",  # Required model for ACE++
        taskUUID="68020b8f-bbcf-4779-ba51-4f3bb00aef6a",
        height=1024,
        width=1024,
        numberResults=1,
        steps=28,
        CFGScale=50.0,
        referenceImages=[reference_image],  # Reference image
        acePlusPlus=IAcePlusPlus(
            inputImages=[init_image],  # Input image
            inputMasks=[mask_image],  # Mask for selective editing
            repaintingScale=1.0,
            taskType="subject"  # Can be one of "portrait", "subject", "local_editing"
        ),
    )
    print(f"Sending request: {request_image}")
    images = await runware.imageInference(requestImage=request_image)
    
    for image in images:
        print(f"Image URL: {image.imageURL}")

```
This example demonstrates how to configure and use a ControlNet to enhance the image inference process.


### Inferencing Video Models

To inference Video Generation Models in the Runware SDK, you can use a class `IVideoInference`. Almost every video model support its own providerSettings: `IMinimaxProviderSettings`, `IBytedanceProviderSettings`, `IGoogleProviderSettings`, `IKlingAIProviderSettings`, `IPixverseProviderSettings`, `IViduProviderSettings`. More examples can be found in examples/video.

Here's an example of an image-to-video (i2v) task using Google's Veo3:  

```python
import asyncio
from runware import Runware, IVideoInference, IGoogleProviderSettings, IFrameImage


async def main():
    runware = Runware(
        api_key=RUNWARE_API_KEY,
    )
    await runware.connect()

    request = IVideoInference(
        positivePrompt="spinning galaxy",
        model="google:3@0",
        width=1280,
        height=720,
        numberResults=1,
        seed=10,
        includeCost=True,
        frameImages=[ # Comment this to use t2v
            IFrameImage(
                inputImage="https://github.com/adilentiq/test-images/blob/main/common/image_15_mb.jpg?raw=true",
            ),
        ],
        providerSettings=IGoogleProviderSettings(
            generateAudio=True,
            enhancePrompt=True
        )
    )
    videos = await runware.videoInference(requestVideo=request)
    for video in videos:
        print(f"Video URL: {video.videoURL}")
        print(f"Cost: {video.cost}")
        print(f"Seed: {video.seed}")
        print(f"Status: {video.status}")


if __name__ == "__main__":
    asyncio.run(main())
```


### Model Upload

To upload model using the Runware API, you can use the `uploadModel` method of the `Runware` class. Here are examples:

```python
from runware import Runware, IUploadModelCheckPoint


async def main() -> None:
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()

    payload = IUploadModelCheckPoint(
        air='qatests:68487@08629',
        name='yWO8IaKwez',
        heroImageURL='https://raw.githubusercontent.com/adilentiq/test-images/refs/heads/main/image.jpg',
        downloadURL='https://repo-controlnets-r2.runware.ai/controlnet-zoe-depth-sdxl-1.0.safetensors'
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
from runware import Runware, IUploadModelLora


async def main() -> None:
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()

    payload = IUploadModelLora(
        air='qatests:68487@08629',
        name='yWO8IaKwez',
        heroImageURL='https://raw.githubusercontent.com/adilentiq/test-images/refs/heads/main/image.jpg',
        downloadURL='https://repo-controlnets-r2.runware.ai/controlnet-zoe-depth-sdxl-1.0.safetensors'
                    '/controlnet-zoe-depth-sdxl-1.0.safetensors.part-001-1',
        uniqueIdentifier='aq2w3e4r5t6y7u8i9o0p1q2w3e4r5t6y7u8i9o0p1q2w3e4r5t6y7u8i9o0p1234',
        version='1.0',
        tags=['tag1', 'tag2', 'tag2'],
        architecture='flux1d',
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
from runware import Runware, IUploadModelControlNet


async def main() -> None:
    runware = Runware(api_key=RUNWARE_API_KEY)
    await runware.connect()

    payload = IUploadModelControlNet(
        air='qatests:68487@08629',
        name='yWO8IaKwez',
        heroImageURL='https://raw.githubusercontent.com/adilentiq/test-images/refs/heads/main/image.jpg',
        downloadURL='https://repo-controlnets-r2.runware.ai/controlnet-zoe-depth-sdxl-1.0.safetensors'
                    '/controlnet-zoe-depth-sdxl-1.0.safetensors.part-001-1',
        uniqueIdentifier='aq2w3e4r5t6y7u8i9o0p1q2w3e4r5t6y7u8i9o0p1q2w3e4r5t6y7u8i9o0p1234',
        version='1.0',
        tags=['tag1', 'tag2', 'tag2'],
        architecture='flux1d',
        format='safetensors',
        shortDescription='a model description',
        private=False,
        comment='some comments if you want to add for internal use',
    )


uploaded = await runware.modelUpload(payload)
print(f"Response : {uploaded}")
```
### Image Background Removal
There are two ways to remove the background from an image.
1. Using the `settings` parameter of the `IImageBackgroundRemoval` class.
2. Without using the `settings` parameter and using the `model` parameter to specify the model to use.

### Using the `settings` parameter

> **Note:** When using the `rgba` parameter, the final `a` value is a `float` between `0.0` and `1.0`, but a value of `1-255` will be internally scaled down to the correct float range.

```python
from runware import Runware, RunwareAPIError, IImage, IImageBackgroundRemoval, IBackgroundRemovalSettings
import asyncio
import os
from dotenv import load_dotenv

load_dotenv(override=True)


async def main() -> None:
    runware = Runware(
        api_key=os.environ.get("RUNWARE_API_KEY")
    )
    await runware.connect()
    background_removal_settings = IBackgroundRemovalSettings(
        rgba=[255, 255, 255, 0],
        alphaMatting=True,
        postProcessMask=True,
        returnOnlyMask=False,
        alphaMattingErodeSize=10,
        alphaMattingForegroundThreshold=240,
        alphaMattingBackgroundThreshold=10
        )

    request_image = IImageBackgroundRemoval(
        taskUUID="abcdbb9c-3bd3-4d75-9234-bffeef994772",
        inputImage="https://raw.githubusercontent.com/adilentiq/test-images/refs/heads/main/common/headphones.jpeg",
        settings=background_removal_settings,
        outputType="URL",
        outputFormat="PNG",
        includeCost=True,
    )

    print(f"Payload: {request_image}")
    try:
        processed_images: List[IImage] = await runware.imageBackgroundRemoval(
            removeImageBackgroundPayload=request_image
        )
    except RunwareAPIError as e:
        print(f"API Error: {e}")
        print(f"Error Code: {e.code}")
    except Exception as e:
        print(f"Unexpected Error: {e}")
    else:
        print("Processed Image with the background removed:")
        print(processed_images)
        for image in processed_images:
            print(image.imageURL)


asyncio.run(main())
```

### Using the `model` parameter

```python

from runware import Runware, RunwareAPIError, IImage, IImageBackgroundRemoval
import asyncio
import os
from dotenv import load_dotenv

load_dotenv(override=True)


async def main() -> None:
    runware = Runware(
        api_key=os.environ.get("RUNWARE_API_KEY"),
    )
    await runware.connect()

    request_image = IImageBackgroundRemoval(
        taskUUID="abcdbb9c-3bd3-4d75-9234-bffeef994772",
        model="runware:110@1",
        inputImage="https://raw.githubusercontent.com/adilentiq/test-images/refs/heads/main/common/headphones.jpeg"
    )

    print(f"Payload: {request_image}")
    try:
        processed_images: List[IImage] = await runware.imageBackgroundRemoval(
            removeImageBackgroundPayload=request_image
        )
    except RunwareAPIError as e:
        print(f"API Error: {e}")
        print(f"Error Code: {e.code}")
    except Exception as e:
        print(f"Unexpected Error: {e}")
    else:
        print("Processed Image with the background removed:")
        print(processed_images)
        for image in processed_images:
            print(image.imageURL)


asyncio.run(main())
```

For more detailed usage and additional examples, please refer to the examples directory.
