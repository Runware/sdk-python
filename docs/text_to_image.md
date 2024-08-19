# Image Inference API

Generate images from text prompts or transform existing ones using Runware's API. This powerful feature allows you to create high-quality visuals, bringing creative ideas to life or enhancing existing images with new styles or subjects.

## Introduction

Image inference enables you to:

1. **Text-to-Image**: Generate images from descriptive text prompts.
2. **Image-to-Image**: Transform existing images, controlling the strength of the transformation.
3. **Inpainting**: Replace parts of an image with new content.
4. **Outpainting**: Extend the boundaries of an image with new content.

Advanced features include:

- **ControlNet**: Precise control over image generation using additional input conditions.
- **LoRA**: Adapt models to specific styles or tasks.

Our API is optimized for speed and efficiency, powered by our Sonic Inference Engine.

## Request

Requests are sent as an array of objects, each representing a specific task. Here's the basic structure for an image inference task:

```json
[
  {
    "taskType": "imageInference",
    "taskUUID": "string",
    "outputType": "string",
    "outputFormat": "string",
    "positivePrompt": "string",
    "negativePrompt": "string",
    "height": int,
    "width": int,
    "model": "string",
    "steps": int,
    "CFGScale": float,
    "numberResults": int
  }
]
```

### Parameters

| Parameter        | Type                          | Required | Description                                                                                    |
|------------------|-------------------------------|----------|------------------------------------------------------------------------------------------------|
| taskType         | string                        | Yes      | Must be set to "imageInference" for this task.                                                 |
| taskUUID         | string (UUID v4)              | Yes      | Unique identifier for the task, used to match async responses.                                 |
| outputType       | string                        | No       | Specifies the output format: "base64Data", "dataURI", or "URL" (default).                      |
| outputFormat     | string                        | No       | Specifies the image format: "JPG" (default), "PNG", or "WEBP".                                 |
| positivePrompt   | string                        | Yes      | Text instruction guiding the image generation (4-2000 characters).                             |
| negativePrompt   | string                        | No       | Text instruction to avoid certain elements in the image (4-2000 characters).                   |
| height           | integer                       | Yes      | Height of the generated image (512-2048, must be divisible by 64).                             |
| width            | integer                       | Yes      | Width of the generated image (512-2048, must be divisible by 64).                              |
| model            | string                        | Yes      | AIR identifier of the model to use.                                                            |
| steps            | integer                       | No       | Number of inference steps (1-100, default 20).                                                 |
| CFGScale         | float                         | No       | Guidance scale for prompt adherence (0-30, default 7).                                         |
| numberResults    | integer                       | No       | Number of images to generate (default 1).                                                      |

Additional parameters:

| Parameter           | Type    | Required | Description                                                                     |
|---------------------|---------|----------|---------------------------------------------------------------------------------|
| uploadEndpoint      | string  | No       | URL to upload the generated image using HTTP PUT.                               |
| checkNSFW           | boolean | No       | Enable NSFW content check (adds 0.1s to inference time).                        |
| includeCost         | boolean | No       | Include the cost of the operation in the response.                              |
| seedImage           | string  | No*      | Image to use as a starting point (required for Image-to-Image, In/Outpainting). |
| maskImage           | string  | No*      | Mask image for Inpainting/Outpainting (required for these operations).          |
| strength            | float   | No       | Influence of the seed image (0-1, default 0.8).                                 |
| scheduler           | string  | No       | Specify a different scheduler (default is model's own scheduler).               |
| seed                | integer | No       | Seed for reproducible results (1-9223372036854776000).                          |
| clipSkip            | integer | No       | Number of CLIP layers to skip (0-2, default 0).                                 |
| usePromptWeighting  | boolean | No       | Enable advanced prompt weighting (adds 0.2s to inference time).                 |

### ControlNet

To use ControlNet, include a `controlNet` array in your request with objects containing:

| Parameter | Type    | Required | Description                                               |
|-----------|---------|----------|-----------------------------------------------------------|
| model     | string  | Yes      | AIR identifier of the ControlNet model.                   |
| guideImage| string  | Yes      | Preprocessed guide image (UUID, data URI, base64, or URL).|
| weight    | float   | No       | Weight of this ControlNet model (0-1).                    |
| startStep | integer | No       | Step to start applying ControlNet.                        |
| endStep   | integer | No       | Step to stop applying ControlNet.                         |
| controlMode| string | No       | "prompt", "controlnet", or "balanced".                    |

### LoRA

To use LoRA, include a `lora` array in your request with objects containing:

| Parameter | Type   | Required | Description                            |
|-----------|--------|----------|----------------------------------------|
| model     | string | Yes      | AIR identifier of the LoRA model.      |
| weight    | float  | No       | Weight of this LoRA model (default 1). |

## Response

The API returns results in the following format:

```json
{
  "data": [
    {
      "taskType": "imageInference",
      "taskUUID": "a770f077-f413-47de-9dac-be0b26a35da6",
      "imageUUID": "77da2d99-a6d3-44d9-b8c0-ae9fb06b6200",
      "imageURL": "https://im.runware.ai/image/ws/0.5/ii/a770f077-f413-47de-9dac-be0b26a35da6.jpg",
      "cost": 0.0013
    }
  ]
}
```

### Response Parameters

| Parameter      | Type             | Description                                                         |
|----------------|------------------|---------------------------------------------------------------------|
| taskType       | string           | Type of the task ("imageInference").                                |
| taskUUID       | string (UUID v4) | Unique identifier matching the original request.                    |
| imageUUID      | string (UUID v4) | Unique identifier of the generated image.                           |
| imageURL       | string           | URL to download the image (if outputType is "URL").                 |
| imageBase64Data| string           | Base64-encoded image data (if outputType is "base64Data").          |
| imageDataURI   | string           | Data URI of the image (if outputType is "dataURI").                 |
| NSFWContent    | boolean          | Indicates if the image was flagged as NSFW (if checkNSFW was true). |
| cost           | float            | Cost of the operation in USD (if includeCost was true).             |

Note: The API may return multiple images per message, as they are generated in parallel.
