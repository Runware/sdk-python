# Image Upscaling

Enhance the resolution and quality of your images using Runware's advanced upscaling API. Transform low-resolution images into sharp, high-definition visuals.
Upscaling refers to the process of enhancing the resolution and overall quality of images. This technique is particularly useful for improving the visual clarity and detail of lower-resolution images, making them suitable for various high-definition applications.

## Request

To upscale an image, send a request in the following format:

```json
[
  {
    "taskType": "imageUpscale",
    "taskUUID": "19abad0d-6ec5-40a6-b7af-203775fa5b7f",
    "inputImage": "fd613011-3872-4f37-b4aa-0d343c051a27",
    "outputType": "URL",
    "outputFormat": "JPG",
    "upscaleFactor": 2
  }
]
```

### Parameters

| Parameter     | Type          | Description                                                                                                   |
|---------------|---------------|---------------------------------------------------------------------------------------------------------------|
| taskType      | string        | Must be set to "imageUpscale" for this operation.                                                             |
| taskUUID      | UUIDv4 string | Unique identifier for the task, used to match async responses.                                                |
| inputImage    | UUIDv4 string | The UUID of the image to be upscaled. Can be from a previously uploaded or generated image.                   |
| upscaleFactor | integer       | The level of upscaling to be performed. Can be 2, 3, or 4. Each will increase the image size by that factor.  |
| outputFormat  | string        | Specifies the format of the output image. Supported formats are: PNG, JPG and WEBP.                           |
| includeCost   | boolean       | Optional. If set to true, the response will include the cost of the operation.                                |

## Response

Responses will be delivered in the following format:

```json
{
  "data": [
    {
      "taskType": "imageUpscale",
      "taskUUID": "19abad0d-6ec5-40a6-b7af-203775fa5b7f",
      "imageUUID": "e0b6ed2b-311d-4abc-aa01-8f3fdbdb8860",
      "inputImageUUID": "fd613011-3872-4f37-b4aa-0d343c051a27",
      "imageURL": "https://im.runware.ai/image/ws/0.5/ii/e0b6ed2b-311d-4abc-aa01-8f3fdbdb8860.jpg",
      "cost": 0
    }
  ]
}
```

### Response Parameters

| Parameter    | Type          | Description                                                                                    |
|--------------|---------------|------------------------------------------------------------------------------------------------|
| taskType     | string        | The type of task, in this case "imageUpscale".                                                 |
| taskUUID     | UUIDv4 string | The unique identifier matching the original request.                                           |
| imageUUID    | UUIDv4 string | The UUID of the upscaled image.                                                                |
| imageURL     | string        | The URL where the upscaled image can be downloaded from.                                       |
| cost         | number        | The cost of the operation (included if `includeCost` was set to true).                         |

Note: The NSFW filter occasionally returns false positives and very rarely false negatives.