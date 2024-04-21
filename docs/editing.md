# Image upscaling

## Used to the resolution of generated images or download images at higher resolution

Image upscaling can be performed on previously generated or uploaded images.

To upscale an image, send a request in the following format:

```json
{
    "newUpscaleGan": {
        "imageUUID": "fd613011-3872-4f37-b4aa-0d343c051a27",
        "taskUUID": "19abad0d-6ec5-40a6-b7af-203775fa5b7f",
        "upscaleFactor": 4
    }
}
```

| Parameter     | Type          | Use                                                                                                         |
|---------------|---------------|-------------------------------------------------------------------------------------------------------------|
| imageUUID     | UUIDv4 string | The UUID of the interrogated image. Will be either the UUID of an uploaded image or a generated image.      |
| taskUUID      | UUIDv4 string | Used to identify the async responses to this task. It must be sent to match the response to the task.       |
| upscaleFactor | integer       | Is the level of upscaling performed. Can take values of: 2, 3, 4. Each will increase the size of the image by the corresponding factor. For instance, an upscaleFactor of 2 will 2x image size. |

Responses will be received in the following format:

```json
{
    "newUpscaleGan": {
        "images": [
            {
                "bNSFWContent": false,
                "imageSrc": "https://im.runware.ai/image/ii/088f2c24-68d3-4407-98d1-bb09bf2e0f56.jpg",
                "imageUUID": "088f2c24-68d3-4407-98d1-bb09bf2e0f56",
                "taskUUID": "19abad0d-6ec5-40a6-b7af-203775fa5b7f"
            }
        ]
    }
}
```

An array of objects will be returned with the following parameters:

| Parameter    | Type          | Use                                                                                                                                                                    |
|--------------|---------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| imageSrc     | string        | The URL that the upscaled image can be downloaded from.                                                                                                                |
| imageUUID    | UUIDv4 string | The UUID of the upscaled image.                                                                                                                                        |
| taskUUID     | UUIDv4 string | Used to identify the async responses to this task. It must be sent to match the response to the task.                                                                   |
| bNSFWContent | boolean       | Used to inform if the image has been flagged as potentially sensitive content. True indicates the image has been flagged (is a sensitive image). False indicates the image has not been flagged. The filter occasionally returns false positives and very rarely false negatives. |
