# Image upload

## Images can be uploaded to be used as seed to reverse prompts and get image to text results

## Image upload request

Uploading images is necessary in order to use them as seeds for new image generation, or to run image to text and obtain prompts that would generate similar images. To upload an image send a request in the following format:

```json
{
    "newImageUpload": {
        "imageBase64": "data:image/png;base64,iVBORw0KGgoAAAA...",
        "taskUUID": "50836053-a0ee-4cf5-b9d6-ae7c5d140ada"
    }
}
```

| Parameter   | Type          | Use                                                                                                |
|-------------|---------------|----------------------------------------------------------------------------------------------------|
| imageBase64 | string        | Represents is the image file in the base64 format. Supported formats are: PNG, JPG, WEBP           |
| taskUUID    | UUIDv4 string | Task ID must be sent to match async responses with corresponding tasks.                            |

## Request response

The response to the image upload request will have the following format:

```json
{
    "newUploadedImageUUID": {
        "newImageUUID": "989ba605-1449-4e1e-b462-cd83ec9c1a67",
        "newImageSrc": "https://im.runware.dev/image/ii/989ba605-1449-4e1e-b462-cd83ec9c1a67.png",
        "taskUUID": "9ed8a593-5515-46f3-9cd7-81ab0508176c"
    }
}
```

| Parameter     | Type          | Use                                                                                     |
|---------------|---------------|-----------------------------------------------------------------------------------------|
| newImageUUID  | UUIDv4 string | Image ID. Can be used to reference the image in image-to-image (seeded generation) or image-to-prompt tasks. |
| newImageSrc   | string        | The image URL. It can be used to visualize it or display the image in UIs.              |
| taskUUID      | UUIDv4 string | Task ID must be sent to match async responses with corresponding tasks.                  |

## Image storage & retention

Images are saved in PNG format and downscaled to max. 2048 pixels in either width or height. Used images are currently retained indefinitely. Unused images are automatically deleted in 60 days.