# Image Upload

Image upload is necessary for using images as seeds for new image generation, or to run image-to-text operations and obtain prompts that would generate similar images.

## Request

Image upload requests must have the following format:

```json
[
  {
    "taskType": "imageUpload",
    "taskUUID": "50836053-a0ee-4cf5-b9d6-ae7c5d140ada",
    "image": "data:image/png;base64,iVBORw0KGgo..."
  }
]
```

### Parameters

| Parameter | Type          | Description                                                                           |
|-----------|--------------|---------------------------------------------------------------------------------------|
| taskType  | string       | Must be set to "imageUpload" for this operation.                                      |
| taskUUID  | UUIDv4 string | Unique identifier for the task, used to match async responses.                        |
| image     | string       | The image file in base64 format. Supported formats are: PNG, JPG, WEBP.               |

## Response

The response to the image upload request will have the following format:

```json
{
  "data": [
    {
      "taskType": "imageUpload",
      "taskUUID": "50836053-a0ee-4cf5-b9d6-ae7c5d140ada",
      "imageUUID": "989ba605-1449-4e1e-b462-cd83ec9c1a67",
      "imageURL": "https://im.runware.ai/image/ws/0.5/ii/989ba605-1449-4e1e-b462-cd83ec9c1a67.jpg"
    }
  ]
}
```

### Response Parameters

| Parameter | Type          | Description                                                                                    |
|-----------|---------------|------------------------------------------------------------------------------------------------|
| taskType  | string        | The type of task, in this case "imageUpload".                                                  |
| taskUUID  | UUIDv4 string | The unique identifier matching the original request.                                           |
| imageUUID | UUIDv4 string | Unique identifier for the uploaded image. Use this for referencing in other operations.        |
| imageURL  | string        | The URL of the uploaded image. Can be used to visualize or display the image in UIs.           |
