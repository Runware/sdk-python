# Image to Text

Image to text, also known as image captioning, allows you to obtain descriptive text prompts based on uploaded or previously generated images. This process is instrumental in generating textual descriptions that can be used to create additional images or provide detailed insights into visual content.

## Request

Image to text requests must have the following format:

```json
[
  {
    "taskType": "imageCaption",
    "taskUUID": "f0a5574f-d653-47f1-ab42-e2c1631f1a47",
    "inputImage": "5788104a-1ca7-4b7e-8a16-b27b57e86f87"
  }
]
```

### Parameters

| Parameter   | Type          | Description                                                                           |
|-------------|--------------|---------------------------------------------------------------------------------------|
| taskType    | string       | Must be set to "imageCaption" for this operation.                                     |
| taskUUID    | UUIDv4 string | Unique identifier for the task, used to match async responses.                        |
| inputImage  | UUIDv4 string | The UUID of the image to be analyzed. Can be from an uploaded or generated image.     |
| includeCost | boolean      | Optional. If set to true, the response will include the cost of the operation.        |

## Response

Results will be delivered in the following format:

```json
{
  "data": [
    {
      "taskType": "imageCaption",
      "taskUUID": "f0a5574f-d653-47f1-ab42-e2c1631f1a47",
      "text": "arafed troll in the jungle with a backpack and a stick, cgi animation, cinematic movie image, gremlin, pixie character, nvidia promotional image, park background, with lots of scumbling, hollywood promotional image, on island, chesley, green fog, post-nuclear",
      "cost": 0
    }
  ]
}
```

### Response Parameters

| Parameter | Type          | Description                                                           |
|-----------|---------------|-----------------------------------------------------------------------|
| taskType  | string        | The type of task, in this case "imageCaption".                        |
| taskUUID  | UUIDv4 string | The unique identifier matching the original request.                  |
| text      | string        | The resulting text prompt from analyzing the image.                   |
| cost      | number        | The cost of the operation (included if `includeCost` was set to true).|