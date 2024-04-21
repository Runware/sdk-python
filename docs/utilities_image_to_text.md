# Image to text (Image interrogator)

## Upload an image and generate the prompt used to create similar images.

Image to text is used to obtain a text prompt that can be then selected for generating additional, similar images. Previously uploaded or generated images can be used for running image to text requests.

## Request format

Send image to text requests in the following format:

```json
{
    "newReverseImageClip": {
        "imageUUID": "96eeec33-8e74-4a9f-a06c-ae58acbc3529",
        "taskUUID": "d121dd89-4621-462c-8e6a-937099fa1dcb"
    }
}
```

| Parameter | Type          | Use                                                                                             |
|-----------|---------------|-------------------------------------------------------------------------------------------------|
| imageUUID | UUIDv4 string | The UUID of the interrogated image. Will be either the UUID of an uploaded image or a generated image. |
| taskUUID  | UUIDv4 string | Used to identify the async responses to this task. It must be sent to match the response to the task. |

## Results format

Responses will arrive in the following format:

```json
{
    "newReverseClip": {
        "texts": [
            {
                "taskUUID": "48ec44fc-c484-47a4-9032-cb552976152d",
                "text": "'country house on top of a hill, idyllic, french countryside, beautiful, nature, warm lighting, crisp detail, high definition'"
            }
        ]
    }
}
```

| Parameter | Type          | Use                                                                 |
|-----------|---------------|---------------------------------------------------------------------|
| taskUUID  | UUIDv4 string | Used to identify the async responses to this task. It must be sent to match the response to the task. |
| text      | string        | The resulting text or prompt from interrogating the image.           |
