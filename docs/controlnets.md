# ControlNet Guide

ControlNet offers advanced capabilities for precise image processing through the use of guide images in specific formats, known as preprocessed images. This powerful tool enhances the control and customization of image generation, enabling users to achieve desired artistic styles and detailed adjustments effectively.

Using ControlNet via our API simplifies the integration of guide images into your workflow. By leveraging the API, you can seamlessly incorporate preprocessed images and specify various parameters to tailor the image generation process to your exact requirements.

## Request

Our API always accepts an array of objects as input, where each object represents a specific task to be performed. The structure of the object varies depending on the type of the task. For this section, we will focus on the parameters related to the ControlNet preprocessing task.

The following JSON snippet shows the basic structure of a request object. All properties are explained in detail in the next section:

```json
[
  {
    "taskType": "imageControlNetPreProcess",
    "taskUUID": "3303f1be-b3dc-41a2-94df-ead00498db57",
    "inputImage": "ff1d9a0b-b80f-4665-ae07-8055b99f4aea",
    "preProcessorType": "canny",
    "height": 512,
    "width": 512
  }
]
```

### Parameters

| Parameter                     | Type          | Description                                                                                                                             |
|-------------------------------|---------------|-----------------------------------------------------------------------------------------------------------------------------------------|
| taskType                      | string        | Must be set to "imageControlNetPreProcess" for this operation.                                                                          |
| taskUUID                      | UUIDv4 string | Unique identifier for the task, used to match async responses.                                                                          |
| inputImage                    | UUIDv4 string | The UUID of the image to be preprocessed. Use the Image Upload functionality to obtain this.                                            |
| preProcessorType              | string        | The type of preprocessor to use. See list of available options below.                                                                   |
| width                         | integer       | Optional. Will resize the image to this width.                                                                                          |
| height                        | integer       | Optional. Will resize the image to this height.                                                                                         |
| lowThresholdCanny             | integer       | Optional. Available only for 'canny' preprocessor. Defines the lower threshold. Recommended value is 100.                               |
| highThresholdCanny            | integer       | Optional. Available only for 'canny' preprocessor. Defines the high threshold. Recommended value is 200.                                |
| includeHandsAndFaceOpenPose   | boolean       | Optional. Available only for 'openpose' preprocessor. Includes hands and face in the pose outline. Defaults to false.                   |

### Preprocessor Types

Available preprocessor types are:

```json
canny
depth
mlsd
normalbae
openpose
tile
seg
lineart
lineart_anime
shuffle
scribble
softedge
```

## Response

The response to the ControlNet preprocessing request will have the following format:

```json
{
  "data": [
    {
      "taskType": "imageControlNetPreProcess",
      "taskUUID": "3303f1be-b3dc-41a2-94df-ead00498db57",
      "guideImageUUID": "b6a06b3b-ce32-4884-ad93-c5eca7937ba0",
      "inputImageUUID": "ff1d9a0b-b80f-4665-ae07-8055b99f4aea",
      "guideImageURL": "https://im.runware.ai/image/ws/0.5/ii/b6a06b3b-ce32-4884-ad93-c5eca7937ba0.jpg",
      "cost": 0.0006
    }
  ]
}
```

### Response Parameters

| Parameter      | Type          | Description                                                                                    |
|----------------|---------------|------------------------------------------------------------------------------------------------|
| taskType       | string        | The type of task, in this case "imageControlNetPreProcess".                                    |
| taskUUID       | UUIDv4 string | The unique identifier matching the original request.                                           |
| guideImageUUID | UUIDv4 string | Unique identifier for the preprocessed guide image.                                            |
| inputImageUUID | UUIDv4 string | The UUID of the original input image.                                                          |
| guideImageURL  | string        | The URL of the preprocessed guide image. Can be used to visualize or display the image in UIs. |
| cost           | number        | The cost of the operation (if includeCost was set to true in the request).                     |
