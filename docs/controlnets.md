# Guide for ControlNets

## To use ControlNet it's necessary to use guide images, in specific formats. The easiest way to do this is via the API

## Request format for text to image

Requests must be sent in the following format:

```json
{
    "newPreProcessControlNet": {
        "taskUUID": "string",
        "preProcessorType": "string",
        "guideImageUUID": "string",
        "taskType": 11,
        "width": int,
        "height": int,
        "lowThresholdCanny": int,
        "highThresholdCanny": int
    }
}
```

## Request components

| Parameter                     | Type          | Use                                                                                                                                                       |
|-------------------------------|---------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------|
| `taskUUID`                    | UUIDV4 string | Used to identify the async responses to this task. It must be sent to match the response to the task.                                                     |
| `preProcessorType`            | string        | The preprocessor type. Here are all the available options.                                                                                                |
| `guideImageUUID`              | UUIDV4 string | The image guide UUID. The recommended option is to use the Image Upload Functionality.                                                                    |
| `guideImageUrl`               | string        | Optional. You can provide an image URL, from where the API will download the image. This adds additional delay and is not recommended.                     |
| `taskType`                    | int           | Value: 11                                                                                                                                                 |
| `width`                       | int           | Optional. Will resize the image to this width.                                                                                                            |
| `height`                      | int           | Optional. Will resize the image to this height.                                                                                                           |
| `lowThresholdCanny`           | int           | Optional. Available just for `canny` preprocessors. Defines the lower threshold. The recommended value is 100.                                             |
| `highThresholdCanny`          | int           | Optional. Available just for `canny` preprocessors. Defines the high threshold. The recommended value is 200.                                              |
| `includeHandsAndFaceOpenPose` | bool          | Optional. Available just for `openpose` preprocessors. Will include the hands and face in the pose outline. Defaults to false.                              |
| `guideImageMaskUUID`          | UUIDV4 string | Mandatory for ControlNet inpainting. The UUID of the image mask used for the inpainting process.                                                          |
| `guideImageMaskUrl`           | string        | Optional. A URL can be provided, from where the API will download the guide image for the mask. This adds additional delay and is not recommended.         |

## Preprocessor types

```plaintext
canny,
depth,
mlsd,
normalbae,
openpose,
tile,
seg,
lineart,
lineart_anime,
shuffle,
scribble,
softedge
```

## Request response

The response to the image upload request will have the following format:

```json
{
   "newImages":{
      "images":[
         {
            "imageSrc":"string",
            "imageUUID":"string",
            "bNSFWContent":bool,
            "taskUUID":"string"
         }
      ]
   }
}
```

Results will be received as an array of objects:

| Parameter      | Type          | Use                                                                                                                                                                    |
|----------------|---------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `imageSrc`     | string        | The URL of the image to be downloaded.                                                                                                                                 |
| `imageUUID`    | UUIDv4 string | The unique identifier of the image.                                                                                                                                    |
| `bNSFWContent` | boolean       | Used to inform if the image has been flagged as potentially sensitive content. True indicates the image has been flagged (is a sensitive image). False indicates the image has not been flagged. The filter occasionally returns false positives and very rarely false negatives. |
| `taskUUID`     | UUIDv4 string | Used to match the async responses to their corresponding tasks.                                                                                                        |
