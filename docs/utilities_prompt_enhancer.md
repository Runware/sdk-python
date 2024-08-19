# Prompt Enhancer (Magic Prompt)

Prompt enhancing can be used to generate different or potentially improved results for a particular topic. It works by adding keywords to a given prompt. Note that enhancing a prompt does not always preserve the intended subject and does not guarantee improved results over the original prompt.

## Request

Prompt enhancing requests must have the following format:

```json
[
  {
    "taskType": "promptEnhance",
    "taskUUID": "9da1a4ad-c3de-4470-905d-5be5c042f98a",
    "prompt": "dog",
    "promptMaxLength": 64,
    "promptVersions": 4
  }
]
```

### Parameters

| Parameter       | Type          | Description                                                                                         |
|-----------------|---------------|-----------------------------------------------------------------------------------------------------|
| taskType        | string        | Must be set to "promptEnhance" for this operation.                                                  |
| taskUUID        | UUIDv4 string | Unique identifier for the task, used to match async responses.                                      |
| prompt          | string        | The original prompt you want to enhance.                                                            |
| promptMaxLength | integer       | Maximum length of the enhanced prompt. Value between 4 and 400.                                     |
| promptVersions  | integer       | Number of enhanced prompt versions to generate. Value between 1 and 5.                              |
| includeCost     | boolean       | Optional. If set to true, the response will include the cost of the operation.                      |

## Response

Results will be delivered in the following format:

```json
{
  "data": [
    {
      "taskType": "promptEnhance",
      "taskUUID": "9da1a4ad-c3de-4470-905d-5be5c042f98a",
      "text": "dog, ilya kuvshinov, gaston bussiere, craig mullins, simon bisley, arthur rackham",
      "cost": 0
    },
    {
      "taskType": "promptEnhance",
      "taskUUID": "9da1a4ad-c3de-4470-905d-5be5c042f98a",
      "text": "dog, ilya kuvshinov, artgerm",
      "cost": 0
    },
    {
      "taskType": "promptEnhance",
      "taskUUID": "9da1a4ad-c3de-4470-905d-5be5c042f98a",
      "text": "dog, ilya kuvshinov, gaston bussiere, craig mullins, simon bisley",
      "cost": 0
    },
    {
      "taskType": "promptEnhance",
      "taskUUID": "9da1a4ad-c3de-4470-905d-5be5c042f98a",
      "text": "dog, ilya kuvshinov, artgerm, krenz cushart, greg rutkowski, pixiv. cinematic dramatic atmosphere, sharp focus, volumetric lighting, cinematic lighting, studio quality",
      "cost": 0
    }
  ]
}
```

The response contains an array of objects in the "data" field. The number of objects corresponds to the `promptVersions` requested. Each object represents an enhanced prompt suggestion.

### Response Parameters

| Parameter | Type          | Description                                                           |
|-----------|---------------|-----------------------------------------------------------------------|
| taskType  | string        | The type of task, in this case "promptEnhance".                       |
| taskUUID  | UUIDv4 string | The unique identifier matching the original request.                  |
| text      | string        | The enhanced prompt text.                                             |
| cost      | number        | The cost of the operation (included if `includeCost` was set to true).|
