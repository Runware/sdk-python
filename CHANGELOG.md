# Changelog

All notable changes to this project will be documented in this file.

## [0.4.46]

### Added
- Added `IKlingMultiPrompt`: new dataclass with `prompt: str` and `duration: float` for Kling multiPrompt payloads.

### Changed
- `fileToBase64()`: allow `application/octet-stream` as MIME type for `.glb` and `.ply` files when MIME cannot be guessed.
- Recursive serialization in `SerializableMixin.serialize()`: replaced flat `asdict` + None filter with a recursive implementation that:
  - Skips `None` and keys starting with `_`
  - Recursively serializes nested `SerializableMixin` instances
  - Serializes lists/tuples of `SerializableMixin` as lists of dicts (e.g. for `multiPrompt` and other list payloads).

## [0.4.45]

### Added
- Added `ISparseStructure`, `IShapeSlat`, `ITexSlat` (guidanceStrength, guidanceRescale, steps, rescaleT).
- Added `I3dInference.settings: Optional[ISettings]` and `outputQuality: Optional[int]`.
- Added `I3dInputs.meshFile: Optional[Union[str, File]]`.

### Changed
- Extended `ISettings` with 3D fields: textureSize, decimationTarget, remesh, resolution, sparseStructure, shapeSlat, texSlat.

## [0.4.44]

### Added
- Added support for textInference task type (TEXT_INFERENCE in ETaskType enum)
- Added `ITextInferenceMessage` with role and content for chat messages
- Added `ITextInferenceUsage` for token usage (promptTokens, completionTokens, totalTokens, thinkingTokens)
- Added `IGoogleTextProviderSettings` with thinkingLevel for Gemini
- Added `TextProviderSettings` type alias for text inference provider settings
- Added `ITextInference` for requests (model, messages, taskUUID, deliveryMethod, maxTokens, temperature, topP, topK, seed, stopSequences, includeCost, providerSettings)
- Added `IText` for responses (taskType, taskUUID, text, finishReason, usage, cost, status)
- Added `textInference()` to Runware
- Added `_buildTextRequest`, `_requestText`, `_handleInitialTextResponse` in base
- Added `_addTextProviderSettings` for provider settings (matching image/video/audio pattern)
- Added `getResponse()` support for `List[IText]` when polling textInference tasks
- Added `TEXT_INITIAL_TIMEOUT` for async delivery (configurable via RUNWARE_TEXT_INITIAL_TIMEOUT)
- Added `TEXT_POLLING_DELAY` for polling cadence (configurable via RUNWARE_TEXT_POLLING_DELAY)
- Added `IVideoInference.scheduler: Optional[str] = None`

### Changed
- Enabled async delivery for text inference (returns IAsyncTaskResponse and uses getResponse() for polling)

## [0.4.43]

### Added
- Added support for "3dInference" task type
- Added `I3dInputs` dataclass with image and mask parameters for 3D inference inputs
- Added `I3dOutputFormat` type: Literal["GLB", "PLY"]
- Added `I3dInference` dataclass for 3D inference requests (model, positivePrompt, seed, inputs, outputFormat, etc.)
- Added `IObject` dataclass with uuid and url for 3D output files
- Added `I3dOutput` dataclass with `files: Optional[List[IObject]] = None` for 3D output files in responses
- Added `I3d` dataclass for 3D inference responses (taskType, taskUUID, cost, status, seed, outputs)
- Added `inference3d()` method to Runware for 3D inference
- Added `getResponse()` support for `List[I3d]` when polling 3dInference tasks
- Added `_handleInitial3dResponse`, `_process3dInputs`, `_build3dRequest`, `_request3d` in base
- Added `IUltralytics` dataclass with the following parameters:
  - `maskBlur: Optional[int]`
  - `maskPadding: Optional[int]`
  - `confidence: Optional[float]`
  - `positivePrompt: Optional[str]`
  - `negativePrompt: Optional[str]`
  - `steps: Optional[int]`
  - `CFGScale: Optional[float]`
  - `strength: Optional[float]`
- Added `IImageInference.ultralytics: Optional[IUltralytics]`
- Added `quality: Optional[str] = None` to `ISettings`
- Added `audio: Optional[bool] = None` to `IViduProviderSettings`

### Changed
- Split `IOutput` into media-specific output types: `IOutput` (video: draftId, videoId) and `I3dOutput` (3D: files)

## [0.4.42]

### Added
- Added `IOutput` dataclass with the following parameters:
  - `draftId: Optional[str] = None`
  - `videoId: Optional[str] = None`
- Added `IVideo.outputs: Optional[IOutput] = None` to support draft ID in video responses
- Added `IVideoInputs.draftId: Optional[str] = None` for passing draft task ID in video inference inputs for bytedance:seedance@1.5-pro
- Added `IVideoInputs.videoId: Optional[str] = None` for passing video ID in video inference inputs for openai:3@1
- Added `IBytedanceProviderSettings.draft: Optional[bool] = None` to enable draft mode for Seedance video generation
- Added `IInputs.mask: Optional[Union[str, File]] = None`
- Added `ISourcefulProviderSettings` with the following parameters:
  - `transparency: Optional[bool] = None`
  - `enhancePrompt: Optional[bool] = None`
  - `fontInputs: Optional[List[Dict[str, Any]]] = None`
- Added `IInputs.superResolutionReferences: Optional[List[Union[str, File]]] = None` for super resolution guidance in image-to-image inference

### Changed
- Enhanced `instantiateDataclass()` function to automatically handle nested dataclasses

## [0.4.41]

### Changed
- Updated `IBriaProviderSettings.preserveAudio` default from `Optional[bool] = True` to `None`
- Updated `IBriaProviderSettings.autoTrim` default from `Optional[bool] = False` to `None`

### Added
- Added `taskUUID` attribute to `IUploadModelBaseType` to allow passing `taskUUID` from user in `modelUpload()` method

## [0.4.40]

### Added
- Added `_retry_with_reconnect()` method that wraps all public API methods for automatic reconnection on authentication errors (up to 10 retries)
- Added `MAX_RETRY_ATTEMPTS` constant in `runware/utils.py` for retry mechanism configuration
- Added `IMireloProviderSettings` with `startOffset: Optional[int] = None` parameter
- Added `ISettings.layers: Optional[int] = None` parameter
- Added `ISettings.trueCFGScale: Optional[float] = None` parameter
- Added `IAudio.videoUUID: Optional[str] = None` field
- Added `IAudio.videoURL: Optional[str] = None` field
- Added `IAudio.seed: Optional[int] = None` field

### Fixed
- Fixed duplicate heartbeat tasks by cancelling existing task before creating new one in `runware/server.py`
- Fixed connection loss detection during request waits with session UUID tracking
- Fixed `conflictTaskUUID` error handling to extract `taskType` and `deliveryMethod` from error context instead of relying on global message state


### Changed
- Refactored retry logic: Removed redundant `asyncRetry` calls from 9 methods and them wrapped with `_retry_with_reconnect`

## [0.4.39]

### Added
- Added `MODEL_UPLOAD_TIMEOUT` constant in `runware/utils.py` with 15-minute default (900000ms) for large model uploads
- Added `RUNWARE_MODEL_UPLOAD_TIMEOUT` environment variable support for configuring model upload timeout

### Changed
- Updated `_modelUpload()` in `runware/base.py` to use `MODEL_UPLOAD_TIMEOUT` instead of general timeout to prevent timeouts during large LoRA model uploads

## [0.4.38]

### Added
- Added `IWanAnimate` dataclass with the following parameters:
  - `mode: Optional[str] = None`
  - `retargetPose: Optional[bool] = None`
  - `prevSegCondFrames: Optional[int] = None`
- Added `VideoAdvancedFeatureTypes` union type: `VideoAdvancedFeatureTypes = IWanAnimate`
- Updated `IVideoAdvancedFeatures` to support `advancedFeature: Optional[VideoAdvancedFeatureTypes] = None` for incorporating WanAnimate features
- Added `IVideoInputs.referenceVoices: Optional[List[str]] = None`
- Added `IKlingAIProviderSettings.characterOrientation: Optional[str] = None`

### Fixed
- Fixed NoneType error when initial requests timeout by returning `IAsyncTaskResponse` object instead of processing None values
- Added error handling in `_handleInitialVideoResponse`, `_handleInitialImageResponse`, and `_handleInitialAudioResponse` to handle empty/None initial responses
- Added validation in `instantiateDataclassList()` method to raise descriptive error when `data_list` is None or empty
- Improved error messages for timeout scenarios with detailed context including TaskUUID and delivery method

### Changed
- Refactored serialization: Removed `BaseRequestField` class and moved `to_request_dict()` method to `SerializableMixin`
- Updated multiple dataclasses to inherit from `SerializableMixin` instead of `BaseRequestField`:
  - `IAcceleratorOptions`
  - `ISafety`
  - `ISettings`
  - `IInputs`
  - `IVideoInputs`
  - `IAudioSettings`
  - `IVideoSpeechSettings`
  - `IAudioInputs`
- Updated `IVideoAdvancedFeatures.serialize()` method to properly handle `advancedFeature` serialization using `to_request_dict()`

## [0.4.37]

### Added
- Added `audio` parameter to `IBytedanceProviderSettings` (providerSettings.bytedance.audio)
- Added new parameters to `IBriaProviderSettings`:
  - `preserveAudio` 
  - `autoTrim` 
- Added `IBriaMaskSettings` with the following parameters:
  - `foreground`
  - `prompt`
  - `frameIndex`
  - `keyPoints`

### Changed
- Added validation in `_pollResults()`: if `number_results` is None, then `number_results = 1`

## [0.4.36]

### Added
- Added `IBlackForestLabsProviderSettings` with `safetyTolerance` parameter
- Added new `ISettings` class with the following parameters:
  - `temperature`
  - `systemPrompt`
  - `topP`
- Added `search` parameter to `IGoogleProviderSettings`
- Added new parameters to `IAlibabaProviderSettings`:
  - `promptExtend`
  - `audio`
  - `shotType`

### Changed
- Updated `IInputs.references` default value from `field(default_factory=list)` to `None`
- Updated `IVideoInputs.references` default value from `field(default_factory=list)` to `None`

## [0.4.35] 

### Added
- Added provider setting for Alibaba: `alibaba.promptEnhancer`
- Added `IAudioInput` and `ISpeechInput` to `IVideoInputs` for video inference
- Added `ISyncProviderSettings` to video inference
- Added `sound` parameter to `IKlingAIProviderSettings` (providerSetting.Kling.sound)

### Changed
- Updated `referenceImages` type from `Optional[List[IInputReference]]` to `Optional[List[Union[str, File, IInputReference]]]` in `IInputs`
- Renamed `IPixverseSpeechSettings` to `IVideoSpeechSettings` with backward compatibility
- Made `numberResults` optional in `videoInference`
