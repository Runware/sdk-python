# Changelog

All notable changes to this project will be documented in this file.

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
