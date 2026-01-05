# Changelog

All notable changes to this project will be documented in this file.

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
