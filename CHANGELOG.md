# Changelog

All notable changes to this project will be documented in this file.

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
