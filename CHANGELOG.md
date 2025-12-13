# Changelog

All notable changes to this project will be documented in this file.

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
