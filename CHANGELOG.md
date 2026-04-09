# Changelog

All notable changes to this project will be documented in this file.

## [0.5.7]

### Added

- Streaming support for LLMs.
- **`ISettings`** now includes:
  - `includePrefix: Optional[bool]`
  - `audioTemperature: Optional[float]`
  - `topK: Optional[int]`
- **`IAudioInputs`** now includes:
  - `audios: Optional[List[str]]`

## [0.5.6]

### Added

- **`ISettings`** now includes:
  - `imageAutoFix: Optional[bool]`
  - `faceLimit: Optional[int]`
  - `texture: Optional[bool]`
  - `pbr: Optional[bool]`
  - `textureSeed: Optional[int]`
  - `textureAlignment: Optional[str]`
  - `textureQuality: Optional[str]`
  - `autoSize: Optional[bool]`
  - `orientation: Optional[str]`
  - `quad: Optional[bool]`
  - `compress: Optional[str]`
  - `smartLowPoly: Optional[bool]`
  - `generateParts: Optional[bool]`
  - `exportUv: Optional[bool]`
  - `geometryQuality: Optional[str]`
  - `editRegions: Optional[List[List[List[int]]]]`
  - `sequential: Optional[bool]`
  - `thinking: Optional[bool]`
  - `colorPalette: Optional[List[Union[IColorPaletteEntry, Dict[str, Any]]]]`
  - `style: Optional[str]`
  - `thinking: Optional[str]`
  - `multiClip: Optional[bool]`
  - `voiceDescription: Optional[str]`
  - Image upscale: `enhanceDetails`, `realism`, and existing upscale tuning fields on **`ISettings`** (`steps`, `seed`, `CFGScale`, prompts, scheduler, etc.)
- **`I3dInputs`** now includes `images: Optional[List[Union[str, File]]]`.
- **`IVideoInputs.audios`** for video inference.
- **`IImageUpscale`**: `targetMegapixels`, optional `upscaleFactor`, `deliveryMethod` (default `"sync"`); **`IImageUpscale.settings`** typed as **`ISettings`**.
- **`IUpscaleSettings`**: deprecated subclass of **`ISettings`** (deprecation warning on use).
- **`_upscaleGan`**: async `deliveryMethod` handling for **`IAsyncTaskResponse`** + polling.
- New dataclass **`IColorPaletteEntry`**: `hex: str`, `ratio: Optional[Union[str, float]]`.
- **`IInputs`** now includes `images: Optional[List[Union[str, File]]]`.

### Changed

- **`_pollResults`**: `imageUpscale`, `vectorize`, and `imageBackgroundRemoval` resolve to **`IImage`** (fixes **`getResponse`** returning **`IVideo`** without image URLs).
- **`_upscaleGan`**: conditional `upscaleFactor` / `targetMegapixels`; forward `deliveryMethod`.

## [0.5.5]

### Changed
- Relax `aiofiles` and `python-dotenv` pins to `>=` minimums in `setup.py` and `requirements.txt` so the package installs cleanly alongside environments that pin newer compatible releases.

## [0.5.4]

### Added
- Added `runware/version.py` as the canonical `__version__` string; `runware/__init__.py` imports it so `from runware import __version__` matches the package version.
- WebSocket connections send additional headers on connect: `X-SDK-Name: python` and `X-SDK-Version: <__version__>` (see `runware/server.py`).

## [0.5.3]

### Added
- Added `width: Optional[int]`, `height: Optional[int]`, and `fps: Optional[int]` to `IVideoUpscale`.
- Added `IElements` dataclass for video inference `inputs.elements[]` with:
  - `id: Optional[str]`
  - `description: Optional[str]`
  - `frontalImage: Optional[Union[str, File]]`
  - `images: Optional[List[Union[str, File]]]`
  - `videos: Optional[List[str]]`
  - `voice: Optional[List[str]]`
  - `tags: Optional[List[str]]`
- Added `elements: Optional[List[Union[IElements, Dict[str, Any]]]]` to `IVideoInputs`.
- `IAudioOutputFormat` extended with `wav`, `mp3`, `pcm`, `opus`, `aac`, `flac` (existing `MP3` retained).
- `ISettings`: `maxNewTokens`, `transcript`, `xVectorOnly` for reference-audio / ICL TTS flows.
- `IAudioInputs`: `audio` (reference audio URL or data URI).
- `IAudioSpeech`: `language` (e.g. input language for TTS).

### Changed
- Updated `_requestVideoUpscale()` to forward `width`, `height`, and `fps` when provided.
- Updated `IVideoInputs.__post_init__` to coerce `inputs.elements` dictionary items into `IElements` instances.

## [0.5.2]

### Added
- Added photoMaker as a nested object in imageInference using `IPhotoMakerSettings`; `IImageInference.photoMaker: Optional[Union[IPhotoMakerSettings, Dict[str, Any]]]`. Nested settings use images/inputImages + style/strength. Standalone `taskType="photoMaker"` remains a separate operation via `IPhotoMaker`.
- Added `IIpAdapter.guideImages`, `combineMethod`, `weightType`, `embedScaling`, `weightComposition`; `guideImage` optional; base supports `guideImages` with `process_image`.
- Added `IEmbedding.weight: Optional[float]`.
- Added dict coercion in `IImageInference.post_init` for `photoMaker`, `instantID`, `acePlusPlus`, `puLID`, `ultralytics`, `outpaint`, `refiner`; list coercion for embeddings (`dict` -> `IEmbedding`) and ipAdapters (`dict` -> `IIpAdapter`).
- Added `IInputReference.type` and `IInputReference.strength` (for sketch-style refs).
- Added image inference support so `inputs.referenceImages` can be `IInputReference` items; each image is run through `process_image` before send.
- Added `IGoogleProviderSettings.safetyTolerance`.
- Added `bpm`, `keyScale`, `timeSignature`, `vocalLanguage`, `coverConditioningScale`, `repaintingStart`, and `repaintingEnd` to `ISettings`.
- Added `audio: Optional[str]` to `IAudioInputs`.
- Added `IGoogleProviderSettings.resizeMode`.

### Changed
- `IImageInference`: `refiner`, `outpaint`, `instantID`, `acePlusPlus`, `puLID`, `ultralytics`, `photoMaker` typed as `Union[Type, Dict[str, Any]]`; embeddings as `List[Union[IEmbedding, Dict[str, Any]]]`; ipAdapters as `List[Union[IIpAdapter, Dict[str, Any]]]`.

## [0.5.1]

### Added
- `IVectorize`: `width`, `height`, `positivePrompt`, `providerSettings` (optional).
- `IVectorize.inputs` optional (prompt-only vectorize e.g. recraft:v4@vector).
- `VectorizeProviderSettings` type alias (`IRecraftProviderSettings`) in types.
- `_processVectorizeInputs` and `_buildVectorizeRequest` in base (vectorize follows video/3D flow).
- Single `_addProviderSettings(request_object, payload)` for image, image background removal, image upscale, vectorize, video, audio, text (replaces `_addImageProviderSettings`, `_addAudioProviderSettings`, `_addTextProviderSettings`).
- Added `post_init` to `IAudioInference` dataclass for `speech` = `Optional[Union[IAudioSpeech, Dict[str, Any]]]`.
- Added `textNormalization: Optional[bool] = None` to `ISettings` dataclass.
- Image `IAdvancedFeatures`: `layerDiffuse`, `hiresFix`, `watermark` (`IWatermark`), and `regionalPrompting` (`IRegionalPrompting` with `regions`: `List[IRegion]`).
- Video `IVideoAdvancedFeatures`: `IWatermark` (text/image, position, opacity, colors) and `watermark` field.
- Serializable advanced features: `IFluxKontext`, `IRegion`, `IRegionalPrompting`, `IWatermark`, `IAdvancedFeatures`, and `IVideoAdvancedFeatures` use `SerializableMixin` for request serialization.
- Added `avatar: Optional[str]`, `background: Optional[str]` to `IVideoInputs`.
- Added `speed: Optional[float]`, `pitch: Optional[float]`, `language: Optional[str]` to `IVideoSpeechSettings`.
- Added `expressiveness: Optional[str]`, `removeBackground: Optional[bool]`, `backgroundColor: Optional[str]` to `ISettings`.

### Changed
- Vectorize: `inputs` resolved via `process_image` when present; optional and added with `_addOptionalField`.
- `_addProviderSettings` payload type: `Union[IImageInference, IImageBackgroundRemoval, IImageUpscale, IVectorize, IVideoInference, IAudioInference, ITextInference]`.
- Request wiring: Image/video/audio builders now rely on `_addOptionalBuiltInDataTypesFields()` instead of hardcoded optional field lists, ensuring fields like `acceleration` and `webhookURL` are always forwarded when set.

## [0.5.0]

### Changed
- Replace polling with asyncio.Future for O(1) message routing via `_pending_operations` HashMap
- Add semaphore to limit concurrent requests (default 15, env `RUNWARE_MAX_CONCURRENT_REQUESTS`)
- Add jitter on reconnect to prevent thundering herd
- Increase `PING_TIMEOUT_DURATION` from 10s to 30s for stability
- Graceful cancellation of pending operations on disconnect

### Added
- Added `post_init` to `IImageInference`, `IImageCaption`, `IImageUpscale`, `IVideoInference`, `I3dInference`, `IAudioInference`, `IVideoBackgroundRemoval`
- Added `promptExtend: Optional[bool]` to `ISettings`
- Added `IBytedanceProviderSettings.optimizePromptMode: Optional[str]`
- Added `settings` to `IVideoInference` dataclass
- Added `draft: Optional[bool]`, `audio: Optional[bool]`, and `promptUpsampling: Optional[bool]` to `ISettings` dataclass
- Added `post_init` to `ISettings` dataclass

## [0.4.47]

### Added
- Added `IRGB` dataclass (`rgb: List[int]`) with validation (exactly 3 elements, each 0–255)
- Added `IRecraftProviderSettings` with `styleId: Optional[str]`, `colors: Optional[List[IRGB]]`, `backgroundColor: Optional[IRGB]`; added as `ImageProviderSettings`.
- Added `ISettings.lyrics: Optional[str]` and `ISettings.guidanceType: Optional[str]`.
- Added `negativePrompt`, `seed`, `steps`, `CFGScale`, and `settings` to `IAudioInference` dataclass.
- Added `IAudioSpeech` dataclass (`text`, `voice`, `speed`, `volume`, `pitch`, `emotion`, `tone: Optional[List[str]]`).
- Added `channels: Optional[int]` to `IAudioSettings`.
- Added `speech: Optional[IAudioSpeech]` and `settings: Optional[ISettings]` to `IAudioInference`.
- Added `languageBoost: Optional[str]` and `turbo: Optional[bool]` to `ISettings`.
- Added `edit`, `color`, `lightDirection`, `lightType`, `season` (all `Optional[str]`) to `IBriaProviderSettings`.

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
