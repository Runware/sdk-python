from abc import abstractmethod, ABC
from enum import Enum
from dataclasses import dataclass, field, asdict, InitVar
from typing import List, Union, Optional, Callable, Any, Dict, TypeVar, Literal
import warnings


class Environment(Enum):
    PRODUCTION = "PRODUCTION"
    DEVELOPMENT = "DEVELOPMENT"
    TEST = "TEST"


class EPromptWeighting(Enum):
    COMPEL = "compel"
    SDEMBEDS = "sdembeds"


class SdkType(Enum):
    CLIENT = "CLIENT"
    SERVER = "SERVER"


class EControlMode(Enum):
    BALANCED = "balanced"
    PROMPT = "prompt"
    CONTROL_NET = "controlnet"


class ETaskType(Enum):
    IMAGE_INFERENCE = "imageInference"
    PHOTO_MAKER = "photoMaker"
    IMAGE_UPLOAD = "imageUpload"
    IMAGE_UPSCALE = "imageUpscale"
    IMAGE_BACKGROUND_REMOVAL = "imageBackgroundRemoval"
    VIDEO_BACKGROUND_REMOVAL = "removeBackground"
    VIDEO_UPSCALE = "upscale"
    IMAGE_CAPTION = "imageCaption"
    IMAGE_CONTROL_NET_PRE_PROCESS = "imageControlNetPreProcess"
    PROMPT_ENHANCE = "promptEnhance"
    AUTHENTICATION = "authentication"
    MODEL_UPLOAD = "modelUpload"
    MODEL_SEARCH = "modelSearch"
    VIDEO_INFERENCE = "videoInference"
    INFERENCE_3D = "3dInference"
    TEXT_INFERENCE = "textInference"
    AUDIO_INFERENCE = "audioInference"
    VIDEO_CAPTION = "caption"
    MEDIA_STORAGE = "mediaStorage"
    GET_RESPONSE = "getResponse"
    GET_TASK_DETAILS = "getTaskDetails"
    IMAGE_VECTORIZE = "vectorize"


class EPreProcessorGroup(Enum):
    canny = "canny"
    depth = "depth"
    mlsd = "mlsd"
    normalbae = "normalbae"
    openpose = "openpose"
    tile = "tile"
    seg = "seg"
    lineart = "lineart"
    lineart_anime = "lineart_anime"
    shuffle = "shuffle"
    scribble = "scribble"
    softedge = "softedge"


class EPreProcessor(Enum):
    canny = "canny"
    depth_leres = "depth_leres"
    depth_midas = "depth_midas"
    depth_zoe = "depth_zoe"
    inpaint_global_harmonious = "inpaint_global_harmonious"
    lineart_anime = "lineart_anime"
    lineart_coarse = "lineart_coarse"
    lineart_realistic = "lineart_realistic"
    lineart_standard = "lineart_standard"
    mlsd = "mlsd"
    normal_bae = "normal_bae"
    scribble_hed = "scribble_hed"
    scribble_pidinet = "scribble_pidinet"
    seg_ofade20k = "seg_ofade20k"
    seg_ofcoco = "seg_ofcoco"
    seg_ufade20k = "seg_ufade20k"
    shuffle = "shuffle"
    softedge_hed = "softedge_hed"
    softedge_hedsafe = "softedge_hedsafe"
    softedge_pidinet = "softedge_pidinet"
    softedge_pidisafe = "softedge_pidisafe"
    tile_gaussian = "tile_gaussian"
    openpose = "openpose"
    openpose_face = "openpose_face"
    openpose_faceonly = "openpose_faceonly"
    openpose_full = "openpose_full"
    openpose_hand = "openpose_hand"


class EOpenPosePreProcessor(Enum):
    openpose = "openpose"
    openpose_face = "openpose_face"
    openpose_faceonly = "openpose_faceonly"
    openpose_full = "openpose_full"
    openpose_hand = "openpose_hand"

class EDeliveryMethod(Enum):
    SYNC = "sync"
    ASYNC = "async"
    STREAM = "stream"

class OperationState(Enum):
    """State machine for pending operations."""
    REGISTERED = "registered"      # Future created, request NOT sent
    SENT = "sent"                  # send() completed successfully
    DISCONNECTED = "disconnected"  # Connection lost after SENT


# Define the types using Literal
IOutputType = Literal["base64Data", "dataURI", "URL"]
IOutputFormat = Literal["JPG", "PNG", "WEBP", "SVG"]
IAudioOutputFormat = Literal["wav", "mp3", "pcm", "opus", "aac", "flac", "MP3"]


@dataclass
class File:
    data: bytes


@dataclass
class IAsyncTaskResponse:
    taskType: str
    taskUUID: str


@dataclass
class IGetResponseRequest:
    taskUUID: str
    numberResults: int = 1


@dataclass
class IGetTaskDetailsRequest:
    taskUUID: str


@dataclass
class IUploadImageRequest:
    file: Union[File, str]
    taskUUID: str


@dataclass
class IUploadMediaRequest:
    media_url: str
    taskUUID: str


@dataclass
class RunwareBaseType:
    apiKey: str
    url: Optional[str] = None


@dataclass
class IImage:
    taskType: str
    imageUUID: str
    taskUUID: str
    seed: Optional[int] = None
    inputImageUUID: Optional[str] = None
    imageURL: Optional[str] = None
    imageBase64Data: Optional[str] = None
    imageDataURI: Optional[str] = None
    NSFWContent: Optional[bool] = None
    cost: Optional[float] = None


@dataclass
class ILora:
    model: str
    weight: float


@dataclass
class ILycoris:
    model: str
    weight: float


@dataclass
class IEmbedding:
    model: str
    weight: Optional[float] = None


@dataclass
class IRefiner:
    model: Union[int, str]
    startStep: Optional[int] = None
    startStepPercentage: Optional[float] = None


@dataclass(kw_only=True)
class IControlNetGeneral:
    model: str
    guideImage: Union[str, File]
    weight: Optional[float] = None
    startStep: Optional[int] = None
    endStep: Optional[int] = None
    startStepPercentage: Optional[int] = None
    endStepPercentage: Optional[int] = None
    controlMode: Optional[EControlMode] = None

    def __post_init__(self):
        if (self.startStep and self.startStepPercentage) or (
            self.endStep and self.endStepPercentage
        ):
            raise ValueError(
                "Exactly one of 'startStep/endStep' or 'startStepPercentage/endStepPercentage' must be provided."
            )


@dataclass
class IControlNetCanny(IControlNetGeneral):
    lowThresholdCanny: Optional[int] = None
    highThresholdCanny: Optional[int] = None
    preprocessor: EPreProcessor = EPreProcessor.canny


@dataclass
class IControlNetOpenPose(IControlNetGeneral):
    model: Optional[str] = None
    includeHandsAndFaceOpenPose: bool = True
    preprocessor: EOpenPosePreProcessor = EOpenPosePreProcessor.openpose


IControlNet = Union[IControlNetGeneral, IControlNetCanny, IControlNetOpenPose]


@dataclass
class IError:
    error: bool
    error_message: str
    task_uuid: str
    error_code: Optional[str] = None
    parameter: Optional[str] = None
    error_type: Optional[str] = None
    documentation: Optional[str] = None


class EModelArchitecture(Enum):
    FLUX1D = "flux1d"
    FLUX1S = "flux1s"
    PONY = "pony"
    SDHYPER = "sdhyper"
    SD1X = "sd1x"
    SD1XLCM = "sd1xlcm"
    SD3 = "sd3"
    SDXL = "sdxl"
    SDXL_LCM = "sdxllcm"
    SDXL_DISTILLED = "sdxldistilled"
    SDXL_HYPER = "sdxlhyper"
    SDXL_LIGHTNING = "sdxllightning"
    SDXL_TURBO = "sdxlturbo"


@dataclass
class IModel:
    air: str
    name: str
    version: str
    category: str
    architecture: str
    tags: List[str]
    heroImage: str
    private: bool
    comment: str

    type: Optional[str] = None
    defaultWidth: Optional[int] = None
    defaultHeight: Optional[int] = None
    defaultSteps: Optional[int] = None
    defaultScheduler: Optional[str] = None
    defaultCFG: Optional[float] = None
    defaultStrength: float = 0.0
    conditioning: Optional[str] = None
    positiveTriggerWords: Optional[str] = None

    additional_fields: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        for key, value in self.additional_fields.items():
            setattr(self, key, value)


@dataclass
class IModelSearchResponse:
    results: List[IModel]
    taskUUID: str
    taskType: str
    totalResults: int


@dataclass
class IModelSearch:
    search: Optional[str] = None
    tags: Optional[List[str]] = None
    category: Optional[Literal["checkpoint", "lora", "controlnet"]] = None
    type: Optional[str] = None
    architecture: Optional[EModelArchitecture] = None
    conditioning: Optional[str] = None
    visibility: Optional[Literal["public", "private", "all"]] = None
    limit: int = 20
    offset: int = 0
    customTaskUUID: Optional[str] = None
    retry: Optional[int] = None
    additional_params: Dict[str, Union[str, int, float, bool, None]] = field(
        default_factory=dict
    )

    def __post_init__(self):
        standard_fields = {
            "search",
            "tags",
            "category",
            "type",
            "architecture",
            "conditioning",
            "visibility",
            "limit",
            "offset",
            "customTaskUUID",
            "retry",
        }
        for key in list(self.additional_params.keys()):
            if key in standard_fields:
                del self.additional_params[key]


@dataclass
class IPhotoMaker:
    model: Union[int, str]
    positivePrompt: str
    height: int
    width: int
    numberResults: int = 1
    steps: Optional[int] = None
    outputType: Optional[IOutputType] = None
    inputImages: Optional[List[Union[str, File]]] = None
    style: Optional[str] = None
    strength: Optional[float] = None
    outputFormat: Optional[IOutputFormat] = None
    includeCost: Optional[bool] = None
    taskUUID: Optional[str] = None
    webhookURL: Optional[str] = None
    negativePrompt: Optional[str] = None
    CFGScale: Optional[float] = None
    seed: Optional[int] = None
    scheduler: Optional[str] = None
    checkNsfw: Optional[bool] = None


@dataclass
class IPhotoMakerSettings:
    images: Optional[List[Union[str, File]]] = None
    inputImages: Optional[List[Union[str, File]]] = None
    style: Optional[str] = None
    strength: Optional[float] = None

class SerializableMixin:
    def serialize(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for k, v in vars(self).items():
            if v is None or k.startswith("_"):
                continue
            if isinstance(v, SerializableMixin):
                nested = v.serialize()
                if nested:
                    result[k] = nested
            elif isinstance(v, (list, tuple)) and v and all(isinstance(x, SerializableMixin) for x in v):
                result[k] = [x.serialize() for x in v]
            else:
                result[k] = v
        return result

    def to_request_dict(self) -> Dict[str, Any]:
        data = self.serialize()
        if data:
            return {self.request_key: data}
        return {}

@dataclass
class IOutpaint:
    top: Optional[int] = None
    right: Optional[int] = None
    bottom: Optional[int] = None
    left: Optional[int] = None
    blur: Optional[int] = None


@dataclass
class IInstantID:
    inputImage: Optional[Union[File, str]] = None
    poseImage: Optional[Union[File, str]] = None
    inputImages: Optional[List[Union[str, File]]] = None
    identityNetStrength: Optional[float] = None
    adapterStrength: Optional[float] = None
    controlNetCannyWeight: Optional[float] = None
    controlNetDepthWeight: Optional[float] = None
    enhanceNonFaceRegion: bool = True


@dataclass
class IIpAdapter:
    model: Union[int, str]
    guideImage: Optional[Union[File, str]] = None
    guideImages: Optional[List[Union[str, File]]] = None
    weight: Optional[float] = None
    combineMethod: Optional[str] = None
    weightType: Optional[str] = None
    embedScaling: Optional[str] = None
    weightComposition: Optional[float] = None


@dataclass
class IAcePlusPlus:
    taskType: str
    repaintingScale: float = 0.0
    inputImages: Optional[List[Union[str, File]]] = field(default_factory=list)
    inputMasks: Optional[List[Union[str, File]]] = field(default_factory=list)
    _VALID_TASK_TYPES = ("portrait", "subject", "local_editing")

    def __post_init__(self):
        # Validate repaintingScale
        if not 0.0 <= self.repaintingScale <= 1.0:
            raise ValueError("repaintingScale must be between 0.0 and 1.0")

        # Validate taskType
        if self.taskType not in self._VALID_TASK_TYPES:
            raise ValueError(
                f"taskType must be one of {self._VALID_TASK_TYPES}, got: {self.taskType}"
            )


@dataclass
class IPuLID:
    inputImages: Optional[List[Union[str, File]]] = None  # Array of reference images (min: 1, max: 1)
    idWeight: Optional[int] = None  # Min: 0, Max: 3, Default: 1
    trueCFGScale: Optional[float] = None  # Min: 0, Max: 10
    CFGStartStep: Optional[int] = None  # Min: 0, Max: 10
    CFGStartStepPercentage: Optional[int] = None  # Min: 0, Max: 100


@dataclass
class IAcceleratorOptions(SerializableMixin):
    fbcache: Optional[bool] = None
    cacheDistance: Optional[float] = None
    teaCache: Optional[bool] = None
    cacheStartStep: Optional[int] = None
    cacheStopStep: Optional[int] = None
    cacheStartStepPercentage: Optional[int] = None
    cacheEndStepPercentage: Optional[int] = None
    cacheMaxConsecutiveSteps: Optional[int] = None
    teaCacheDistance: Optional[float] = None
    deepCache: Optional[bool] = None
    deepCacheInterval: Optional[float] = None
    deepCacheBranchId: Optional[int] = None
    deepCacheSkipMode: Optional[str] = None

    @property
    def request_key(self) -> str:
        return "acceleratorOptions"


@dataclass
class IFluxKontext(SerializableMixin):
    guidanceEndStep: Optional[int] = None
    guidanceEndStepPercentage: Optional[float] = None

    @property
    def request_key(self) -> str:
        return "fluxKontext"


@dataclass
class IRegion(SerializableMixin):
    prompt: str
    mask: Union[List[int], str]  

    @property
    def request_key(self) -> str:
        return "regions"

    def __post_init__(self) -> None:
        if isinstance(self.mask, list):
            if len(self.mask) != 4:
                raise ValueError("IRegion.mask must be a list of exactly 4 integers [x0, y0, x1, y1]")
            if not all(isinstance(v, int) for v in self.mask):
                raise TypeError("IRegion.mask list elements must all be ints")


@dataclass
class IRegionalPrompting(SerializableMixin):
    injectSteps: int
    backgroundPrompt: Optional[str] = None
    baseRatio: Optional[float] = None
    regions: Optional[List[IRegion]] = None

    @property
    def request_key(self) -> str:
        return "regionalPrompting"


@dataclass
class IWatermark(SerializableMixin):
    text: Optional[str] = None  
    image: Optional[str] = None  
    displayPosition: Optional[str] = None
    tiled: Optional[bool] = None  
    opacity: Optional[float] = None  
    fontColor: Optional[str] = None  
    bgColor: Optional[str] = None  

    @property
    def request_key(self) -> str:
        return "watermark"


@dataclass
class IAdvancedFeatures(SerializableMixin):
    fluxKontext: Optional[IFluxKontext] = None
    layerDiffuse: Optional[bool] = None  
    hiresFix: Optional[bool] = None  
    regionalPrompting: Optional[IRegionalPrompting] = None
    watermark: Optional[IWatermark] = None  

    @property
    def request_key(self) -> str:
        return "advancedFeatures"  


@dataclass
class IWanAnimate(SerializableMixin):
    mode: Optional[str] = None
    retargetPose: Optional[bool] = None
    prevSegCondFrames: Optional[int] = None

    @property
    def request_key(self) -> str:
        return "wanAnimate"


VideoAdvancedFeatureTypes = IWanAnimate


@dataclass
class IVideoAdvancedFeatures(SerializableMixin):
    videoCFGScale: Optional[float] = None  
    audioCFGScale: Optional[float] = None  
    fps: Optional[int] = None  
    videoNegativePrompt: Optional[str] = None  
    audioNegativePrompt: Optional[str] = None  
    slgLayer: Optional[int] = None
    advancedFeature: Optional[VideoAdvancedFeatureTypes] = None
    watermark: Optional[IWatermark] = None

    @property
    def request_key(self) -> str:
        return "advancedFeatures"

    def serialize(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for k, v in vars(self).items():
            if v is None or k.startswith("_"):
                continue
            if isinstance(v, SerializableMixin):
                result.update(v.to_request_dict())
            elif isinstance(v, (list, tuple)) and v and all(isinstance(x, SerializableMixin) for x in v):
                result[k] = [x.serialize() for x in v]
            else:
                result[k] = v
        return result


@dataclass
class BaseProviderSettings(SerializableMixin, ABC):
    @property
    @abstractmethod
    def provider_key(self) -> str:
        pass

    def to_request_dict(self) -> Dict[str, Any]:
        data = self.serialize()
        if data:
            return {self.provider_key: data}
        return {}

@dataclass
class IOpenAIProviderSettings(BaseProviderSettings):
    quality: Optional[str] = None
    background: Optional[str] = None
    style: Optional[str] = None

    @property
    def provider_key(self) -> str:
        return "openai"


@dataclass
class IBriaMaskSettings(SerializableMixin):
    foreground: Optional[bool] = None
    prompt: Optional[str] = None
    frameIndex: Optional[int] = None
    # An array of coordinate dictionaries defining the mask hints.
    # Each dictionary must have: x (int), y (int), type (str: "positive" or "negative")
    keyPoints: Optional[List[Dict[str, Any]]] = None


@dataclass
class IBriaProviderSettings(BaseProviderSettings):
    medium: Optional[Literal["photography", "art"]] = None
    promptEnhancement: Optional[bool] = None
    enhanceImage: Optional[bool] = None
    promptContentModeration: Optional[bool] = None
    contentModeration: Optional[bool] = None
    ipSignal: Optional[bool] = None
    preserveAlpha: Optional[bool] = None
    mode: Optional[Literal["base", "high_control", "fast"]] = None
    enhanceReferenceImages: Optional[bool] = None
    refinePrompt: Optional[bool] = None
    originalQuality: Optional[bool] = None
    forceBackgroundDetection: Optional[bool] = None
    preserveAudio: Optional[bool] = None
    autoTrim: Optional[bool] = None
    mask: Optional[IBriaMaskSettings] = None
    edit: Optional[str] = None
    color: Optional[str] = None
    lightDirection: Optional[str] = None
    lightType: Optional[str] = None
    season: Optional[str] = None

    @property
    def provider_key(self) -> str:
        return "bria"


@dataclass
class ILightricksProviderSettings(BaseProviderSettings):
    generateAudio: Optional[bool] = None

    startTime: Optional[int] = None
    duration: Optional[int] = None
    mode: Optional[Literal["replace_audio", "replace_video", "replace_audio_and_video"]] = None

    @property
    def provider_key(self) -> str:
        return "lightricks"

@dataclass
class IMidjourneyProviderSettings(BaseProviderSettings):
    quality: Optional[float] = None
    stylize: Optional[int] = None
    chaos: Optional[int] = None
    weird: Optional[int] = None
    niji: Optional[str] = None

    @property
    def provider_key(self) -> str:
        return "midjourney"


@dataclass
class IAlibabaProviderSettings(BaseProviderSettings):
    promptEnhancer: Optional[bool] = None
    promptExtend: Optional[bool] = None
    audio: Optional[bool] = None
    shotType: Optional[str] = None

    @property
    def provider_key(self) -> str:
        return "alibaba"


@dataclass
class IBlackForestLabsProviderSettings(BaseProviderSettings):
    safetyTolerance: Optional[int] = None

    @property
    def provider_key(self) -> str:
        return "bfl"


@dataclass
class IMireloProviderSettings(BaseProviderSettings):
    startOffset: Optional[int] = None

    @property
    def provider_key(self) -> str:
        return "mirelo"


@dataclass
class ISourcefulFontInput(SerializableMixin):
    fontUrl: Optional[str] = None
    text: Optional[str] = None


@dataclass
class ISourcefulProviderSettings(BaseProviderSettings):
    transparency: Optional[bool] = None
    enhancePrompt: Optional[bool] = None
    fontInputs: Optional[List[ISourcefulFontInput]] = None

    @property
    def provider_key(self) -> str:
        return "sourceful"


@dataclass
class IRGB(SerializableMixin):
    rgb: List[int]

    def __post_init__(self) -> None:
        if len(self.rgb) != 3:
            raise ValueError("IRGB.rgb must have exactly 3 elements")
        for i, v in enumerate(self.rgb):
            if not isinstance(v, int) or v < 0 or v > 255:
                raise ValueError(f"IRGB.rgb[{i}] must be an int in 0-255, got {v!r}")


@dataclass
class IRecraftProviderSettings(BaseProviderSettings):
    styleId: Optional[str] = None
    colors: Optional[List[IRGB]] = None
    backgroundColor: Optional[IRGB] = None

    @property
    def provider_key(self) -> str:
        return "recraft"


@dataclass
class IUltralytics(SerializableMixin):

    maskBlur: Optional[int] = None
    maskPadding: Optional[int] = None
    confidence: Optional[float] = None
    positivePrompt: Optional[str] = None
    negativePrompt: Optional[str] = None
    steps: Optional[int] = None
    CFGScale: Optional[float] = None
    strength: Optional[float] = None
    @property
    def request_key(self) -> str:
        return "ultralytics"


ImageProviderSettings = (
    IOpenAIProviderSettings
    | IBriaProviderSettings
    | ILightricksProviderSettings
    | IMidjourneyProviderSettings
    | IAlibabaProviderSettings
    | IBlackForestLabsProviderSettings
    | ISourcefulProviderSettings
    | IRecraftProviderSettings
)

VectorizeProviderSettings = IRecraftProviderSettings

@dataclass
class ISafety(SerializableMixin):
    tolerance: Optional[bool] = None
    checkInputs: Optional[bool] = None
    checkContent: Optional[bool] = None
    mode: Optional[str] = None

    @property
    def request_key(self) -> str:
        return "safety"


@dataclass
class ISparseStructure(SerializableMixin):
    guidanceStrength: Optional[float] = None
    guidanceRescale: Optional[float] = None
    steps: Optional[int] = None
    rescaleT: Optional[float] = None

    @property
    def request_key(self) -> str:
        return "sparseStructure"


@dataclass
class IShapeSlat(SerializableMixin):
    guidanceStrength: Optional[float] = None
    guidanceRescale: Optional[float] = None
    steps: Optional[int] = None
    rescaleT: Optional[float] = None

    @property
    def request_key(self) -> str:
        return "shapeSlat"


@dataclass
class ITexSlat(SerializableMixin):
    guidanceStrength: Optional[float] = None
    guidanceRescale: Optional[float] = None
    steps: Optional[int] = None
    rescaleT: Optional[float] = None

    @property
    def request_key(self) -> str:
        return "texSlat"


@dataclass
class ITextInferenceTool(SerializableMixin):
    """Tool definition for text inference (e.g. function-calling / JSON-schema tools)."""

    name: str
    description: str
    input_schema: Dict[str, Any]


@dataclass
class ITextInferenceToolChoice(SerializableMixin):
    """Selects how tools are used (provider-specific shape, e.g. type + name)."""

    type: str
    name: Optional[str] = None


@dataclass
class IColorPaletteEntry(SerializableMixin):
    hex: str
    ratio: Optional[Union[str, float]] = None


@dataclass
class IEditRegion(SerializableMixin):
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class ISettings(SerializableMixin):
    # Image / Text
    temperature: Optional[float] = None
    systemPrompt: Optional[str] = None
    topP: Optional[float] = None
    minP: Optional[float] = None
    repetitionPenalty: Optional[float] = None
    presencePenalty: Optional[float] = None
    frequencyPenalty: Optional[float] = None
    thinkingLevel: Optional[str] = None
    layers: Optional[int] = None
    trueCFGScale: Optional[float] = None
    quality: Optional[str] = None
    promptExtend: Optional[bool] = None
    editRegions: Optional[List[List[Union[IEditRegion, Dict[str, Any]]]]] = None
    sequential: Optional[bool] = None
    thinking: Optional[bool] = None
    colorPalette: Optional[List[Union[IColorPaletteEntry, Dict[str, Any]]]] = None
    # 3D inference
    textureSize: Optional[int] = None
    decimationTarget: Optional[int] = None
    remesh: Optional[bool] = None
    resolution: Optional[int] = None
    sparseStructure: Optional[Union[ISparseStructure, Dict[str, Any]]] = None
    shapeSlat: Optional[Union[IShapeSlat, Dict[str, Any]]] = None
    texSlat: Optional[Union[ITexSlat, Dict[str, Any]]] = None
    imageAutoFix: Optional[bool] = None
    faceLimit: Optional[int] = None
    texture: Optional[bool] = None
    pbr: Optional[bool] = None
    textureSeed: Optional[int] = None
    textureAlignment: Optional[str] = None
    textureQuality: Optional[str] = None
    useOriginalAlpha: Optional[bool] = None
    material: Optional[str] = None
    polyCount: Optional[float] = None
    taPose: Optional[bool] = None
    boundingBox: Optional[List[int]] = None
    meshMode: Optional[str] = None
    addons: Optional[List[str]] = None
    hdTexture: Optional[bool] = None
    autoSize: Optional[bool] = None
    orientation: Optional[str] = None
    quad: Optional[bool] = None
    compress: Optional[str] = None
    smartLowPoly: Optional[bool] = None
    generateParts: Optional[bool] = None
    exportUv: Optional[bool] = None
    geometryQuality: Optional[str] = None
    # Audio
    languageBoost: Optional[str] = None
    turbo: Optional[bool] = None
    lyrics: Optional[str] = None
    instrumental: Optional[bool] = None
    lyricsOptimizer: Optional[bool] = None
    guidanceType: Optional[str] = None
    textNormalization: Optional[bool] = None
    maxNewTokens: Optional[int] = None
    transcript: Optional[str] = None
    xVectorOnly: Optional[bool] = None
    bpm: Optional[int] = None
    keyScale: Optional[str] = None
    timeSignature: Optional[Union[int, str]] = None
    vocalLanguage: Optional[str] = None
    coverConditioningScale: Optional[float] = None
    repaintingStart: Optional[float] = None
    repaintingEnd: Optional[float] = None
    includePrefix: Optional[bool] = None
    audioTemperature: Optional[float] = None
    # Video
    draft: Optional[bool] = None
    audio: Optional[bool] = None
    voiceDescription: Optional[str] = None
    style: Optional[str] = None
    thinking: Optional[str] = None
    multiClip: Optional[bool] = None
    promptUpsampling: Optional[bool] = None
    expressiveness: Optional[str] = None
    removeBackground: Optional[bool] = None
    backgroundColor: Optional[str] = None
    # Text
    maxTokens: Optional[int] = None
    topK: Optional[int] = None
    stopSequences: Optional[List[str]] = None
    tools: Optional[List[Union[ITextInferenceTool, Dict[str, Any]]]] = None
    toolChoice: Optional[Union[ITextInferenceToolChoice, Dict[str, Any]]] = None
    # Image upscale 
    steps: Optional[int] = None
    seed: Optional[int] = None
    CFGScale: Optional[float] = None
    positivePrompt: Optional[str] = None
    negativePrompt: Optional[str] = None
    controlNetWeight: Optional[float] = None
    strength: Optional[float] = None
    scheduler: Optional[str] = None
    colorFix: Optional[bool] = None
    tileDiffusion: Optional[bool] = None
    clipSkip: Optional[int] = None
    enhanceDetails: Optional[bool] = None
    realism: Optional[bool] = None

    def __post_init__(self):
        if self.sparseStructure is not None and isinstance(self.sparseStructure, dict):
            self.sparseStructure = ISparseStructure(**self.sparseStructure)
        if self.shapeSlat is not None and isinstance(self.shapeSlat, dict):
            self.shapeSlat = IShapeSlat(**self.shapeSlat)
        if self.texSlat is not None and isinstance(self.texSlat, dict):
            self.texSlat = ITexSlat(**self.texSlat)
        if self.tools is not None:
            self.tools = [
                ITextInferenceTool(**t) if isinstance(t, dict) else t
                for t in self.tools
            ]
        if self.toolChoice is not None and isinstance(self.toolChoice, dict):
            self.toolChoice = ITextInferenceToolChoice(**self.toolChoice)
        if self.editRegions is not None:
            self.editRegions = [
                [
                    IEditRegion(**item) if isinstance(item, dict) else item
                    for item in image_regions
                ]
                for image_regions in self.editRegions
            ]
        if self.colorPalette is not None:
            self.colorPalette = [
                IColorPaletteEntry(**item) if isinstance(item, dict) else item
                for item in self.colorPalette
            ]

    @property
    def request_key(self) -> str:
        return "settings"


@dataclass
class IUpscaleSettings(ISettings):

    def __post_init__(self):
        super().__post_init__()
        warnings.warn(
            "IUpscaleSettings is deprecated and will be removed in a future release; use ISettings for image upscale settings instead.",
            DeprecationWarning,
            stacklevel=3,
        )


@dataclass
class IInputFrame(SerializableMixin):
    image: Union[str, File]
    frame: Optional[Union[Literal["first", "last"], int]] = None


@dataclass
class IInputReference(SerializableMixin):
    image: Union[str, File]
    tag: Optional[str] = None
    refType: Optional[str] = None
    strength: Optional[float] = None

    def serialize(self) -> Dict[str, Any]:
        data = super().serialize()
        if self.refType is not None:
            data["type"] = self.refType
            data.pop("refType", None)
        return data


@dataclass
class IInputs(SerializableMixin):
    references: Optional[List[Union[str, File]]] = None
    referenceImages: Optional[List[Union[str, File, IInputReference]]] = None
    image: Optional[Union[str, File]] = None
    images: Optional[List[Union[str, File]]] = None
    mask: Optional[Union[str, File]] = None
    superResolutionReferences: Optional[List[Union[str, File]]] = None

    @property
    def request_key(self) -> str:
        return "inputs"

    def __post_init__(self):
        if self.references:
            warnings.warn(
                "The 'references' parameter is deprecated and will be removed in a future release. Use 'referenceImages' instead.",
                DeprecationWarning,
                stacklevel=3
            )
            if self.referenceImages is None:
                self.referenceImages = self.references


@dataclass
class ITextInputs(SerializableMixin):
    images: Optional[List[Union[str, File]]] = None
    videos: Optional[List[Union[str, File]]] = None

    @property
    def request_key(self) -> str:
        return "inputs"


@dataclass
class IAudioInput(SerializableMixin):
    id: Optional[str] = None
    source: Optional[str] = None


@dataclass
class ISpeechInput(SerializableMixin):
    id: Optional[str] = None
    provider: Optional[str] = None
    voiceId: Optional[str] = None
    text: Optional[str] = None


@dataclass
class IElements(SerializableMixin):
    id: Optional[str] = None
    description: Optional[str] = None
    frontalImage: Optional[Union[str, File]] = None
    images: Optional[List[Union[str, File]]] = None
    videos: Optional[List[str]] = None
    voice: Optional[List[str]] = None
    tags: Optional[List[str]] = None


@dataclass
class IVideoInputs(SerializableMixin):
    references: Optional[List[Union[str, File, Dict[str, Any]]]] = None
    image: Optional[Union[str, File]] = None
    images: Optional[List[Union[str, File]]] = None
    frames: Optional[List[IInputFrame]] = None
    frameImages: Optional[List[IInputFrame]] = None
    referenceImages: Optional[List[Union[str, File]]] = None
    referenceVideos: Optional[List[str]] = None
    referenceAudios: Optional[List[str]] = None
    referenceVoices: Optional[List[str]] = None
    video: Optional[str] = None
    audio: Optional[Union[str, List[IAudioInput]]] = None
    audios: Optional[List[str]] = None
    speech: Optional[List[ISpeechInput]] = None
    mask: Optional[Union[str, File]] = None
    frame: Optional[str] = None
    draftId: Optional[str] = None
    videoId: Optional[str] = None
    avatar: Optional[str] = None
    background: Optional[str] = None
    elements: Optional[List[Union[IElements, Dict[str, Any]]]] = None

    def __post_init__(self):
        if self.frames is not None:
            warnings.warn(
                "The 'frames' parameter is deprecated and will be removed in a future release. Use 'frameImages' instead.",
                DeprecationWarning,
                stacklevel=3
            )
            if self.frameImages is None:
                self.frameImages = self.frames
        
        if self.references:
            warnings.warn(
                "The 'references' parameter is deprecated and will be removed in a future release. Use 'referenceImages' instead.",
                DeprecationWarning,
                stacklevel=3
            )
            if self.referenceImages is None:
                self.referenceImages = [ref for ref in self.references]
        
        if self.referenceImages:
            # Check if IInputReference objects are used and convert them to strings
            if any(isinstance(ref, IInputReference) for ref in self.referenceImages):
                warnings.warn(
                    "Using 'IInputReference' objects in 'IVideoInputs.referenceImages' is deprecated. Use strings or File objects directly instead.",
                    DeprecationWarning,
                    stacklevel=3
                )
                self.referenceImages = [ref.image if isinstance(ref, IInputReference) else ref for ref in self.referenceImages]
        if self.elements:
            self.elements = [
                IElements(**item) if isinstance(item, dict) else item
                for item in self.elements
            ]

    @property
    def request_key(self) -> str:
        return "inputs"


@dataclass
class I3dInputs(SerializableMixin):
    image: Optional[Union[str, File]] = None
    images: Optional[List[Union[str, File]]] = None
    mask: Optional[Union[str, File]] = None
    meshFile: Optional[Union[str, File]] = None

    @property
    def request_key(self) -> str:
        return "inputs"


@dataclass
class IImageInference:
    model: Union[int, str]
    positivePrompt: Optional[str] = None
    taskUUID: Optional[str] = None
    deliveryMethod: str = "sync"  
    outputType: Optional[IOutputType] = None
    outputFormat: Optional[IOutputFormat] = None
    uploadEndpoint: Optional[str] = None
    checkNsfw: InitVar[Optional[bool]] = None
    negativePrompt: Optional[str] = None
    seedImage: Optional[Union[File, str]] = None
    maskImage: Optional[Union[File, str]] = None
    strength: Optional[float] = None
    height: Optional[int] = None
    width: Optional[int] = None
    acceleratorOptions: Optional[IAcceleratorOptions] = None
    acceleration: Optional[str] = None
    advancedFeatures: Optional[IAdvancedFeatures] = None
    steps: Optional[int] = None
    scheduler: Optional[str] = None
    seed: Optional[int] = None
    CFGScale: Optional[float] = None
    clipSkip: Optional[int] = None
    promptWeighting: Optional[EPromptWeighting] = None
    numberResults: Optional[int] = 1  # default to 1
    controlNet: Optional[List[IControlNet]] = field(default_factory=list)
    lora: Optional[List[ILora]] = field(default_factory=list)
    lycoris: Optional[List[ILycoris]] = field(default_factory=list)
    includeCost: Optional[bool] = None
    onPartialImages: Optional[Callable[[List[IImage], Optional[IError]], None]] = None
    refiner: Optional[Union[IRefiner, Dict[str, Any]]] = None
    vae: Optional[str] = None
    maskMargin: Optional[int] = None
    outputQuality: Optional[int] = None
    embeddings: Optional[List[Union[IEmbedding, Dict[str, Any]]]] = field(default_factory=list)
    outpaint: Optional[Union[IOutpaint, Dict[str, Any]]] = None
    instantID: Optional[Union[IInstantID, Dict[str, Any]]] = None
    ipAdapters: Optional[List[Union[IIpAdapter, Dict[str, Any]]]] = field(default_factory=list)
    referenceImages: Optional[List[Union[str, File]]] = field(default_factory=list)
    acePlusPlus: Optional[Union[IAcePlusPlus, Dict[str, Any]]] = None
    puLID: Optional[Union[IPuLID, Dict[str, Any]]] = None
    photoMaker: Optional[Union[IPhotoMakerSettings, Dict[str, Any]]] = None
    providerSettings: Optional[ImageProviderSettings] = None
    safety: Optional[Union[ISafety, Dict[str, Any]]] = None
    settings: Optional[Union[ISettings, Dict[str, Any]]] = None
    inputs: Optional[Union[IInputs, Dict[str, Any]]] = None
    ultralytics: Optional[Union[IUltralytics, Dict[str, Any]]] = None
    useCache: Optional[bool] = None
    resolution: Optional[str] = None
    extraArgs: Optional[Dict[str, Any]] = field(default_factory=dict)
    webhookURL: Optional[str] = None
    ttl: Optional[int] = None  # time-to-live (TTL) in seconds, only applies when outputType is "URL"

    def __post_init__(self, checkNsfw: Optional[bool] = None):
        if checkNsfw is not None:
            warnings.warn(
                "checkNsfw has been deprecated and will be removed in a future version; please use safety.checkContent instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if checkNsfw:
                if self.safety is None:
                    self.safety = ISafety(checkContent=True)
                elif isinstance(self.safety, dict):
                    self.safety.setdefault("checkContent", True)
                elif hasattr(self.safety, "checkContent") and getattr(self.safety, "checkContent") is None:
                    self.safety.checkContent = True
        if self.safety is not None and isinstance(self.safety, dict):
            self.safety = ISafety(**self.safety)
        if self.settings is not None and isinstance(self.settings, dict):
            self.settings = ISettings(**self.settings)
        if self.inputs is not None and isinstance(self.inputs, dict):
            self.inputs = IInputs(**self.inputs)
        if self.outpaint is not None and isinstance(self.outpaint, dict):
            self.outpaint = IOutpaint(**self.outpaint)
        if self.refiner is not None and isinstance(self.refiner, dict):
            self.refiner = IRefiner(**self.refiner)
        if self.embeddings:
            self.embeddings = [
                IEmbedding(**item) if isinstance(item, dict) else item
                for item in self.embeddings
            ]
        if self.photoMaker is not None and isinstance(self.photoMaker, dict):
            self.photoMaker = IPhotoMakerSettings(**self.photoMaker)
        if self.instantID is not None and isinstance(self.instantID, dict):
            self.instantID = IInstantID(**self.instantID)
        if self.acePlusPlus is not None and isinstance(self.acePlusPlus, dict):
            self.acePlusPlus = IAcePlusPlus(**self.acePlusPlus)
        if self.puLID is not None and isinstance(self.puLID, dict):
            self.puLID = IPuLID(**self.puLID)
        if self.ultralytics is not None and isinstance(self.ultralytics, dict):
            self.ultralytics = IUltralytics(**self.ultralytics)
        if self.ipAdapters:
            self.ipAdapters = [
                IIpAdapter(**item) if isinstance(item, dict) else item
                for item in self.ipAdapters
            ]


@dataclass
class IImageCaption:
    inputImages: Optional[List[Union[File, str]]] = None  # Primary: array of images (UUIDs, URLs, base64, dataURI)
    inputImage: Optional[Union[File, str]] = None  # Convenience: single image, defaults to inputImages[0] if not provided
    prompt: Optional[List[str]] = None  
    model: Optional[str] = None  # Optional: AIR ID (runware:150@1, runware:150@2) - backend handles default
    includeCost: bool = False
    template: Optional[str] = None
    webhookURL: Optional[str] = None


@dataclass
class IAudioSettings(SerializableMixin):
    sampleRate: Optional[int] = None  # Min: 8000, Max: 48000, Default: 44100
    bitrate: Optional[int] = None  # Min: 32, Max: 320, Default: 128
    channels: Optional[int] = None  

    @property
    def request_key(self) -> str:
        return "audioSettings"


@dataclass
class IElevenLabsCompositionSection(SerializableMixin):
    sectionName: str  # 1-100 characters
    positiveLocalStyles: List[str]  # Styles that should be present in this section
    negativeLocalStyles: List[str]  # Styles that should not be present in this section
    lines: List[str]  # Lyrics of the section
    duration: Optional[int] = None  # Duration in seconds (3-120s)


@dataclass
class IElevenLabsCompositionPlan(SerializableMixin):
    positiveGlobalStyles: List[str]  # Styles that should be present in the entire song
    negativeGlobalStyles: List[str]  # Styles that should not be present in the entire song
    sections: List[IElevenLabsCompositionSection]  # Sections of the song


@dataclass
class IElevenLabsMusicSettings(SerializableMixin):
    compositionPlan: IElevenLabsCompositionPlan  # Music composition structure


@dataclass
class IImageToTextStructuredData:
    ageGroup: Optional[str] = None
    confidence: Optional[float] = None


@dataclass
class IImageToText:
    taskType: ETaskType
    taskUUID: str
    text: Optional[str] = None  
    structuredData: Optional[IImageToTextStructuredData] = None
    cost: Optional[float] = None


@dataclass
class IBackgroundRemovalSettings:
    returnOnlyMask: bool = False
    alphaMatting: bool = False
    postProcessMask: bool = False
    alphaMattingErodeSize: Optional[int] = None
    alphaMattingForegroundThreshold: Optional[int] = None
    alphaMattingBackgroundThreshold: Optional[int] = None
    rgba: Optional[List[int]] = None


@dataclass
class IImageBackgroundRemoval(IImageCaption):
    outputType: Optional[IOutputType] = None
    outputFormat: Optional[IOutputFormat] = None
    outputQuality: Optional[int] = None
    model: Optional[Union[int, str]] = None
    taskUUID: Optional[str] = None
    settings: Optional[Union[IBackgroundRemovalSettings, Dict[str, Any]]] = None
    providerSettings: Optional[ImageProviderSettings] = None
    safety: Optional[Union[ISafety, Dict[str, Any]]] = None

    def __post_init__(self):
        if self.settings is not None and isinstance(self.settings, dict):
            self.settings = IBackgroundRemovalSettings(**self.settings)
        if self.safety is not None and isinstance(self.safety, dict):
            self.safety = ISafety(**self.safety)


@dataclass
class IVectorize:
    inputs: Optional[IInputs] = None
    includeCost: bool = False
    taskUUID: Optional[str] = None
    model: Optional[str] = None
    outputType: Optional[IOutputType] = "URL"
    outputFormat: Optional[IOutputFormat] = "SVG"
    webhookURL: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    positivePrompt: Optional[str] = None
    providerSettings: Optional[VectorizeProviderSettings] = None


@dataclass
class IPromptEnhance:
    promptMaxLength: int
    promptVersions: int
    prompt: str
    includeCost: bool = False
    webhookURL: Optional[str] = None


@dataclass
class IEnhancedPrompt(IImageToText):
    pass

    def __hash__(self):
        return hash((self.taskType, self.taskUUID, self.text, self.cost))


@dataclass
class IImageUpscale:
    upscaleFactor: Optional[float] = None  
    targetMegapixels: Optional[int] = None
    inputImage: Optional[Union[str, File]] = None
    model: Optional[str] = None  # Model AIR ID (runware:500@1, runware:501@1, runware:502@1, runware:503@1)
    settings: Optional[Union[ISettings, Dict[str, Any]]] = None
    outputType: Optional[IOutputType] = None
    outputFormat: Optional[IOutputFormat] = None
    includeCost: bool = False
    webhookURL: Optional[str] = None
    providerSettings: Optional[ImageProviderSettings] = None
    safety: Optional[Union[ISafety, Dict[str, Any]]] = None
    inputs: Optional[Union[IInputs, Dict[str, Any]]] = None
    deliveryMethod: str = "sync"

    def __post_init__(self):
        if self.settings is not None and isinstance(self.settings, dict):
            self.settings = ISettings(**self.settings)
        if self.safety is not None and isinstance(self.safety, dict):
            self.safety = ISafety(**self.safety)
        if self.inputs is not None and isinstance(self.inputs, dict):
            self.inputs = IInputs(**self.inputs)


class ReconnectingWebsocketProps:
    def __init__(self, websocket: Any):
        self.websocket = websocket

    def add_event_listener(self, event_type: str, listener: Callable, options: Any):
        self.websocket.addEventListener(event_type, listener, options)

    def send(self, data: Any):
        self.websocket.send(data)

    def __getattr__(self, name: str):
        return getattr(self.websocket, name)


@dataclass
class UploadImageType:
    imageUUID: str
    imageURL: str
    taskUUID: str


@dataclass
class MediaStorageType:
    mediaUUID: str
    taskUUID: str


@dataclass
class IUploadModelBaseType:
    air: str
    architecture: str
    name: str
    downloadURL: str
    uniqueIdentifier: str
    version: str
    format: str
    private: bool
    category: str
    taskUUID: Optional[str] = None
    heroImageURL: Optional[str] = None
    tags: Optional[List[str]] = field(default_factory=list)
    shortDescription: Optional[str] = None
    comment: Optional[str] = None
    retry: Optional[int] = None


@dataclass
class IUploadModelControlNet(IUploadModelBaseType):
    category: str = "controlnet"
    conditioning: Optional[str] = None

    def __post_init__(self):
        if self.conditioning is None:
            raise ValueError("conditioning is required for IUploadModelCheckPoint")


@dataclass
class IUploadModelCheckPoint(IUploadModelBaseType):
    category: str = "checkpoint"
    defaultScheduler: Optional[str] = None
    type: Optional[str] = None
    defaultStrength: Optional[float] = None
    defaultWeight: Optional[float] = None
    positiveTriggerWords: Optional[str] = None
    defaultGuidanceScale: Optional[float] = None
    defaultSteps: Optional[int] = None
    negativeTriggerWords: Optional[str] = None

    def __post_init__(self):
        if self.type is None:
            raise ValueError("type is required for IUploadModelCheckPoint")

        if self.defaultScheduler is None:
            raise ValueError("defaultScheduler is required for IUploadModelCheckPoint")


@dataclass
class IUploadModelLora(IUploadModelBaseType):
    category: str = "lora"
    defaultWeight: Optional[float] = None
    positiveTriggerWords: Optional[str] = None


@dataclass
class IUploadModelResponse:
    air: str
    taskUUID: str
    taskType: str


@dataclass
class IFrameImage:
    inputImage: Union[str, File]
    frame: Optional[Union[Literal["first", "last"], int]] = None






@dataclass
class IKlingCameraConfig(SerializableMixin):
    horizontal: Optional[int] = None
    vertical: Optional[int] = None
    zoom: Optional[int] = None
    roll: Optional[int] = None
    tilt: Optional[int] = None
    pan: Optional[int] = None


@dataclass
class IKlingCameraControl(SerializableMixin):
    camera_type: Optional[str] = None
    config: Optional[IKlingCameraConfig] = None

    def serialize(self) -> Dict[str, Any]:
        result = {}
        if self.camera_type:
            result["type"] = self.camera_type
        if self.config:
            config_data = self.config.serialize()
            if config_data:
                result["config"] = config_data
        return result


@dataclass
class IGoogleProviderSettings(BaseProviderSettings):
    generateAudio: Optional[bool] = None
    enhancePrompt: Optional[bool] = None
    search: Optional[bool] = None
    searchLatitude: Optional[float] = None
    searchLongitude: Optional[float] = None
    resizeMode: Optional[str] = None
    safetyTolerance: Optional[str] = None

    @property
    def provider_key(self) -> str:
        return "google"


@dataclass
class IMinimaxProviderSettings(BaseProviderSettings):
    promptOptimizer: Optional[bool] = None

    @property
    def provider_key(self) -> str:
        return "minimax"


@dataclass
class IBytedanceProviderSettings(BaseProviderSettings):
    cameraFixed: Optional[bool] = None
    maxSequentialImages: Optional[int] = None  # Min: 1, Max: 15 - Maximum number of sequential images to generate
    fastMode: Optional[bool] = None  # When enabled, speeds up generation by sacrificing some effects. Default: false. RTF: 25-28 (fast) vs 35 (normal)
    audio: Optional[bool] = None
    draft: Optional[bool] = None
    optimizePromptMode: Optional[str] = None  

    @property
    def provider_key(self) -> str:
        return "bytedance"
    
    


@dataclass
class IKlingMultiPrompt(SerializableMixin):
    prompt: str
    duration: float


@dataclass
class IKlingAIProviderSettings(BaseProviderSettings):
    sound: Optional[bool] = None
    cameraControl: Optional[IKlingCameraControl] = None
    soundVolume: Optional[float] = None
    originalAudioVolume: Optional[float] = None
    soundEffectPrompt: Optional[str] = None
    bgmPrompt: Optional[str] = None
    asmrMode: Optional[bool] = None
    keepOriginalSound: Optional[bool] = None
    characterOrientation: Optional[str] = None
    multiPrompt: Optional[List[IKlingMultiPrompt]] = None

    @property
    def provider_key(self) -> str:
        return "klingai"


@dataclass
class ILumaConcept(SerializableMixin):
    key: Optional[str] = None


@dataclass
class ILumaProviderSettings(BaseProviderSettings):
    loop: Optional[bool] = None
    concepts: Optional[List[ILumaConcept]] = None

    @property
    def provider_key(self) -> str:
        return "lumaai"


@dataclass
class IVideoSpeechSettings(SerializableMixin):
    voice: Optional[str] = None  # Speaker voice from the available TTS speaker list
    text: Optional[str] = None  # Text script to be converted to speech (~200 characters, not UTF-8 Encoding)
    speed: Optional[float] = None
    pitch: Optional[float] = None
    language: Optional[str] = None

    @property
    def request_key(self) -> str:
        return "speech"


@dataclass
class IPixverseSpeechSettings(IVideoSpeechSettings):
    
    def __post_init__(self):
        warnings.warn(
            "IPixverseSpeechSettings is deprecated and will be removed in a future release. Use IVideoSpeechSettings instead.",
            DeprecationWarning,
            stacklevel=2
        )


@dataclass
class IPixverseProviderSettings(BaseProviderSettings):
    effect: Optional[str] = None
    cameraMovement: Optional[str] = None
    style: Optional[str] = None
    motionMode: Optional[str] = None
    soundEffectSwitch: Optional[bool] = None
    soundEffectContent: Optional[str] = None
    audio: Optional[bool] = None  
    multiClip: Optional[bool] = None  
    thinking: Optional[str] = None  

    @property
    def provider_key(self) -> str:
        return "pixverse"


@dataclass
class IViduTemplate(SerializableMixin):
    name: Optional[str] = None
    area: Optional[str] = None
    beast: Optional[str] = None

@dataclass
class IViduProviderSettings(BaseProviderSettings):
    bgm: Optional[bool] = None
    style: Optional[str] = None
    movementAmplitude: Optional[str] = None
    template: Optional[IViduTemplate] = None
    audio: Optional[bool] = None

    @property
    def provider_key(self) -> str:
        return "vidu"


@dataclass
class IElevenLabsProviderSettings(BaseProviderSettings):
    music: Optional[IElevenLabsMusicSettings] = None

    @property
    def provider_key(self) -> str:
        return "elevenlabs"


@dataclass
class IRunwayContentModeration(SerializableMixin):
    publicFigureThreshold: str = None


@dataclass
class IRunwayProviderSettings(BaseProviderSettings):
    contentModeration: Optional[IRunwayContentModeration] = None

    @property
    def provider_key(self) -> str:
        return "runway"

    def serialize(self) -> Dict[str, Any]:
        result = {}
        if self.contentModeration:
            content_moderation_data = self.contentModeration.serialize()
            if content_moderation_data:
                result["contentModeration"] = content_moderation_data
        return result


@dataclass
class ISyncSegment(SerializableMixin):
    startTime: float
    endTime: float
    ref: str
    audioStartTime: Optional[float] = None
    audioEndTime: Optional[float] = None


@dataclass
class ISyncProviderSettings(BaseProviderSettings):
    syncMode: Optional[str] = None
    editRegion: Optional[str] = None
    emotionPrompt: Optional[str] = None
    temperature: Optional[float] = None
    activeSpeakerDetection: Optional[bool] = None
    occlusionDetectionEnabled: Optional[bool] = None
    segments: Optional[List[ISyncSegment]] = None

    @property
    def provider_key(self) -> str:
        return "sync"


AudioProviderSettings = IElevenLabsProviderSettings | IKlingAIProviderSettings | IMireloProviderSettings
VideoProviderSettings = (
    IKlingAIProviderSettings
    | IGoogleProviderSettings
    | IMinimaxProviderSettings
    | IBytedanceProviderSettings
    | IPixverseProviderSettings
    | IViduProviderSettings
    | IRunwayProviderSettings
    | ILightricksProviderSettings
    | ILumaProviderSettings
    | ISyncProviderSettings
)

@dataclass
class IVideoInference:
    model: str
    positivePrompt: Optional[str] = None
    duration: Optional[float] = None
    width: Optional[int] = None
    height: Optional[int] = None
    deliveryMethod: str = "async"
    taskUUID: Optional[str] = None
    outputType: Optional[IOutputType] = None
    outputFormat: Optional[Literal["MP4", "WEBM"]] = None
    outputQuality: Optional[int] = None
    uploadEndpoint: Optional[str] = None
    includeCost: Optional[bool] = None
    negativePrompt: Optional[str] = None
    frameImages: Optional[List[Union[IFrameImage, str]]] = field(default_factory=list)
    referenceImages: Optional[List[Union[str, File]]] = field(default_factory=list)
    lora: Optional[List[ILora]] = field(default_factory=list)
    referenceVideos: Optional[List[int]] = None  # Array of video media IDs (integers) - Max 30 seconds, supported formats (mp4, mov)
    inputAudios: Optional[List[str]] = None
    fps: Optional[int] = None
    steps: Optional[int] = None
    scheduler: Optional[str] = None
    seed: Optional[int] = None
    CFGScale: Optional[float] = None
    acceleration: Optional[str] = None
    numberResults: Optional[int] = None
    providerSettings: Optional[VideoProviderSettings] = None
    speech: Optional[IVideoSpeechSettings] = None
    webhookURL: Optional[str] = None
    nsfw_check: Optional[Literal["none", "fast", "full"]] = None
    safety: Optional[Union[ISafety, Dict[str, Any]]] = None
    advancedFeatures: Optional[IVideoAdvancedFeatures] = None
    acceleratorOptions: Optional[IAcceleratorOptions] = None
    inputs: Optional[Union[IVideoInputs, Dict[str, Any]]] = None
    resolution: Optional[str] = None
    settings: Optional[Union[ISettings, Dict[str, Any]]] = None
    skipResponse: InitVar[Optional[bool]] = None

    def __post_init__(self, skipResponse: Optional[bool] = None) -> None:
        if skipResponse is not None:
            warnings.warn(
                "skipResponse has been deprecated; use deliveryMethod='async' instead",
                DeprecationWarning,
                stacklevel=2,
            )
        if self.settings is not None and isinstance(self.settings, dict):
            self.settings = ISettings(**self.settings)
        if self.safety is not None and isinstance(self.safety, dict):
            self.safety = ISafety(**self.safety)
        if self.inputs is not None and isinstance(self.inputs, dict):
            self.inputs = IVideoInputs(**self.inputs)


I3dOutputFormat = Literal["GLB", "PLY"]


@dataclass
class I3dInference:
    model: str
    positivePrompt: Optional[str] = None
    seed: Optional[int] = None
    taskUUID: Optional[str] = None
    numberResults: Optional[int] = 1
    outputType: Optional[IOutputType] = None
    outputFormat: Optional[I3dOutputFormat] = None  # "GLB" | "PLY"
    outputQuality: Optional[int] = None
    includeCost: Optional[bool] = None
    deliveryMethod: str = "async"
    webhookURL: Optional[str] = None
    inputs: Optional[Union[I3dInputs, Dict[str, Any]]] = None
    settings: Optional[Union[ISettings, Dict[str, Any]]] = None

    def __post_init__(self):
        if self.settings is not None and isinstance(self.settings, dict):
            self.settings = ISettings(**self.settings)
        if self.inputs is not None and isinstance(self.inputs, dict):
            self.inputs = I3dInputs(**self.inputs)


@dataclass
class IAudioInputs(SerializableMixin):
    audio: Optional[str] = None
    audios: Optional[List[str]] = None
    video: Optional[str] = None
    videos: Optional[List[str]] = None

    @property
    def request_key(self) -> str:
        return "inputs"


@dataclass
class IAudioSpeech(SerializableMixin):
    text: Optional[str] = None  
    voice: Optional[str] = None
    language: Optional[str] = None
    speed: Optional[float] = None
    volume: Optional[int] = None
    pitch: Optional[int] = None
    emotion: Optional[str] = None
    tone: Optional[List[str]] = None  

    @property
    def request_key(self) -> str:
        return "speech"


@dataclass
class IAudioInference:
    model: str
    positivePrompt: Optional[str] = None  # Optional when using composition plan
    negativePrompt: Optional[str] = None
    duration: Optional[float] = None  # Min: 10, Max: 300 - Optional when using composition plan
    seed: Optional[int] = None
    steps: Optional[int] = None
    strength: Optional[float] = None
    CFGScale: Optional[float] = None
    taskUUID: Optional[str] = None
    outputType: Optional[IOutputType] = None
    outputFormat: Optional[IAudioOutputFormat] = None
    outputQuality: Optional[int] = None
    audioSettings: Optional[IAudioSettings] = None
    includeCost: Optional[bool] = None
    numberResults: Optional[int] = 1
    deliveryMethod: str = "sync"  # "sync" | "async"
    uploadEndpoint: Optional[str] = None
    webhookURL: Optional[str] = None
    providerSettings: Optional[AudioProviderSettings] = None  
    inputs: Optional[Union[IAudioInputs, Dict[str, Any]]] = None
    speech: Optional[Union[IAudioSpeech, Dict[str, Any]]] = None
    settings: Optional[Union[ISettings, Dict[str, Any]]] = None

    def __post_init__(self):
        if self.settings is not None and isinstance(self.settings, dict):
            self.settings = ISettings(**self.settings)
        if self.inputs is not None and isinstance(self.inputs, dict):
            self.inputs = IAudioInputs(**self.inputs)
        if self.speech is not None and isinstance(self.speech, dict):
            self.speech = IAudioSpeech(**self.speech)


@dataclass
class IObject:
    uuid: str
    url: str


@dataclass
class I3dOutput:
    files: Optional[List[IObject]] = None


@dataclass
class IOutput:
    draftId: Optional[str] = None
    videoId: Optional[str] = None


@dataclass
class IVideo:
    taskType: str
    taskUUID: str
    status: Optional[str] = None
    videoUUID: Optional[str] = None
    videoURL: Optional[str] = None
    mediaUUID: Optional[str] = None
    mediaURL: Optional[str] = None
    cost: Optional[float] = None
    seed: Optional[int] = None
    outputs: Optional[IOutput] = None


@dataclass
class I3d:
    taskType: str
    taskUUID: str
    cost: Optional[float] = None
    status: Optional[str] = None
    seed: Optional[int] = None
    outputs: Optional[I3dOutput] = None


@dataclass
class ITextInferenceMessage:
    role: str
    content: str


@dataclass
class ITextInferenceCompletionTokensDetails:
    reasoningTokens: Optional[int] = None


@dataclass
class ITextInferenceUsageModality:
    modality: Optional[str] = None
    tokens: Optional[int] = None
    cost: Optional[float] = None
    costDisplay: Optional[str] = None


@dataclass
class ITextInferenceUsageTokenPromptCache:
    modalities: Optional[List[ITextInferenceUsageModality]] = None
    billableTokens: Optional[int] = None
    cost: Optional[float] = None
    costDisplay: Optional[str] = None


@dataclass
class ITextInferenceUsageTokenCompletion:
    billableTokens: Optional[int] = None
    textTokens: Optional[int] = None
    reasoningTokens: Optional[int] = None
    cost: Optional[float] = None
    costDisplay: Optional[str] = None


@dataclass
class ITextInferenceUsageTokensBreakdown:
    prompt: Optional[ITextInferenceUsageTokenPromptCache] = None
    cache: Optional[ITextInferenceUsageTokenPromptCache] = None
    completion: Optional[ITextInferenceUsageTokenCompletion] = None


@dataclass
class ITextInferenceUsageCostBreakdown:
    tokens: Optional[ITextInferenceUsageTokensBreakdown] = None
    total: Optional[float] = None
    totalDisplay: Optional[str] = None


@dataclass
class ITextInferenceUsage:
    promptTokens: Optional[int] = None
    completionTokens: Optional[int] = None
    totalTokens: Optional[int] = None
    thinkingTokens: Optional[int] = None
    completionTokensDetails: Optional[ITextInferenceCompletionTokensDetails] = None
    costBreakdown: Optional[ITextInferenceUsageCostBreakdown] = None


TextProviderSettings = IGoogleProviderSettings


@dataclass
class ITextInference:
    model: str
    messages: List[ITextInferenceMessage]
    taskUUID: Optional[str] = None
    deliveryMethod: str = "sync"
    numberResults: Optional[int] = 1
    seed: Optional[int] = None
    includeCost: Optional[bool] = None
    includeUsage: Optional[bool] = None
    settings: Optional[Union[ISettings, Dict[str, Any]]] = None
    inputs: Optional[Union[ITextInputs, Dict[str, Any]]] = None
    providerSettings: Optional[TextProviderSettings] = None
    webhookURL: Optional[str] = None

    def __post_init__(self) -> None:
        if self.settings is not None and isinstance(self.settings, dict):
            self.settings = ISettings(**self.settings)
        if self.inputs is not None and isinstance(self.inputs, dict):
            self.inputs = ITextInputs(**self.inputs)


@dataclass
class IText:
    taskType: str
    taskUUID: str
    text: Optional[str] = None
    finishReason: Optional[str] = None
    usage: Optional[ITextInferenceUsage] = None
    cost: Optional[float] = None
    status: Optional[str] = None
    reasoningContent: Optional[List[str]] = None
    seed: Optional[int] = None
    thoughtSignature: Optional[str] = None


@dataclass
class IAudio:
    taskType: str
    taskUUID: str
    status: Optional[str] = None
    audioUUID: Optional[str] = None
    audioURL: Optional[str] = None
    audioBase64Data: Optional[str] = None
    audioDataURI: Optional[str] = None
    videoUUID: Optional[str] = None
    videoURL: Optional[str] = None
    seed: Optional[int] = None
    cost: Optional[float] = None


@dataclass
class IVideoCaptionInputs:
    video: str  # Video URL or UUID


@dataclass
class IVideoBackgroundRemovalInputs:
    video: str  # Video URL or UUID


@dataclass
class IVideoCaption:
    model: str
    inputs: IVideoCaptionInputs
    deliveryMethod: str = "async"
    taskUUID: Optional[str] = None
    includeCost: Optional[bool] = None
    webhookURL: Optional[str] = None


@dataclass
class IVideoBackgroundRemovalSettings:
    rgba: Optional[List[int]] = None  # Background color [r, g, b, a]
    background_color: Optional[str] = None  # Predefined colors: "Transparent", "Black", "White", etc.


@dataclass
class IVideoBackgroundRemoval:
    model: str
    inputs: IVideoBackgroundRemovalInputs
    deliveryMethod: str = "async"
    taskUUID: Optional[str] = None
    includeCost: Optional[bool] = None
    webhookURL: Optional[str] = None
    outputFormat: Optional[str] = None  # MP4, WEBM
    settings: Optional[Union[IVideoBackgroundRemovalSettings, Dict[str, Any]]] = None

    def __post_init__(self):
        if self.settings is not None and isinstance(self.settings, dict):
            self.settings = IVideoBackgroundRemovalSettings(**self.settings)


@dataclass
class IVideoUpscaleInputs:
    video: str  # Video URL or UUID


@dataclass
class IVideoUpscale:
    model: str
    inputs: IVideoUpscaleInputs
    upscaleFactor: Optional[int] = None  # 2 or 4 (optional, not supported by all models)
    deliveryMethod: str = "async"
    taskUUID: Optional[str] = None
    includeCost: Optional[bool] = None
    webhookURL: Optional[str] = None
    outputType: Optional[IOutputType] = None
    outputFormat: Optional[str] = None  # MP4, WEBM
    width: Optional[int] = None
    height: Optional[int] = None
    fps: Optional[int] = None


@dataclass
class IVideoToText:
    taskType: str
    taskUUID: str
    text: Optional[str] = None
    structuredData: Optional[Dict[str, Any]] = None
    status: Optional[str] = None
    cost: Optional[float] = None


@dataclass
class ITaskDetails:
    taskType: str
    taskUUID: str
    request: List[
        Union[
            IImageInference,
            IPhotoMaker,
            IImageCaption,
            IImageBackgroundRemoval,
            IImageUpscale,
            IPromptEnhance,
            IModelSearch,
            IVideoInference,
            IVideoCaption,
            IVideoBackgroundRemoval,
            IVideoUpscale,
            IAudioInference,
            I3dInference,
            ITextInference,
            IGetResponseRequest,
            IGetTaskDetailsRequest,
            IVectorize,
            Dict[str, Any],
        ]
    ]
    response: List[
        Union[IImage, IVideo, IAudio, IVideoToText, IImageToText, I3d, IText, IEnhancedPrompt, Dict[str, Any]]
    ]


# The GetWithPromiseCallBackType is defined using the Callable type from the typing module. It represents a function that takes a dictionary
# with specific keys and returns either a boolean or None.
# The dictionary should have the following keys:
# resolve: A function that takes a value of any type and returns None.
# reject: A function that takes a value of any type and returns None.
# intervalId: A value of any type representing the interval ID.
# You can use these types in your Python code to define variables, parameters, or return types that match the corresponding TypeScript types.
#
# def on_message(event: Any):
#     # Handle WebSocket message event
#     pass
#
# websocket = ReconnectingWebsocketProps(websocket_object)
# websocket.add_event_listener("message", on_message, {})
#
# uploaded_image = UploadImageType("abc123", "image.png", "task123")
#
# def get_with_promise(callback_data: Dict[str, Union[Callable[[Any], None], Any]]) -> Union[bool, None]:
#     # Implement the callback function logic here
#     pass


GetWithPromiseCallBackType = Callable[
    [Dict[str, Union[Callable[[Any], None], Any]]], Union[bool, None]
]


# The ListenerType class is defined to represent the structure of a listener.
# The key parameter is a string that represents a unique identifier for the listener.
# The listener parameter is a callable function that takes a single argument msg of type Any and returns None.
# It represents the function to be called when the corresponding event occurs.
# The group_key parameter is an optional string that represents a group identifier for the listener. It allows grouping listeners together based on a common key.
# You can create instances of ListenerType by providing the required parameters:
#
# def on_message(msg: Any):
#     # Handle the message
#     print(msg)
#
# listener = ListenerType("message_listener", on_message, group_key="message_group")

# In this example, we define a function on_message that takes a single argument msg and handles the received message.
# We then create an instance of ListenerType called listener by providing the key "message_listener",
# the on_message function as the listener, and an optional group key "message_group".
# You can store instances of ListenerType in a list or dictionary to manage multiple listeners in your application.

# listeners = [
#     ListenerType("listener1", on_message1),
#     ListenerType("listener2", on_message2, group_key="group1"),
#     ListenerType("listener3", on_message3, group_key="group1"),
# ]


class ListenerType:
    def __init__(
        self,
        key: str,
        listener: Callable[[Any], None],
        group_key: Optional[str] = None,
        debug_message: Optional[str] = None,
    ):
        """
        Initialize a new ListenerType instance.

        :param key: str, a unique identifier for the listener.
        :param listener: Callable[[Any], None], the function to be called when the listener is triggered.
        :param group_key: Optional[str], an optional grouping key that can be used to categorize listeners.
        """
        self.key = key
        self.listener = listener
        self.group_key = group_key
        self.debug_message = debug_message

    def __str__(self):
        return f"ListenerType(key={self.key}, listener={self.listener}, group_key={self.group_key}, debug_message={self.debug_message})"

    def __repr__(self):
        return self.__str__()


T = TypeVar("T")
Keys = TypeVar("Keys")


class RequireAtLeastOne:
    def __init__(self, data: Dict[str, Any], required_keys: Union[str, Keys]):
        if not isinstance(data, dict):
            raise TypeError("data must be a dictionary")

        self.data = data
        self.required_keys = required_keys

        if not isinstance(required_keys, (list, tuple)):
            required_keys = [required_keys]

        missing_keys = [key for key in required_keys if key not in data]
        if len(missing_keys) == len(required_keys):
            raise ValueError(
                f"At least one of the required keys must be present: {', '.join(required_keys)}"
            )

    def __getitem__(self, key: str):
        return self.data[key]

    def __setitem__(self, key: str, value: Any):
        self.data[key] = value

    def __delitem__(self, key: str):
        del self.data[key]

    def __contains__(self, key: str):
        return key in self.data

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)


class RequireOnlyOne(RequireAtLeastOne):
    def __init__(self, data: Dict[str, Any], required_keys: Union[str, Keys]):
        super().__init__(data, required_keys)

        if not isinstance(required_keys, (list, tuple)):
            required_keys = [required_keys]

        provided_keys = [key for key in required_keys if key in data]
        if len(provided_keys) > 1:
            raise ValueError(
                f"Only one key can be provided: {', '.join(provided_keys)}"
            )
