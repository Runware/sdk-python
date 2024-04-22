from enum import Enum
from typing import List, Union, Optional, Callable, Any, Dict, TypeVar


class Environment(Enum):
    PRODUCTION = "PRODUCTION"
    DEVELOPMENT = "DEVELOPMENT"
    TEST = "TEST"


class SdkType(Enum):
    CLIENT = "CLIENT"
    SERVER = "SERVER"


class EControlMode(Enum):
    BALANCED = "balanced"
    PROMPT = "prompt"
    CONTROL_NET = "controlnet"


class EPreProcessorGroup(Enum):
    CANNY = "canny"
    DEPTH = "depth"
    MLSDD = "mlsd"
    NORMALBAE = "normalbae"
    OPENPOSE = "openpose"
    TILE = "tile"
    SEG = "seg"
    LINEART = "lineart"
    LINEART_ANIME = "lineart_anime"
    SHUFFLE = "shuffle"
    SCRIBBLE = "scribble"
    SOFTEDGE = "softedge"


class EPreProcessor(Enum):
    CANNY = "canny"
    DEPTH_LERES = "depth_leres"
    DEPTH_MIDAS = "depth_midas"
    DEPTH_ZOE = "depth_zoe"
    INPAINT_GLOBAL_HARMONIOUS = "inpaint_global_harmonious"
    LINEART_ANIME = "lineart_anime"
    LINEART_COARSE = "lineart_coarse"
    LINEART_REALISTIC = "lineart_realistic"
    LINEART_STANDARD = "lineart_standard"
    MLSDD = "mlsd"
    NORMAL_BAE = "normal_bae"
    SCRIBBLE_HED = "scribble_hed"
    SCRIBBLE_PIDINET = "scribble_pidinet"
    SEG_OFADE20K = "seg_ofade20k"
    SEG_OFCOCO = "seg_ofcoco"
    SEG_UFADE20K = "seg_ufade20k"
    SHUFFLE = "shuffle"
    SOFTEDGE_HED = "softedge_hed"
    SOFTEDGE_HEDSAFE = "softedge_hedsafe"
    SOFTEDGE_PIDINET = "softedge_pidinet"
    SOFTEDGE_PIDISAFE = "softedge_pidisafe"
    TILE_GAUSSIAN = "tile_gaussian"
    OPENPOSE = "openpose"
    OPENPOSE_FACE = "openpose_face"
    OPENPOSE_FACEONLY = "openpose_faceonly"
    OPENPOSE_FULL = "openpose_full"
    OPENPOSE_HAND = "openpose_hand"


class EOpenPosePreProcessor(Enum):
    OPENPOSE = "openpose"
    OPENPOSE_FACE = "openpose_face"
    OPENPOSE_FACEONLY = "openpose_faceonly"
    OPENPOSE_FULL = "openpose_full"
    OPENPOSE_HAND = "openpose_hand"


class RunwareBaseType:
    def __init__(self, apiKey: str, url: str = None):
        self.apiKey = apiKey
        self.url = url


class IImage:
    def __init__(
        self, imageSrc: str, imageUUID: str, taskUUID: str, bNSFWContent: bool
    ):
        self.imageSrc = imageSrc
        self.imageUUID = imageUUID
        self.taskUUID = taskUUID
        self.bNSFWContent = bNSFWContent


class ILora:
    def __init__(self, modelId: str, weight: float):
        self.modelId = modelId
        self.weight = weight


class IControlNet:
    def __init__(
        self,
        preprocessor: EPreProcessor,
        weight: float,
        start_step: int,
        end_step: int,
        guide_image: Union[str, bytes],
        guide_image_unprocessed: Union[str, bytes],
        control_mode: EControlMode,
    ):
        self.preprocessor = preprocessor
        self.weight = weight
        self.start_step = start_step
        self.end_step = end_step
        self.guide_image = guide_image
        self.guide_image_unprocessed = guide_image_unprocessed
        self.control_mode = control_mode


class IControlNetGeneral:
    def __init__(
        self,
        preprocessor: EPreProcessor,
        weight: float,
        start_step: int,
        end_step: int,
        guide_image: Union[str, bytes],
        guide_image_unprocessed: Union[str, bytes],
        control_mode: EControlMode,
    ):
        self.preprocessor = preprocessor
        self.weight = weight
        self.start_step = start_step
        self.end_step = end_step
        self.guide_image = guide_image
        self.guide_image_unprocessed = guide_image_unprocessed
        self.control_mode = control_mode


class IControlNetA(IControlNetGeneral):
    def __init__(
        self,
        preprocessor: EPreProcessor,
        weight: float,
        start_step: int,
        end_step: int,
        guide_image: Union[str, bytes] = None,
        guide_image_unprocessed: Union[str, bytes] = None,
        control_mode: EControlMode = None,
    ):
        super().__init__(
            preprocessor,
            weight,
            start_step,
            end_step,
            guide_image,
            guide_image_unprocessed,
            control_mode,
        )
        if guide_image is None and guide_image_unprocessed is None:
            raise ValueError(
                "Either guide_image or guide_image_unprocessed must be provided."
            )


class IControlNetCanny(IControlNetA):
    def __init__(
        self,
        weight: float,
        start_step: int,
        end_step: int,
        guide_image: Union[str, bytes] = None,
        guide_image_unprocessed: Union[str, bytes] = None,
        control_mode: EControlMode = None,
        low_threshold_canny: int = None,
        high_threshold_canny: int = None,
    ):
        super().__init__(
            EPreProcessor.CANNY,
            weight,
            start_step,
            end_step,
            guide_image,
            guide_image_unprocessed,
            control_mode,
        )
        self.low_threshold_canny = low_threshold_canny
        self.high_threshold_canny = high_threshold_canny


class IControlNetHandsAndFace(IControlNetA):
    def __init__(
        self,
        preprocessor: EOpenPosePreProcessor,
        weight: float,
        start_step: int,
        end_step: int,
        guide_image: Union[str, bytes] = None,
        guide_image_unprocessed: Union[str, bytes] = None,
        control_mode: EControlMode = None,
        include_hands_and_face_open_pose: bool = None,
    ):
        super().__init__(
            preprocessor,
            weight,
            start_step,
            end_step,
            guide_image,
            guide_image_unprocessed,
            control_mode,
        )
        self.include_hands_and_face_open_pose = include_hands_and_face_open_pose


IControlNet = Union[IControlNetCanny, IControlNetA, IControlNetHandsAndFace]


class IControlNetWithUUID:
    def __init__(
        self,
        preprocessor: EPreProcessor,
        weight: float,
        start_step: int,
        end_step: int,
        guide_image_uuid: str,
        guide_image_unprocessed: Union[str, bytes],
        control_mode: EControlMode,
    ):
        self.preprocessor = preprocessor
        self.weight = weight
        self.start_step = start_step
        self.end_step = end_step
        self.guide_image_uuid = guide_image_uuid
        self.guide_image_unprocessed = guide_image_unprocessed
        self.control_mode = control_mode


class IError:
    def __init__(self, error: bool, error_message: str, task_uuid: str):
        self.error = error
        self.error_message = error_message
        self.task_uuid = task_uuid


class File:
    def __init__(self, data: bytes):
        self.data = data


class IRequestImage:
    def __init__(
        self,
        positive_prompt: str,
        image_size: int,
        model_id: int,
        number_of_images: Optional[int] = None,  # default to 1
        negative_prompt: Optional[str] = None,
        use_cache: Optional[bool] = None,
        lora: Optional[List[ILora]] = None,
        control_net: Optional[List[IControlNet]] = None,
        image_initiator: Optional[Union[File, str]] = None,
        image_mask_initiator: Optional[Union[File, str]] = None,
        steps: Optional[int] = None,
        on_partial_images: Optional[
            Callable[[List[IImage], Optional[IError]], None]
        ] = None,
        seed: Optional[int] = None,
    ):
        self.positive_prompt = positive_prompt
        self.image_size = image_size
        self.model_id = model_id
        self.number_of_images = number_of_images
        self.negative_prompt = negative_prompt
        self.use_cache = use_cache
        self.lora = lora
        self.control_net = control_net
        self.image_initiator = image_initiator
        self.image_mask_initiator = image_mask_initiator
        self.steps = steps
        self.on_partial_images = on_partial_images
        self.seed = seed


class IRequestImageToText:
    def __init__(self, image_initiator: Optional[Union[File, str]] = None):
        self.image_initiator = image_initiator


class IImageToText:
    def __init__(self, task_uuid: str, text: str):
        self.task_uuid = task_uuid
        self.text = text


class IRemoveImageBackground(IRequestImageToText):
    pass


class IPromptEnhancer:
    def __init__(
        self,
        prompt_max_length: Optional[int] = None,
        prompt_language_id: Optional[int] = None,
        prompt_versions: Optional[int] = None,
        prompt: str = "",
    ):
        self.prompt_max_length = prompt_max_length
        self.prompt_language_id = prompt_language_id
        self.prompt_versions = prompt_versions
        self.prompt = prompt


class IEnhancedPrompt(IImageToText):
    pass


class IUpscaleGan:
    def __init__(
        self,
        image_initiator: Union[File, str],
        upscale_factor: int,
        is_image_uuid: Optional[bool] = None,
    ):
        self.image_initiator = image_initiator
        self.upscale_factor = upscale_factor
        self.is_image_uuid = is_image_uuid


class ReconnectingWebsocketProps:
    def __init__(self, websocket: Any):
        self.websocket = websocket

    def add_event_listener(self, event_type: str, listener: Callable, options: Any):
        self.websocket.addEventListener(event_type, listener, options)

    def send(self, data: Any):
        self.websocket.send(data)

    def __getattr__(self, name: str):
        return getattr(self.websocket, name)


class UploadImageType:
    def __init__(self, new_image_uuid: str, new_image_src: str, task_uuid: str):
        self.new_image_uuid = new_image_uuid
        self.new_image_src = new_image_src
        self.task_uuid = task_uuid


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
        self, key: str, listener: Callable[[Any], None], group_key: Optional[str] = None
    ):
        self.key = key
        self.listener = listener
        self.group_key = group_key


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
