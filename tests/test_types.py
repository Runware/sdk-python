import pytest
import sys
import os

from typing import List, Union, Optional, Callable, Any, Dict
from runware.types import (
    IControlNet,
    IControlNetA,
    IControlNetCanny,
    IControlNetHandsAndFace,
    IImageInference,
    File,
    RequireAtLeastOne,
    RequireOnlyOne,
    ListenerType,
    EPreProcessor,
    EOpenPosePreProcessor,
    EControlMode,
)


def test_icontrol_net_union():
    canny_control_net = IControlNetCanny(
        weight=0.5,
        start_step=0,
        end_step=10,
        guide_image="canny_image.png",
        control_mode=EControlMode.BALANCED,
        low_threshold_canny=100,
        high_threshold_canny=200,
    )
    assert isinstance(canny_control_net, IControlNet)

    hands_face_control_net = IControlNetHandsAndFace(
        preprocessor=EOpenPosePreProcessor.openpose_face,
        weight=0.7,
        start_step=5,
        end_step=15,
        guide_image_unprocessed="hands_face_image.png",
        control_mode=EControlMode.PROMPT,
        include_hands_and_face_open_pose=True,
    )
    assert isinstance(hands_face_control_net, IControlNet)


def test_irequest_image():
    request_image = IImageInference(
        positive_prompt="A beautiful landscape",
        image_size=512,
        model_id=1,
        image_initiator=File(b"image_data"),
        image_mask_initiator="mask.png",
    )
    assert request_image.positive_prompt == "A beautiful landscape"
    assert request_image.image_size == 512
    assert request_image.model_id == 1
    assert isinstance(request_image.image_initiator, File)
    assert request_image.image_mask_initiator == "mask.png"


def test_require_at_least_one():
    data = {"key1": "value1", "key2": "value2", "key3": "value3"}
    obj = RequireAtLeastOne(data, ["key1", "key3"])
    assert obj["key1"] == "value1"
    assert "key2" in obj
    assert len(obj) == 3


def test_require_only_one():
    data = {"key1": "value1"}
    obj = RequireOnlyOne(data, ["key1"])
    assert obj["key1"] == "value1"
    data = {"key1": "value1", "key2": "value2"}
    with pytest.raises(ValueError):
        RequireOnlyOne(data, ["key1", "key2"])


def test_require_at_least_one_missing_keys():
    data = {"key1": "value1"}
    obj = RequireAtLeastOne(data, ["key1", "key2"])
    assert obj["key1"] == "value1"
    assert "key2" not in obj
    assert len(obj) == 1


def test_require_at_least_one_non_string_keys():
    data = {1: "value1", "key2": "value2"}  # One key is an integer
    obj = RequireAtLeastOne(data, [1, "key2"])
    assert obj[1] == "value1"


def test_require_at_least_one_extra_keys():
    data = {"key1": "value1", "key2": "value2", "extra_key": "extra"}
    obj = RequireAtLeastOne(data, ["key1"])
    assert obj["extra_key"] == "extra"
    assert obj["key1"] == "value1"


def test_require_at_least_one_non_dict_data():
    with pytest.raises(TypeError, match="data must be a dictionary"):
        RequireAtLeastOne("some_string", ["key1"])


def test_require_at_least_one_invalid_required_keys():
    data = {"key1": "value1"}
    with pytest.raises(TypeError):
        RequireAtLeastOne(data, 123)  # 123 as an example of a non-iterable, non-string


def test_require_at_least_one_removing_key():
    data = {"key1": "value1", "key2": "value2"}
    obj = RequireAtLeastOne(data, ["key1", "key2"])
    del obj["key1"]
    assert "key2" in obj
    assert len(obj) == 1

    del obj["key2"]
    assert len(obj) == 0

    with pytest.raises(
        ValueError,
        match="At least one of the required keys must be present: key1, key2",
    ):
        RequireAtLeastOne({"extra_key": "extra"}, ["key1", "key2"])


def test_require_only_one_adding_keys():
    data = {"key1": "value1"}
    obj = RequireOnlyOne(data, ["key1"])  # Specify only one required key

    obj["key2"] = "value2"  # Adding a non-required key should be allowed
    assert "key2" in obj
    assert len(obj) == 2

    with pytest.raises(ValueError):
        RequireOnlyOne({"key1": "value1", "key2": "value2"}, ["key1", "key2"])


def test_listener_type():
    def on_message(msg: Any):
        print(msg)

    listener = ListenerType("message_listener", on_message, group_key="group1")
    assert listener.key == "message_listener"
    assert listener.group_key == "group1"
    listener.listener("Hello")  # Prints "Hello"


def test_icontrol_net_creation():
    control_net = IControlNetA(
        preprocessor=EPreProcessor.canny,
        weight=0.5,
        start_step=0,
        end_step=10,
        guide_image="guide_image.png",
        control_mode=EControlMode.BALANCED,
    )
    assert isinstance(control_net, IControlNet)
    assert control_net.preprocessor == EPreProcessor.canny
    assert control_net.weight == 0.5
    assert control_net.start_step == 0
    assert control_net.end_step == 10
    assert control_net.guide_image == "guide_image.png"
    assert control_net.control_mode == EControlMode.BALANCED


def test_icontrol_net_canny_creation():
    control_net_canny = IControlNetCanny(
        weight=0.8,
        start_step=2,
        end_step=8,
        guide_image="canny_guide_image.png",
        control_mode=EControlMode.PROMPT,
        low_threshold_canny=100,
        high_threshold_canny=200,
    )
    assert isinstance(control_net_canny, IControlNetCanny)
    assert isinstance(control_net_canny, IControlNet)
    assert control_net_canny.preprocessor == EPreProcessor.canny
    assert control_net_canny.weight == 0.8
    assert control_net_canny.start_step == 2
    assert control_net_canny.end_step == 8
    assert control_net_canny.guide_image == "canny_guide_image.png"
    assert control_net_canny.control_mode == EControlMode.PROMPT
    assert control_net_canny.low_threshold_canny == 100
    assert control_net_canny.high_threshold_canny == 200


def test_icontrol_net_hands_and_face_creation():
    control_net_hands_and_face = IControlNetHandsAndFace(
        preprocessor=EOpenPosePreProcessor.openpose_face,
        weight=0.6,
        start_step=1,
        end_step=9,
        guide_image_unprocessed="hands_face_guide_image_unprocessed.png",
        control_mode=EControlMode.CONTROL_NET,
        include_hands_and_face_open_pose=True,
    )
    assert isinstance(control_net_hands_and_face, IControlNetHandsAndFace)
    assert isinstance(control_net_hands_and_face, IControlNet)
    assert (
        control_net_hands_and_face.preprocessor == EOpenPosePreProcessor.openpose_face
    )
    assert control_net_hands_and_face.weight == 0.6
    assert control_net_hands_and_face.start_step == 1
    assert control_net_hands_and_face.end_step == 9
    assert (
        control_net_hands_and_face.guide_image_unprocessed
        == "hands_face_guide_image_unprocessed.png"
    )
    assert control_net_hands_and_face.control_mode == EControlMode.CONTROL_NET
    assert control_net_hands_and_face.include_hands_and_face_open_pose == True


def test_icontrol_net_union():
    control_net_canny = IControlNetCanny(
        weight=0.7,
        start_step=3,
        end_step=7,
        guide_image="canny_guide_image.png",
        control_mode=EControlMode.BALANCED,
        low_threshold_canny=150,
        high_threshold_canny=250,
    )
    control_net_hands_and_face = IControlNetHandsAndFace(
        preprocessor=EOpenPosePreProcessor.openpose_full,
        weight=0.9,
        start_step=4,
        end_step=6,
        guide_image_unprocessed="hands_face_guide_image_unprocessed.png",
        control_mode=EControlMode.PROMPT,
        include_hands_and_face_open_pose=False,
    )
    control_nets = [control_net_canny, control_net_hands_and_face]
    for control_net in control_nets:
        assert isinstance(control_net, IControlNet)
