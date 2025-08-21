# Copyright (C) 2023 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions
# and limitations under the License.
import logging
import os
import platform
import sys
from collections.abc import Sequence
from contextlib import contextmanager
from logging.handlers import MemoryHandler

import cv2
import numpy as np
from openvino import get_version

import geti_sdk
from geti_sdk.data_models import Image, Video
from geti_sdk.geti import DEFAULT_LOG_FORMAT
from geti_sdk.http_session import GetiSession

try:
    # Updated OpenVINO API, from v2023.2.0
    from openvino.core import Core
    from openvino.properties import device as ov_device
except ImportError:
    # The old API
    from openvino import Core
    from openvino.properties import device as ov_device


def get_system_info(device: str = "CPU") -> dict[str, str]:
    """
    Retrieve relevant information about the system

    :param device: Hardware device to retrieve the info for, for example "CPU", "GPU",
        etc. Defaults to "CPU".
    :return: Dictionary containing system information
        - Operating system
        - Processor name
        - Python version
        - Geti SDK version
        - OpenVINO version
    """
    ov_core = Core()
    try:
        device_info = ov_core.get_property(device_name=device, property=ov_device.full_name())
    except RuntimeError as e:
        logging.warning(f"Unable to retrieve device info for device `{device}`. Failed with error: `{e}`")
        device_info = device
    info: dict[str, str] = {}
    info["operating_system"] = platform.system()
    info["device_info"] = device_info
    info["python"] = platform.python_version()
    info["geti-sdk"] = geti_sdk.__version__
    info["openvino"] = get_version()
    return info


def load_benchmark_media(
    session: GetiSession,
    images: Sequence[Image | np.ndarray | os.PathLike] | None = None,
    video: Video | os.PathLike | None = None,
    frames: int = 200,
) -> list[np.ndarray]:
    """
    Load and standardize a list of media. This method will return a list of numpy 2D
    arrays containing the frame data. The list will contain exactly `frames` elements.

    Either `video` or `images` should be specified, but not both.

    If the list of `images` contains less than `frames` elements, the first image in
    the list will be used to pad the list up to a length of `frames`.
    Similarly, if the `video` contains less frames than what is specified in `frames`,
    the first frame of the video will be used as padding.

    :param session: GetiSession pointing to the Geti server that can be used to
        retrieve the media data
    :param images: List of either:
        - Geti Image objects, identifying images on the Geti server
        - numpy 2D arrays, containing the image data
        - PathLike objects, pointing to the filepaths of images to load
    :param video: Either a Geti Video object or a filepath pointing to a video file on
        disk.
        NOTE: Either `video` or `images` should be specified, but not both
    :param frames: Number of frames to return
    :return: List of 2D arrays containing image/frame data. The returned list has a
        length of exactly `frames`.
    """
    if video is not None and images is not None:
        raise ValueError("Please specify either `video` or `images`, but not both.")
    if frames > 1000:
        logging.warning(
            f"You have specified {frames} to be loaded into memory. This may cause "
            f"out of memory problems on systems with constrained resources. Please "
            f"consider reducing the number of frames to load"
        )
    loaded_frames: list[np.ndarray] = []
    if images is not None:
        if len(images) < frames:
            # Fill the list of images up to `frames` using the first image
            images += [images[0]] * (frames - len(images))
        elif len(images) > frames:
            # Select only the first `frames` images
            images = images[0:frames]
        for image_object in images:
            if isinstance(image_object, Image):
                image_np = image_object.get_data(session=session)
                loaded_frames.append(image_np)
            elif isinstance(image_object, np.ndarray):
                loaded_frames.append(image_object)
            elif isinstance(image_object, os.PathLike | str):
                image_cv = cv2.imread(image_object)
                image_cv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
                loaded_frames.append(image_cv)
            else:
                raise TypeError(
                    f"Encountered invalid image object of type {type(image_object)}. "
                    f"Please pass either Geti Image objects, numpy arrays with image "
                    f"data or image filepaths."
                )
    elif video is not None:
        video_data_path: str
        if isinstance(video, Video):
            video_data_path = video.get_data(session=session)
        elif isinstance(video, os.PathLike):
            video_data_path = video
        else:
            raise TypeError(
                f"Encountered invalid video object of type {video.type}. Please "
                f"pass either a Geti Video object or a filepath to a video on disk."
            )
        cap = cv2.VideoCapture(video_data_path)

        if cap is None or not cap.isOpened():
            raise ValueError(f"Unable to read video from {video_data_path}")

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        while cap.isOpened() and len(loaded_frames) < frame_count:
            ret, frame = cap.read()
            if ret is True:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                loaded_frames.append(rgb_frame)
            else:
                break
        cap.release()
        if len(loaded_frames) < frame_count:
            # Pad with first frame if the video was too short
            loaded_frames += [loaded_frames[0]] * (frames - len(loaded_frames))
    return loaded_frames


@contextmanager
def suppress_log_output(
    target_logger: logging.Logger | None = None,
    target_handler: logging.Handler | None = None,
):
    """
    Context manager to temporarily capture and suppress log output.
    All logging calls made in the context will be redirected to memory,
    unless ERROR level message are logged. In that case the output will be sent to
    the `target_handler`.

    :param target_logger: Optional Logger of which the output should be captured
    :param target_handler: Optional Handler that should be used in case of ERROR log
        message
    """
    if target_logger is None:
        target_logger = logging.getLogger()

    if target_handler is None:
        target_handler = logging.StreamHandler(stream=sys.stdout)
    formatter = logging.Formatter(DEFAULT_LOG_FORMAT)
    target_handler.setFormatter(formatter)

    memory_handler = MemoryHandler(capacity=1000, flushLevel=logging.ERROR, target=target_handler)

    original_handlers = target_logger.handlers
    for handler in original_handlers:
        target_logger.removeHandler(handler)
    target_logger.addHandler(memory_handler)

    try:
        yield target_logger
    except Exception:
        target_logger.exception("Error in `suppress_log_output`")
        raise
    finally:
        # Flush the memory handler and reinstate the original log handlers
        super(MemoryHandler, memory_handler).flush()
        target_logger.removeHandler(memory_handler)
        for handler in original_handlers:
            target_logger.addHandler(handler)
