# Copyright (C) 2022 Intel Corporation
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

import cv2
import numpy as np
import pytest

from geti_sdk.data_models import Image, MediaType
from geti_sdk.data_models.containers import MediaList
from geti_sdk.data_models.media import (
    ImageInformation,
    MediaPreprocessing,
    Video,
    VideoFrame,
    VideoInformation,
)
from geti_sdk.data_models.media_identifiers import ImageIdentifier, VideoIdentifier
from geti_sdk.demos import EXAMPLE_IMAGE_PATH


@pytest.fixture(scope="session")
def fxt_numpy_image() -> np.ndarray:
    yield cv2.imread(EXAMPLE_IMAGE_PATH)


@pytest.fixture()
def fxt_image_identifier() -> ImageIdentifier:
    yield ImageIdentifier(image_id="image_0", type=MediaType.IMAGE)


@pytest.fixture()
def fxt_video_identifier() -> VideoIdentifier:
    yield VideoIdentifier(video_id="video_0", type=MediaType.VIDEO)


@pytest.fixture()
def fxt_image_identifier_rest() -> dict[str, str]:
    yield {"image_id": "image_0", "type": "image"}


@pytest.fixture()
def fxt_image_information(fxt_numpy_image: np.ndarray) -> ImageInformation:
    yield ImageInformation(
        display_url="dummy_url",
        height=fxt_numpy_image.shape[0],
        width=fxt_numpy_image.shape[1],
    )


@pytest.fixture(scope="session")
def fxt_video_information(fxt_video_path_1_light_bulbs: str) -> VideoInformation:
    cap = cv2.VideoCapture(fxt_video_path_1_light_bulbs)
    if cap.isOpened():
        width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        duration = count / fps

    yield VideoInformation(
        display_url="dummy_url/display/stream",
        width=int(width),
        height=int(height),
        frame_count=int(count),
        duration=duration,
        frame_stride=50,
    )


@pytest.fixture()
def fxt_preprocessing() -> MediaPreprocessing:
    yield MediaPreprocessing(status="FINISHED")


@pytest.fixture()
def fxt_geti_image(
    fxt_numpy_image: np.ndarray,
    fxt_image_information: ImageInformation,
    fxt_image_identifier: ImageIdentifier,
    fxt_preprocessing: MediaPreprocessing,
    fxt_datetime_string: str,
) -> Image:
    image = Image(
        name="dummy_image",
        id=fxt_image_identifier.image_id,
        type=fxt_image_identifier.type,
        media_information=fxt_image_information,
        preprocessing=fxt_preprocessing,
        upload_time=fxt_datetime_string,
    )
    image._data = fxt_numpy_image
    yield image


@pytest.fixture()
def fxt_geti_video(
    fxt_video_path_1_light_bulbs: str,
    fxt_video_identifier: VideoIdentifier,
    fxt_video_information: VideoInformation,
    fxt_preprocessing: MediaPreprocessing,
    fxt_datetime_string: str,
) -> Video:
    video = Video(
        name="dummy_video",
        id=fxt_video_identifier.video_id,
        type=fxt_video_identifier.type,
        upload_time=fxt_datetime_string,
        media_information=fxt_video_information,
        preprocessing=fxt_preprocessing,
    )
    video._data = fxt_video_path_1_light_bulbs
    yield video


@pytest.fixture()
def fxt_video_frames(fxt_geti_video: Video) -> MediaList[VideoFrame]:
    yield fxt_geti_video.to_frames(include_data=True)
