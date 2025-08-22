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
import os

from geti_sdk.demos.constants import DEFAULT_DATA_PATH

from .download_helpers import download_file, validate_hash

VIDEO_PERSON_CAR_BIKE_PATH = os.path.join(DEFAULT_DATA_PATH, "person-bicycle-car-detection.mp4")


def get_person_car_bike_video(
    video_path: str | os.PathLike | None = None,
) -> str:
    """
    Get the path to the 'person-bicycle-car-detection.mp4' video file that is used for
    the notebook demos in the Geti SDK

    :param video_path: Optional file path to video. Only specify this if you have already
        downloaded the video to a different path than the default data path in the SDK.
    :return: Path to the video
    """
    if video_path is None:
        video_path = VIDEO_PERSON_CAR_BIKE_PATH
    if os.path.isfile(video_path):
        video_file_path = video_path
    else:
        video_file_path = download_file(
            url="https://storage.openvinotoolkit.org/data/test_data/videos/person-bicycle-car-detection.mp4",
            target_folder=os.path.dirname(video_path),
        )

    # Compare hashes
    expected_hash = "7eafb94b3491e7554d92637596ddaaf1c69fdb8421c3d49eb09df4b7d05788c76391cb791cc2c8197e66d352306328d09c8d72e42733b1f1a311cd1e35ee1cce"  # noqa: E501
    validate_hash(file_path=video_file_path, expected_hash=expected_hash)
    return video_file_path
