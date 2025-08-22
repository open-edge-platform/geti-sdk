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

from typing import ClassVar

import attr

from geti_sdk.data_models.utils import attr_value_serializer, str_to_media_type


@attr.define
class MediaIdentifier:
    """
    Representation of media identification data as output by the Intel® Geti™
    /annotations REST endpoints.

    :var type: Type of the media to which the annotation belongs
    """

    type: str = attr.field(converter=str_to_media_type)

    def to_dict(self) -> dict[str, str]:
        """
        Return a dictionary form of the MediaIdentifier instance.

        :return: Dictionary containing the media identifier data
        """
        return attr.asdict(self, value_serializer=attr_value_serializer)


@attr.define
class ImageIdentifier(MediaIdentifier):
    """
    Representation of image identification data used by the Intel® Geti™ /annotations
    endpoints. This object uniquely identifies an Image on the Intel® Geti™ server.

    :var image_id: unique database ID of the image
    """

    _identifier_fields: ClassVar[str] = ["image_id"]

    image_id: str


@attr.define
class VideoFrameIdentifier(MediaIdentifier):
    """
    Representation of video frame identification data used by the Intel® Geti™
    /annotations endpoints. This object uniquely identifies a VideoFrame on the
    Intel® Geti™ server.

    :var frame_index: Index of the video frame in the full video
    :var video_id: unique database ID of the video to which the frame belongs
    :var key_index: Index of the key frame in the video
    """

    _identifier_fields: ClassVar[str] = ["video_id"]

    frame_index: int
    video_id: str
    key_index: int | None = None  # Added in Geti 2.5


@attr.define
class VideoIdentifier(MediaIdentifier):
    """
    Representation of video identification data used by the Intel® Geti™ /annotations
    endpoints. This object uniquely identifiers a Video on the Intel® Geti™ server.

    :var video_id: unique database ID of the video
    """

    _identifier_fields: ClassVar[str] = ["video_id"]

    video_id: str
