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

import abc
import os
import tempfile
from multiprocessing import Lock
from pprint import pformat
from typing import Any

import attr
import cv2
import numpy as np

from geti_sdk.http_session import GetiSession

from .enums import MediaType
from .media_identifiers import (
    ImageIdentifier,
    MediaIdentifier,
    VideoFrameIdentifier,
    VideoIdentifier,
)
from .task_annotation_state import TaskAnnotationState
from .utils import (
    attr_value_serializer,
    numpy_from_buffer,
    str_to_datetime,
    str_to_media_type,
)


@attr.define
class MediaInformation:
    """
    Basic information about a media item in Intel® Geti™.

    :var display_url: URL that can be used to download the full size media entity
    :var height: Height of the media entity, in pixels
    :var width: Width of the media entity, in pixels
    :var size: Size of the media entity, in bytes
    """

    display_url: str
    height: int
    width: int
    extension: str | None = None  # Added in Geti v2.5
    size: int | None = None  # Added in Geti v1.2


@attr.define
class VideoInformation(MediaInformation):
    """
    Basic information about a video entity in Intel® Geti™.

    :var duration: Duration of the video
    :var frame_count: Total number of frames in the video
    :var frame_stride: Frame stride of the video
    :var frame_rate: Frame rate of the video
    """

    duration: int = attr.ib(kw_only=True)
    frame_count: int = attr.ib(kw_only=True)
    frame_stride: int = attr.ib(kw_only=True)
    frame_rate: float | None = attr.ib(kw_only=True, default=None)  # Added in Geti v1.1


@attr.define
class ImageInformation(MediaInformation):
    """
    Basic information about an image entity in Intel® Geti™
    """

    pass


@attr.define
class VideoFrameInformation(MediaInformation):
    """
    Basic information about a video frame in Intel® Geti™.
    """

    frame_index: int = attr.ib(kw_only=True)
    video_id: str = attr.ib(kw_only=True)


@attr.define
class MediaPreprocessing:
    """
    Basic information about a media preprocessing in Intel® Geti™.
    """

    status: str = attr.ib(kw_only=True)
    message: str | None = attr.ib(kw_only=True, default=None)


@attr.define
class MediaItem:
    """
    Representation of a media entity in Intel® Geti™.

    :var id: Unique database ID of the media entity
    :var name: Filename of the media entity
    :var type: MediaType of the entity
    :var upload_time: Time and date at which the entity was uploaded to the system
    :var thumbnail: URL that can be used to get a thumbnail for the media entity
    :var annotation_state_per_task: Annotation state of the media entity
    :var media_information: Container holding basic information such as width and
        height about the media entity
    :var preprocessing: Container holding basic information about media preprocessing
    :param last_annotator_id: the name or id of the editor.
    """

    id: str
    name: str
    type: str = attr.field(converter=str_to_media_type)
    upload_time: str = attr.field(converter=str_to_datetime)
    media_information: MediaInformation
    preprocessing: MediaPreprocessing | None = None  # preprocessing was added in Geti 2.10
    annotation_state_per_task: list[TaskAnnotationState] | None = None
    thumbnail: str | None = None
    uploader_id: str | None = None
    # last_annotator_id was added in Geti v 2.0. It replaced `editor_name`
    last_annotator_id: str | None = None

    @property
    def download_url(self) -> str:
        """
        Return the URL that can be used to download the full size media entity from the
        Intel® Geti™ server.

        :return: URL at which the media entity can be downloaded
        """
        return self.media_information.display_url.strip()

    @property
    def base_url(self) -> str:
        """
        Return the base URL for the media item, which is the URL pointing to the
        media details of the entity.

        :return: Base URL of the media entity
        """
        display_url_image = "/display/full"
        display_url_video = "/display/stream"
        if self.download_url.endswith(display_url_image):
            url = self.download_url[: -len(display_url_image)]
        elif self.download_url.endswith(display_url_video):
            url = self.download_url[: -len(display_url_video)]
        elif self.download_url.endswith(self.id):
            url = self.download_url
        else:
            raise ValueError(
                f"Unexpected end pattern found in display URL for media entity {self}. "
                f"Unable to construct base_url for entity"
            )
        return url

    @property
    def identifier(self) -> MediaIdentifier:
        """
        Return the media identifier for the media item.

        :return: MediaIdentifier which uniquely identifies the media item.
        """
        return MediaIdentifier(type=self.type)

    @abc.abstractmethod
    def get_data(self, session: GetiSession) -> np.ndarray | str:
        """
        Get the pixel data for this MediaItem. Uses caching.

        For Image and VideoFrames, this method will return a np.ndarray. For Video
        objects, this method will return a path-like string, pointing to the video
        on the local disk.

        :param session: REST session to the Intel® Geti™ server on which the MediaItem
            lives
        :raises ValueError: If the cache is empty and no data can be downloaded
            from the cluster
        :return: numpy array holding the pixel data for the MediaItem, or string
            pointing to the file location for Videos.
        """
        raise NotImplementedError

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the MediaItem to a dictionary representation.

        :return: Dictionary holding the annotation scene data
        """
        return attr.asdict(self, value_serializer=attr_value_serializer)

    @property
    def overview(self) -> str:
        """
        Return a string holding an overview of the media item.

        :return: overview string of the media item
        """
        return pformat(self.to_dict())


@attr.define(slots=False)
class Image(MediaItem):
    """
    Representation of an image in Intel® Geti™.

    :var media_information: Container holding basic information such as width and
            height about the image entity
    """

    media_information: ImageInformation = attr.field(kw_only=True)
    annotation_scene_id: str | None = attr.field(kw_only=True, default=None, repr=False)  # Added in Geti v 1.16
    roi_id: str | None = attr.field(kw_only=True, default=None, repr=False)  # Added in Geti v 1.16

    def __attrs_post_init__(self):
        """
        Initialize private attributes.
        """
        self._data: np.ndarray | None = None

    @property
    def identifier(self) -> ImageIdentifier:
        """
        Return the media identifier for the Image instance

        :return: ImageIdentifier object that contains the identifiers of the image
        """
        return ImageIdentifier(image_id=self.id, type=self.type)

    def get_data(self, session: GetiSession) -> np.ndarray:
        """
        Get the pixel data for this Image. This method uses caching: If the cache is
        empty, it will download the data using the provided session. Otherwise it
        will return the cached data directly

        NOTE: The pixel data will be returned in BGR channel order

        :param session: REST session to the Intel® Geti™ server on which the Image lives
        :raises ValueError: If the cache is empty and no data can be downloaded
            from the cluster
        :return: Numpy.ndarray holding the pixel data for this Image.
        """
        if self._data is None:
            response = session.get_rest_response(url=self.download_url, method="GET", contenttype="jpeg")
            if response.status_code == 200:
                self._data = numpy_from_buffer(response.content)
            else:
                raise ValueError(
                    f"Unable to retrieve data for image {self}, received response {response} from Intel® Geti™ server."
                )
        return self._data

    @property
    def numpy(self) -> np.ndarray | None:
        """
        Pixel data for the Image, as a numpy array of shape (width x heigth x 3).
        If this attribute is None, the pixel data should be downloaded from the
        cluster first using the `get_data` method

        :return: numpy.ndarray containing the pixel data
        """
        return self._data


@attr.define(slots=False)
class VideoAnnotationStatistics:
    """
    Annotation statistics for a video in Intel® Geti™.

    :var annotated: Number of frames in the video that have been fully annotated
    :var partially_annotated: Number of frames in the video that have been partially
        annotated, meaning that one task in the task chain is still missing annotations
    :var unannotated: Number of frames in the video that have not yet been annotated
    """

    annotated: int
    partially_annotated: int
    unannotated: int


@attr.define(slots=False)
class Video(MediaItem):
    """
    Representation of a video in Intel® Geti™.

    :var media_information: Container holding basic information such as width,
            height and duration about the video entity
    """

    media_information: VideoInformation = attr.field(kw_only=True)
    matched_frames: int | None = attr.field(kw_only=True, default=None, repr=False)
    annotation_statistics: VideoAnnotationStatistics | None = attr.field(kw_only=True, default=None, repr=False)

    def __attrs_post_init__(self):
        """
        Initialize private attributes.
        """
        self._data: str | None = None
        self._needs_tempfile_deletion: bool = False
        self._data_lock: Lock = Lock()

    @property
    def identifier(self) -> VideoIdentifier:
        """
        Return the media identifier for the Video instance.

        :return: VideoIdentifier object that contains the identifiers of the video
        """
        return VideoIdentifier(video_id=self.id, type=self.type)

    def get_data(self, session: GetiSession) -> str:
        """
        Get the video data from the Intel® Geti™ server. Calling this method will
        download the video data to a temporary file, and returns the path to file.

        :param session: GetiSession object pointing to the instance from which the video
            should be downloaded.
        :return: Path to the temporary video file on local disk.
        """
        if self._data is None:
            response = session.get_rest_response(url=self.download_url, method="GET", contenttype="jpeg")
            if response.status_code == 200:
                file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")  # noqa: SIM115
                file.write(response.content)
                file.close()
                self._data = file.name
                self._needs_tempfile_deletion = True
            else:
                raise ValueError(
                    f"Unable to retrieve data for image {self}, received response {response} from Intel® Geti™ server."
                )
        return self._data

    def to_frames(self, frame_stride: int | None = None, include_data: bool = False) -> list["VideoFrame"]:
        """
        Extract VideoFrames from the Video. Returns a list of VideoFrame objects.

        :param frame_stride: Stride to use for frame extraction. If left as None (the
            default), the frame_stride stored in the media_information is used.
        :param include_data: True to include pixel data for the frames, if available.
        :return: List of VideoFrames constructed from the Video
        """
        if frame_stride is None:
            frame_stride = self.media_information.frame_stride
        if frame_stride < 0:
            raise ValueError(f"Invalid frame stride {frame_stride}. Frame stride cannot be negative.")
        if frame_stride > self.media_information.frame_count:
            frame_stride = self.media_information.frame_count - 1
        frames_range = range(0, self.media_information.frame_count, frame_stride)
        if not include_data:
            frames = [VideoFrame.from_video(self, frame_index=index) for index in frames_range]
        else:
            frames = [self.get_frame(frame_index=index) for index in frames_range]
        return frames

    def get_frame(self, frame_index: int) -> "VideoFrame":
        """
        Return a VideoFrame extracted from the Video, at the specified index.

        This method loads the numpy data for the frame, if available.

        :param frame_index: Index of the frame to extract
        :return: VideoFrame at the specified index
        """
        video_frame = VideoFrame.from_video(video=self, frame_index=frame_index)
        if self._data is not None:
            video_capture = cv2.VideoCapture(self._data)
            video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            success, frame = video_capture.read()
            if success:
                video_frame._data = frame
        return video_frame

    def __del__(self):
        """
        Clean up the temporary file created to store the video data (if any). This
        method is called when the Video object is deleted.
        """
        with self._data_lock:
            if self._needs_tempfile_deletion and os.path.exists(self._data) and os.path.isfile(self._data):
                os.remove(self._data)


@attr.define(slots=False)
class VideoFrame(MediaItem):
    """
    Representation of a video frame in Intel® Geti™.

    :var media_information: Container holding basic information such as width and
            height about the VideoFrame entity
    :var data: Pixel data for the VideoFrame. If this is None, the data can be
        downloaded using the 'get_data' method
    """

    media_information: VideoFrameInformation = attr.field(kw_only=True)
    video_name: str | None = attr.field(kw_only=True, default=None)

    def __attrs_post_init__(self):
        """
        Initialize private attributes.
        """
        self._data: np.ndarray | None = None

    @classmethod
    def from_video(cls, video: Video, frame_index: int) -> "VideoFrame":
        """
        Create a VideoFrame entity from a `video` and a `frame_index`.

        :param video: Video to extract the VideoFrame from
        :param frame_index: index at which the frame lives in the video
        :return:
        """
        base_url = f"{video.base_url}/frames/{frame_index}"
        frame_information = VideoFrameInformation(
            frame_index=frame_index,
            width=video.media_information.width,
            height=video.media_information.height,
            video_id=video.id,
            display_url=f"{base_url}/display/full",
        )
        return VideoFrame(
            name=f"{video.name}_frame_{frame_index}",
            type=str(MediaType.VIDEO_FRAME),
            upload_time=video.upload_time,
            thumbnail=f"{base_url}/display/thumb",
            media_information=frame_information,
            preprocessing=video.preprocessing,
            annotation_state_per_task=video.annotation_state_per_task,
            id=video.id,
            video_name=video.name,
        )

    @property
    def identifier(self) -> VideoFrameIdentifier:
        """
        Return the media identifier for the VideoFrame instance.

        :return: VideoFrameIdentifier object that contains the identifiers of the
            video frame
        """
        return VideoFrameIdentifier(
            video_id=self.id,
            type=self.type,
            frame_index=self.media_information.frame_index,
        )

    @property
    def numpy(self) -> np.ndarray | None:
        """
        Pixel data for the Image, as a numpy array of shape (width x heigth x 3).
        If this attribute is None, the pixel data should be downloaded from the
        cluster first using the `get_data` method

        :return: numpy.ndarray containing the pixel data
        """
        return self._data

    def get_data(self, session: GetiSession) -> np.ndarray:
        """
        Get the pixel data for this VideoFrame. This method uses caching: If the cache
        is empty, it will download the data using the provided session. Otherwise it
        will return the cached data directly

        :param session: REST session to the Intel® Geti™ server on which the
            VideoFrame lives
        :raises ValueError: If the cache is empty and no data can be downloaded
            from the cluster
        :return: Numpy.ndarray holding the pixel data for this VideoFrame.
        """
        if self._data is None:
            response = session.get_rest_response(url=self.download_url, method="GET", contenttype="jpeg")
            if response.status_code == 200:
                self._data = numpy_from_buffer(response.content)
            else:
                raise ValueError(
                    f"Unable to retrieve data for image {self}, received response {response} from Intel® Geti™ server."
                )
        return self._data
