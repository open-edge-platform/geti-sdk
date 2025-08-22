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
import atexit
import os
import tempfile
from collections.abc import Sequence
from datetime import datetime

import cv2
import numpy as np

from geti_sdk.data_models import Dataset, MediaType, Video
from geti_sdk.data_models.containers import MediaList
from geti_sdk.http_session import GetiRequestException
from geti_sdk.rest_converters import MediaRESTConverter

from .media_client import BaseMediaClient


class VideoClient(BaseMediaClient[Video]):
    """
    Class to manage video uploads and downloads for a certain project
    """

    _MEDIA_TYPE = MediaType.VIDEO

    def get_all_videos(self, dataset: Dataset | None = None) -> MediaList[Video]:
        """
        Get the ID's and filenames of all videos in the project, from a specific
        dataset. If no dataset is passed, videos from the training dataset will be
        returned

        :param dataset: Dataset for which to retrieve the videos. If no dataset is
            passed, videos from the training dataset are returned.
        :return: A list containing all Video's in the project
        """
        return self._get_all(dataset=dataset)

    def upload_video(
        self,
        video: np.ndarray | str | os.PathLike,
        dataset: Dataset | None = None,
    ) -> Video:
        """
        Upload a video file to the server. Accepts either a path to a video file, or
        a numpy array containing pixel data for video frames.

        In case a numpy array is passed, this method expects the array to be 4
        dimensional, it's dimensions shaped as: [frames, heigth, width, channels]. The
        framerate of the created video will be set to 1 fps.

        :param video: full path to the video on disk, or numpy array holding the video
            pixel data
        :param dataset: Dataset to which to upload the video. If no dataset is
            passed, the video is uploaded to the training dataset
        :return: Video object representing the uploaded video on the server
        """
        temporary_file_created = False
        if isinstance(video, str | os.PathLike):
            video_path = video
        elif isinstance(video, np.ndarray):
            try:
                n_frames, frame_height, frame_width, channels = video.shape
            except GetiRequestException as error:
                raise ValueError(
                    f"Invalid video input shape, expected a 4D numpy array with "
                    f"dimensions representing [frames, height, width, channels]. Got "
                    f"shape {video.shape}"
                ) from error
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            video_file = tempfile.NamedTemporaryFile(  # noqa: SIM115
                prefix="geti-sdk_temp_video_", suffix=f"_{timestamp}.avi", delete=False
            )
            # Close the file, opencv will open it again from the path
            video_file.close()
            video_path = video_file.name
            out = cv2.VideoWriter(
                video_path,
                cv2.VideoWriter_fourcc("M", "J", "P", "G"),
                1,
                (frame_width, frame_height),
            )
            for frame in video[:, ...]:
                out.write(frame)
            out.release()
            temporary_file_created = True
        else:
            raise TypeError(f"Invalid video type: {type(video)}.")

        video_dict = self._upload(video_path, dataset=dataset)
        uploaded_video = MediaRESTConverter.from_dict(input_dict=video_dict, media_type=Video)
        uploaded_video._data = video_path
        if temporary_file_created:
            uploaded_video._needs_tempfile_deletion = True

            # Register cleanup function on system exit to ensure __del__ gets called
            def clean_temp_video():
                uploaded_video.__del__()

            atexit.register(clean_temp_video)
        return uploaded_video

    def upload_folder(
        self,
        path_to_folder: str,
        n_videos: int = -1,
        skip_if_filename_exists: bool = False,
        dataset: Dataset | None = None,
        max_threads: int = 5,
    ) -> MediaList[Video]:
        """
        Upload all videos in a folder to the project. Returns the mapping of video
        filename to the unique ID assigned by Intel Geti.

        :param path_to_folder: Folder with videos to upload
        :param n_videos: Number of videos to upload from folder
        :param skip_if_filename_exists: Set to True to skip uploading of a video
            if a video with the same filename already exists in the project.
            Defaults to False
        :param dataset: Dataset to which to upload the video. If no dataset is
            passed, the video is uploaded to the training dataset
        :param max_threads: Maximum number of threads to use for downloading. Defaults to 5.
            Set to -1 to use all available threads.
        :return: MediaList containing all video's in the project
        """
        return self._upload_folder(
            path_to_folder=path_to_folder,
            n_media=n_videos,
            skip_if_filename_exists=skip_if_filename_exists,
            dataset=dataset,
            max_threads=max_threads,
        )

    def download_all(
        self,
        path_to_folder: str,
        append_video_uid: bool = False,
        max_threads: int = 10,
        dataset: Dataset | None = None,
    ) -> None:
        """
        Download all videos in a project to a folder on the local disk.

        :param path_to_folder: path to the folder in which the videos should be saved
        :param append_video_uid: True to append the UID of a video to the
            filename (separated from the original filename by an underscore, i.e.
            '{filename}_{video_id}'). If there are videos in the project with
            duplicate filename, this must be set to True to ensure all videos are
            downloaded. Otherwise videos with the same name will be skipped.
        :param max_threads: Maximum number of threads to use for downloading. Defaults to 10.
            Set to -1 to use all available threads.
        :param dataset: Dataset from which to download the videos. If no dataset is
            passed, videos from all datasets are downloaded.
        """
        self._download_all(
            path_to_folder,
            append_media_uid=append_video_uid,
            max_threads=max_threads,
            dataset=dataset,
        )

    def delete_videos(self, videos: Sequence[Video]) -> bool:
        """
        Delete all Video entities in `videos` from the project.

        :param videos: List of Video entities to delete
        :return: True if all videos on the list were deleted successfully,
            False otherwise
        """
        return self._delete_media(media_list=videos)
