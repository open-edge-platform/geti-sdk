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
import logging
from collections.abc import Sequence
from typing import Generic

from requests import Response

from geti_sdk.data_models import AnnotationScene, Image, Video, VideoFrame
from geti_sdk.data_models.containers import MediaList

from .base_annotation_client import AnnotationReaderType, BaseAnnotationClient


class AnnotationClient(BaseAnnotationClient, Generic[AnnotationReaderType]):
    """
    Class to up- or download annotations for images or videos to an existing project.
    """

    def get_latest_annotations_for_video(self, video: Video) -> list[AnnotationScene]:
        """
        Retrieve all latest annotations for a video from the cluster.

        If the video does not have any annotations yet, this method returns an
        empty list

        :param video: Video to get the annotations for
        :return: List of AnnotationScene's, each entry corresponds to an
            AnnotationScene for a single frame in the video
        """
        response = self.session.get_rest_response(
            url=f"{video.base_url}/annotations/latest",
            method="GET",
            include_organization_id=False,
        )
        if type(response) is Response and response.status_code == 204:
            return []
        annotations = response["video_annotations"]
        annotation_scenes = [
            self.annotation_scene_from_rest_response(annotation_scene, media_information=video.media_information)
            for annotation_scene in annotations
            if annotation_scene["annotations"]
        ]
        for scene in annotation_scenes:
            scene.resolve_label_names_and_colors(labels=self._project.get_all_labels())
        return annotation_scenes

    def upload_annotations_for_video(self, video: Video, append_annotations: bool = False, max_threads: int = 5):
        """
        Upload annotations for a video. If append_annotations is set to True,
        annotations will be appended to the existing annotations for the video in the
        project. If set to False, existing annotations will be overwritten.

        :param video: Video to upload annotations for
        :param append_annotations: True to append annotations from the local disk to
            the existing annotations on the server, False to overwrite the server
            annotations by those on the local disk.
        :param max_threads: Maximum number of threads to use for uploading. Defaults to 5.
            Set to -1 to use all available threads.
        :return:
        """
        annotation_filenames = self.annotation_reader.get_data_filenames()
        video_annotation_names = [
            filename for filename in annotation_filenames if filename.startswith(f"{video.name}_frame_")
        ]
        frame_indices = [int(name.split("_")[-1]) for name in video_annotation_names]
        video_frames = MediaList(
            [VideoFrame.from_video(video=video, frame_index=frame_index) for frame_index in frame_indices]
        )
        return self._upload_annotations_for_2d_media_list(
            media_list=video_frames,
            append_annotations=append_annotations,
            max_threads=max_threads,
        )

    def upload_annotations_for_videos(
        self,
        videos: Sequence[Video],
        append_annotations: bool = False,
        max_threads: int = 5,
    ):
        """
        Upload annotations for a list of videos. If append_annotations is set to True,
        annotations will be appended to the existing annotations for the video in the
        project. If set to False, existing annotations will be overwritten.

        :param videos: List of videos to upload annotations for
        :param append_annotations: True to append annotations from the local disk to
            the existing annotations on the server, False to overwrite the server
            annotations by those on the local disk.
        :param max_threads: Maximum number of threads to use for uploading. Defaults to 5.
            Set to -1 to use all available threads.
        :return:
        """
        logging.info("Starting video annotation upload...")
        upload_count = 0
        for video in videos:
            upload_count += self.upload_annotations_for_video(
                video=video,
                append_annotations=append_annotations,
                max_threads=max_threads,
            )
        if upload_count > 0:
            logging.info(f"Upload complete. Uploaded {upload_count} new video frame annotations")
        else:
            logging.info("No new video frame annotations were found.")

    def upload_annotations_for_images(
        self,
        images: Sequence[Image],
        append_annotations: bool = False,
        max_threads: int = 5,
    ):
        """
        Upload annotations for a list of images. If append_annotations is set to True,
        annotations will be appended to the existing annotations for the image in the
        project. If set to False, existing annotations will be overwritten.

        :param images: List of images to upload annotations for
        :param append_annotations: True to append annotations from the local disk to
            the existing annotations on the server, False to overwrite the server
            annotations by those on the local disk.
        :param max_threads: Maximum number of threads to use for uploading. Defaults to 5.
            Set to -1 to use all available threads.
        :return:
        """
        logging.info("Starting image annotation upload...")
        upload_count = self._upload_annotations_for_2d_media_list(
            media_list=images,
            append_annotations=append_annotations,
            max_threads=max_threads,
        )
        if upload_count > 0:
            logging.info(f"Upload complete. Uploaded {upload_count} new image annotations")
        else:
            logging.info("No new image annotations were found.")

    def download_annotations_for_video(
        self,
        video: Video,
        path_to_folder: str,
        append_video_uid: bool = False,
        max_threads: int = 10,
    ) -> float:
        """
        Download video annotations from the server to a target folder on disk.

        :param video: Video for which to download the annotations
        :param path_to_folder: Folder to save the annotations to
        :param append_video_uid: True to append the UID of the video to the
             annotation filename (separated from the original filename by an underscore,
             i.e. '{filename}_{media_id}'). This can be useful if the project contains
             videos with duplicate filenames. If left as False, the video filename and
             frame index for the annotation are used as filename for the downloaded
             annotation.
        :param max_threads: Maximum number of threads to use for downloading. Defaults to 10.
            Set to -1 to use all available threads.
        :return: Returns the time elapsed to download the annotations, in seconds
        """
        annotations = self.get_latest_annotations_for_video(video=video)
        frame_list = MediaList[VideoFrame](
            [
                VideoFrame.from_video(video=video, frame_index=annotation.media_identifier.frame_index)
                for annotation in annotations
            ]
        )
        if len(frame_list) > 0:
            return self._download_annotations_for_2d_media_list(
                media_list=frame_list,
                path_to_folder=path_to_folder,
                verbose=False,
                append_media_uid=append_video_uid,
                max_threads=max_threads,
            )
        return 0

    def download_annotations_for_images(
        self,
        images: MediaList[Image],
        path_to_folder: str,
        append_image_uid: bool = False,
        max_threads: int = 10,
    ) -> float:
        """
        Download image annotations from the server to a target folder on disk.

        :param images: List of images for which to download the annotations
        :param path_to_folder: Folder to save the annotations to
        :param append_image_uid: True to append the UID of the image to the
            annotation filename (separated from the original filename by an underscore,
             i.e. '{filename}_{media_id}'). This can be useful if the project contains
             images with duplicate filenames. If left as False, the image filename is
             used as filename for the downloaded annotation as well.
        :param max_threads: Maximum number of threads to use for downloading. Defaults to 10.
            Set to -1 to use all available threads.
        :return: Returns the time elapsed to download the annotations, in seconds
        """
        return self._download_annotations_for_2d_media_list(
            media_list=images,
            path_to_folder=path_to_folder,
            append_media_uid=append_image_uid,
            verbose=False,
            max_threads=max_threads,
        )

    def download_annotations_for_videos(
        self,
        videos: MediaList[Video],
        path_to_folder: str,
        append_video_uid: bool = False,
        max_threads: int = 10,
    ) -> float:
        """
        Download annotations for a list of videos from the server to a target folder
        on disk.

        :param videos: List of videos for which to download the annotations
        :param path_to_folder: Folder to save the annotations to
        :param append_video_uid: True to append the UID of the video to the
             annotation filename (separated from the original filename by an underscore,
             i.e. '{filename}_{media_id}'). This can be useful if the project contains
             videos with duplicate filenames. If left as False, the video filename and
             frame index for the annotation are used as filename for the downloaded
             annotation.
        :param max_threads: Maximum number of threads to use for downloading. Defaults to 10.
            Set to -1 to use all available threads.
        :return: Time elapsed to download the annotations, in seconds
        """
        t_total = 0
        logging.info(
            f"Starting annotation download... saving annotations for "
            f"{len(videos)} videos to folder {path_to_folder}/annotations"
        )
        for video in videos:
            t_total += self.download_annotations_for_video(
                video=video,
                path_to_folder=path_to_folder,
                append_video_uid=append_video_uid,
                max_threads=max_threads,
            )
        logging.info(f"Video annotation download finished in {t_total:.1f} seconds.")
        return t_total

    def download_all_annotations(self, path_to_folder: str, max_threads: int = 10) -> None:
        """
        Download all annotations for the project to a target folder on disk.

        :param path_to_folder: Folder to save the annotations to
        :param max_threads: Maximum number of threads to use for downloading. Defaults to 10.
            Set to -1 to use all available threads.
        """
        image_list = self._get_all_media_by_type(media_type=Image)
        video_list = self._get_all_media_by_type(media_type=Video)
        if len(image_list) > 0:
            self.download_annotations_for_images(
                images=image_list,
                path_to_folder=path_to_folder,
                append_image_uid=image_list.has_duplicate_filenames,
                max_threads=max_threads,
            )
        if len(video_list) > 0:
            self.download_annotations_for_videos(
                video_list,
                path_to_folder=path_to_folder,
                append_video_uid=video_list.has_duplicate_filenames,
                max_threads=max_threads,
            )

    def upload_annotations_for_all_media(self, append_annotations: bool = False, max_threads: int = 5):
        """
        Upload annotations for all media in the project, If append_annotations is set
        to True, annotations will be appended to the existing annotations for the
        media on the server. If set to False, existing annotations will be overwritten.

        :param append_annotations: True to append annotations from the local disk to
            the existing annotations on the server, False to overwrite the server
            annotations by those on the local disk. Defaults to False.
        :param max_threads: Maximum number of threads to use for uploading. Defaults to 5.
            Set to -1 to use all available threads.
        """
        image_list = self._get_all_media_by_type(media_type=Image)
        video_list = self._get_all_media_by_type(media_type=Video)
        if len(image_list) > 0:
            self.upload_annotations_for_images(
                images=image_list,
                append_annotations=append_annotations,
                max_threads=max_threads,
            )
        if len(video_list) > 0:
            self.upload_annotations_for_videos(
                videos=video_list,
                append_annotations=append_annotations,
                max_threads=max_threads,
            )

    def upload_annotation(self, media_item: Image | VideoFrame, annotation_scene: AnnotationScene) -> AnnotationScene:
        """
        Upload an annotation for an image or video frame to the Intel® Geti™ server.

        :param media_item: Image or VideoFrame to apply and upload the annotation to
        :param annotation_scene: AnnotationScene to upload
        :return: The uploaded annotation
        """
        if not isinstance(media_item, Image | VideoFrame):
            raise ValueError(
                f"Cannot upload annotation for media item {media_item.name}. This "
                f"method only supports uploading annotations for single images and "
                f"video frames. Please use the method `upload_annotations_for_video` "
                f"to upload video annotations"
            )
        return self._upload_annotation_for_2d_media_item(media_item=media_item, annotation_scene=annotation_scene)

    def get_annotation(self, media_item: Image | VideoFrame) -> AnnotationScene | None:
        """
        Retrieve the latest annotations for an image or video frame from the
        Intel® Geti™ platform.
        If no annotation is available, this method returns None.

        :param media_item: Image or VideoFrame to retrieve the annotations for
        :return: AnnotationScene instance containing the latest annotation data
        """
        if not isinstance(media_item, Image | VideoFrame):
            raise ValueError(
                f"Cannot get annotation for media item {media_item.name}. This method "
                f"only supports getting annotations for images and video frames. "
                f"Please use the method `get_latest_annotations_for_video` to retrieve "
                f"video annotations"
            )
        return self._get_latest_annotation_for_2d_media_item(media_item=media_item)
