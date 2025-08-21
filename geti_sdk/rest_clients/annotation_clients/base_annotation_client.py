# Copyright (C) 2024 Intel Corporation
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

import json
import logging
import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, TypeVar

from requests import Response
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from geti_sdk.annotation_readers import AnnotationReader
from geti_sdk.data_models import (
    AnnotationKind,
    AnnotationScene,
    Dataset,
    Image,
    Project,
    Video,
    VideoFrame,
)
from geti_sdk.data_models.containers.media_list import MediaList
from geti_sdk.data_models.label import Label
from geti_sdk.data_models.media import MediaInformation, MediaItem
from geti_sdk.http_session import GetiRequestException, GetiSession
from geti_sdk.rest_clients.dataset_client import DatasetClient
from geti_sdk.rest_converters import AnnotationRESTConverter

AnnotationReaderType = TypeVar("AnnotationReaderType", bound=AnnotationReader)
MediaType = TypeVar("MediaType", Image, Video)


class BaseAnnotationClient:
    """
    Class to up- or download annotations for 2d media to an existing project.
    """

    def __init__(
        self,
        session: GetiSession,
        workspace_id: str,
        project: Project,
        annotation_reader: AnnotationReaderType | None = None,
    ):
        self.session = session
        self.workspace_id = workspace_id
        self.annotation_reader = annotation_reader
        self._project = project
        label_mapping = None if annotation_reader is None else self.__get_label_mapping(project)
        self._label_mapping = label_mapping
        self._dataset_client = DatasetClient(session=session, project=project, workspace_id=workspace_id)

    def _get_all_media_by_type(self, media_type: type[MediaType]) -> MediaList[MediaType]:
        """
        Get a list holding all media entities of type `media_type` in the project.

        :param media_type: Type of media item to retrieve. Can be 'Image' or 'Video'
        :return: MediaList holding all media of a certain type in the project
        """
        datasets = self._dataset_client.get_all_datasets()
        media_list = MediaList[media_type]([])
        for dataset in datasets:
            media_list.extend(self._get_all_media_in_dataset_by_type(media_type=media_type, dataset=dataset))
        return media_list

    def _get_all_media_in_dataset_by_type(self, media_type: type[MediaType], dataset: Dataset) -> MediaList[MediaType]:
        """
        Return a list of all media items of type `media_type` in the dataset `dataset`

        :param media_type: Type of media item to retrieve. Can be 'Image' or 'Video'
        :param dataset: Dataset to retrieve media items for
        :return: List of all media items of a certain media_type in the dataset
        """
        if media_type == Image:
            media_name = "images"
            single_media_name = "image"
        elif media_type == Video:
            media_name = "videos"
            single_media_name = "video"
        else:
            raise ValueError(f"Invalid media type specified: {media_type}.")

        url = f"workspaces/{self.workspace_id}/projects/{self._project.id}/datasets/{dataset.id}/media:query?limit=100"
        data = {
            "condition": "and",
            "rules": [
                {
                    "field": "MEDIA_TYPE",
                    "operator": "EQUAL",
                    "value": single_media_name,
                }
            ],
        }
        response = self.session.get_rest_response(url=url, method="POST", data=data)
        total_number_of_media: int = response[f"total_matched_{media_name}"]

        raw_media_list: list[dict[str, Any]] = []
        while len(raw_media_list) < total_number_of_media:
            for media_item_dict in response["media"]:
                raw_media_list.append(media_item_dict)
            if "next_page" in response:
                response = self.session.get_rest_response(
                    url=response["next_page"],
                    method="POST",
                    data=data,
                    include_organization_id=False,
                )
        return MediaList.from_rest_list(rest_input=raw_media_list, media_type=media_type)

    def __get_label_mapping(self, project: Project) -> dict[str, str]:
        """
        Get the mapping of the label names to the label ids for the project

        :param project:
        :return: Dictionary containing the label names as keys and the label ids as
            values
        """
        project_label_name_to_label: dict[str, Label] = {}
        for label in project.pipeline.get_all_labels():
            if label.is_empty:
                # We perform a casefold on the label name to ensure that we can match
                # the empy labels from projects created in older versions of the
                # Intel Geti platform.
                label_name = label.name.casefold()
            else:
                label_name = label.name
            project_label_name_to_label[label_name] = label

        source_label_names = set(self.annotation_reader.get_all_label_names())
        source_label_name_to_project_label_id: dict[str, str] = {}
        # We include the project label names in the mapping, as we want to ensure that
        # we can match the labels from the source to the project labels.
        for source_label_name in source_label_names.union(project_label_name_to_label.keys()):
            if source_label_name in project_label_name_to_label:
                label_id = project_label_name_to_label[source_label_name].id
            elif (
                source_label_name.casefold() in project_label_name_to_label
                and project_label_name_to_label[source_label_name.casefold()].is_empty
            ):
                label_id = project_label_name_to_label[source_label_name.casefold()].id
            else:
                raise ValueError(
                    f"Found label {source_label_name} in source labels, but this label is not in the project labels."
                )
            if label_id is None:
                raise ValueError(f"Unable to find label id for label {source_label_name}.")
            source_label_name_to_project_label_id[source_label_name] = label_id
        return source_label_name_to_project_label_id

    @property
    def label_mapping(self) -> dict[str, str]:
        """
        Return a dictionary with label names as keys and label ids as values.

        :return:
        """
        if self.annotation_reader is not None:
            if self._label_mapping is None:
                self._label_mapping = self.__get_label_mapping(self._project)
            return self._label_mapping
        raise ValueError(
            "Unable to get label mapping for this annotation client, no annotation reader has been defined."
        )

    def _upload_annotation_for_2d_media_item(
        self,
        media_item: Image | VideoFrame,
        annotation_scene: AnnotationScene | None = None,
    ) -> AnnotationScene:
        """
        Upload a new annotation for an image or video frame to the cluster. This will
        overwrite any current annotations for the media item.

        If an `annotation_scene` is passed, this annotation will be applied to the
        media_item.

        If `annotation_scene` is left as None and an AnnotationReader is defined for
        the AnnotationClient, this method will read the annotation for
        the media item from the AnnotationReader.

        If `annotation_scene` is left as None and no AnnotationReader is defined for
        the AnnotationClient, this method will raise a ValueError

        :param media_item: Image or VideoFrame to upload annotation for
        :param annotation_scene: Optional AnnotationScene to apply to the media_item.
            If left as None, this method will read the annotation data using the
            AnnotationReader
        :return: AnnotationScene that was uploaded
        """
        if annotation_scene is not None:
            scene_to_upload = annotation_scene.apply_identifier(media_identifier=media_item.identifier)
        else:
            if self.annotation_reader is not None:
                scene_to_upload = self._read_2d_media_annotation_from_source(media_item=media_item)
            else:
                raise ValueError(
                    "You attempted to upload an annotation for a media item, but no "
                    "annotation data was passed directly and no annotation reader was "
                    "defined for the AnnotationClient. Therefore, the "
                    "AnnotationClient is unable to upload any annotation data."
                )
        if scene_to_upload.has_data:
            scene_to_upload.prepare_for_post()
            rest_data = AnnotationRESTConverter.to_dict(scene_to_upload, deidentify=False)
            rest_data.pop("kind")
            self.session.get_rest_response(
                url=f"{media_item.base_url}/annotations",
                method="POST",
                data=rest_data,
                include_organization_id=False,
            )
        return scene_to_upload

    def _append_annotation_for_2d_media_item(self, media_item: Image | VideoFrame) -> AnnotationScene:
        """
        Add an annotation to the existing annotations for the `media_item`.

        :param media_item: Image or VideoFrame to append the annotation for
        :return: Returns the response of the REST endpoint to post the updated
            annotation
        """
        new_annotation_scene = self._read_2d_media_annotation_from_source(
            media_item=media_item, preserve_shape_for_global_labels=True
        )
        annotation_scene = self._get_latest_annotation_for_2d_media_item(media_item)
        if annotation_scene is None:
            logging.info(f"No existing annotation found for {str(media_item.type)} named {media_item.name}")
            annotation_scene = AnnotationScene(
                media_identifier=media_item.identifier,
                annotations=[],
                kind=AnnotationKind.ANNOTATION.value,
            )
        annotation_scene.extend(new_annotation_scene.annotations)

        if annotation_scene.has_data:
            rest_data = AnnotationRESTConverter.to_dict(annotation_scene, deidentify=False)
            rest_data.pop("kind", None)
            rest_data.pop("annotation_state_per_task", None)
            rest_data.pop("id", None)
            response = self.session.get_rest_response(
                url=f"{media_item.base_url}/annotations",
                method="POST",
                data=rest_data,
                include_organization_id=False,
            )
            return AnnotationRESTConverter.from_dict(response)
        return annotation_scene

    def _upload_annotations_for_2d_media_list(
        self,
        media_list: Sequence[MediaItem],
        append_annotations: bool,
        max_threads: int = 5,
    ) -> int:
        """
        Upload annotations to the server.

        :param media_list: List of images or video frames to upload the annotations
            for
        :param append_annotations: True to append annotations from the local disk to
            the existing annotations on the server, False to overwrite the server
            annotations by those on the local disk.
        :param max_threads: Maximum number of threads to use for uploading. Defaults to 5.
            Set to -1 to use all available threads.
        :return: Returns the number of uploaded annotations.
        """
        if max_threads <= 0:
            # ThreadPoolExecutor will use minimum 5 threads for 1 core cpu
            # and maximum 32 threads for multi-core cpu.
            max_threads = None
        upload_count = 0
        skip_count = 0
        tqdm_prefix = "Uploading media annotations"

        def upload_annotation(media_item: MediaItem) -> None:
            nonlocal upload_count, skip_count
            try:
                if not append_annotations:
                    response = self._upload_annotation_for_2d_media_item(media_item=media_item)
                else:
                    response = self._append_annotation_for_2d_media_item(media_item=media_item)
            except GetiRequestException as error:
                skip_count += 1
                if error.status_code == 500:
                    logging.error(f"Failed to upload annotation for {media_item.name}. ")
                    return
                raise error
            if response is not None:
                upload_count += 1

        t_start = time.time()
        with ThreadPoolExecutor(max_workers=max_threads) as executor, logging_redirect_tqdm(tqdm_class=tqdm):
            list(
                tqdm(
                    executor.map(upload_annotation, media_list),
                    total=len(media_list),
                    desc=tqdm_prefix,
                )
            )

        t_elapsed = time.time() - t_start
        if upload_count > 0:
            logging.info(f"Uploaded {upload_count} annotations in {t_elapsed:.1f} seconds.")
        if skip_count > 0:
            logging.info(f"Skipped {skip_count} media items, unable to upload annotations.")
        return upload_count

    def annotation_scene_from_rest_response(
        self, response_dict: dict[str, Any], media_information: MediaInformation
    ) -> AnnotationScene:
        """
        Convert a dictionary with annotation data obtained from the Intel® Geti™
        /annotations rest endpoint into an annotation scene.

        :param response_dict: Dictionary containing the annotation data
        :param media_information: MediaInformation about the media item to which the
            annotation applies
        :return: AnnotationScene object corresponding to the data in the response_dict
        """
        return AnnotationRESTConverter.from_dict(response_dict)

    def _get_latest_annotation_for_2d_media_item(self, media_item: Image | VideoFrame) -> AnnotationScene | None:
        """
        Retrieve the latest annotation for an image or video frame from the cluster.
        If no annotation is available, this method returns None

        :param media_item: Image or VideoFrame to retrieve the annotations for
        :return: Dictionary containing the annotations data
        """
        response = self.session.get_rest_response(
            url=f"{media_item.base_url}/annotations/latest",
            method="GET",
            include_organization_id=False,
        )
        if type(response) is Response and response.status_code == 204:
            return None
        annotation_scene = self.annotation_scene_from_rest_response(response, media_item.media_information)
        annotation_scene.resolve_label_names_and_colors(labels=self._project.get_all_labels())
        return annotation_scene

    def _read_2d_media_annotation_from_source(
        self,
        media_item: Image | VideoFrame,
        preserve_shape_for_global_labels: bool = False,
    ) -> AnnotationScene:
        """
        Retrieve the annotation for the media_item, and return it in the
        proper format to be sent to the GETi /annotations endpoint. This method uses the
        `self.annotation_reader` to get the annotation data.

        :param media_item: MediaItem to read the annotation for
        :return: Dictionary containing the annotation, in GETi format
        """
        annotation_list = self.annotation_reader.get_data(
            filename=media_item.name,
            label_name_to_id_mapping=self.label_mapping,
            media_information=media_item.media_information,
            preserve_shape_for_global_labels=preserve_shape_for_global_labels,
        )
        return AnnotationRESTConverter.from_dict(
            {
                "media_identifier": media_item.identifier,
                "annotations": annotation_list,
                "kind": AnnotationKind.ANNOTATION,
            }
        )

    def _download_annotations_for_2d_media_list(
        self,
        media_list: MediaList[Image] | MediaList[VideoFrame],
        path_to_folder: str,
        append_media_uid: bool = False,
        verbose: bool = True,
        max_threads: int = 10,
    ) -> float:
        """
        Download annotations from the server to a target folder on disk.

        :param media_list: List of images or video frames to download the annotations
            for
        :param path_to_folder: Folder to save the annotations to
        :param append_media_uid: True to append the UID of a media item to the
            annotation filename (separated from the original filename by an underscore,
             i.e. '{filename}_{media_id}').
        :param max_threads: Maximum number of threads to use for downloading. Defaults to 10.
            Set to -1 to use all available threads.
        :return: Returns the time elapsed to download the annotations, in seconds
        """
        if max_threads <= 0:
            # ThreadPoolExecutor will use minimum 5 threads for 1 core cpu
            # and maximum 32 threads for multi-core cpu.
            max_threads = None
        path_to_annotations_folder = os.path.join(path_to_folder, "annotations")
        os.makedirs(path_to_annotations_folder, exist_ok=True, mode=0o770)
        if media_list.media_type == Image:
            media_name = "image"
            media_name_plural = "images"
        elif media_list.media_type == VideoFrame:
            media_name = "video frame"
            media_name_plural = "video frames"
        else:
            raise ValueError("Invalid media type found in media_list, unable to download annotations.")
        if verbose:
            logging.info(
                f"Starting annotation download... saving annotations for "
                f"{len(media_list)} {media_name_plural} to folder "
                f"{path_to_annotations_folder}"
            )
        t_start = time.time()
        download_count = 0
        skip_count = 0
        tqdm_prefix = f"Downloading {media_name} annotations"

        def download_annotation(media_item: Image | VideoFrame) -> None:
            nonlocal skip_count, download_count
            annotation_scene = self._get_latest_annotation_for_2d_media_item(media_item)
            if annotation_scene is None:
                log_msg = (
                    f"Unable to retrieve latest annotation for {media_name} "
                    f"{media_item.name}. Skipping this {media_name}"
                )
                if verbose:
                    logging.info(log_msg)
                else:
                    logging.debug(log_msg)
                skip_count += 1
                return
            kind = annotation_scene.kind
            if kind != AnnotationKind.ANNOTATION:
                log_msg = f"Received invalid annotation of kind {kind} for {media_name} with name{media_item.name}"
                if verbose:
                    logging.info(log_msg)
                else:
                    logging.debug(log_msg)
                skip_count += 1
                return
            export_data = AnnotationRESTConverter.to_dict(annotation_scene)

            base_media_item_name = os.path.basename(media_item.name)
            filename = base_media_item_name + ".json"
            if append_media_uid:
                if isinstance(media_item, Image):
                    filename = f"{base_media_item_name}_{media_item.id}.json"
                elif isinstance(media_item, VideoFrame):
                    if media_item.video_name is not None:
                        base_video_name = os.path.basename(media_item.video_name)
                        filename = (
                            f"{base_video_name}_"
                            f"{media_item.media_information.video_id}_frame_"
                            f"{media_item.media_information.frame_index}.json"
                        )
                    else:
                        video_name = base_media_item_name.split("_frame_")[0]
                        filename = (
                            f"{video_name}_"
                            f"{media_item.media_information.video_id}_frame_"
                            f"{media_item.media_information.frame_index}.json"
                        )
                else:
                    raise TypeError(f"Received invalid media item of type {type(media_item)}.")

            annotation_path = os.path.join(path_to_annotations_folder, filename)

            with open(annotation_path, "w") as f:
                json.dump(export_data, f, indent=4)
            download_count += 1

        with ThreadPoolExecutor(max_workers=max_threads) as executor, logging_redirect_tqdm(tqdm_class=tqdm):
            list(
                tqdm(  # List unwraps the generator
                    executor.map(download_annotation, media_list),
                    total=len(media_list),
                    desc=tqdm_prefix,
                )
            )

        t_elapsed = time.time() - t_start
        if download_count > 0:
            msg = (
                f"Downloaded {download_count} annotations to folder "
                f"{path_to_annotations_folder} in {t_elapsed:.1f} seconds."
            )
        else:
            msg = "No annotations were downloaded."
        if skip_count > 0:
            msg = (
                msg + f" Was unable to retrieve annotations for {skip_count} "
                f"{media_name_plural}, these {media_name_plural} were skipped."
            )
        if verbose:
            logging.info(msg)
        return t_elapsed
