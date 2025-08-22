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
import os
import time
import warnings
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from glob import glob
from typing import (
    Any,
    BinaryIO,
    ClassVar,
    Generic,
)

from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from geti_sdk.data_models import Dataset, Image, MediaType, Project, Video, VideoFrame
from geti_sdk.data_models.containers.media_list import MediaList, MediaTypeVar
from geti_sdk.data_models.enums.media_type import (
    SUPPORTED_IMAGE_FORMATS,
    SUPPORTED_VIDEO_FORMATS,
)
from geti_sdk.data_models.utils import numpy_from_buffer
from geti_sdk.http_session import GetiRequestException, GetiSession
from geti_sdk.rest_clients.dataset_client import DatasetClient
from geti_sdk.rest_converters.media_rest_converter import MediaRESTConverter

MEDIA_TYPE_MAPPING = {MediaType.IMAGE: Image, MediaType.VIDEO: Video}
MEDIA_SUPPORTED_FORMAT_MAPPING = {
    MediaType.IMAGE: SUPPORTED_IMAGE_FORMATS,
    MediaType.VIDEO: SUPPORTED_VIDEO_FORMATS,
}
MEDIA_DOWNLOAD_FORMAT_MAPPING = {MediaType.IMAGE: ".jpg", MediaType.VIDEO: ".mp4"}


class BaseMediaClient(Generic[MediaTypeVar]):
    """
    Class to manage media up and downloads for a certain project.
    """

    _MEDIA_TYPE: ClassVar[MediaType]

    def __init__(self, session: GetiSession, workspace_id: str, project: Project):
        self.session = session
        self._workspace_id = workspace_id
        self._base_url = f"workspaces/{workspace_id}/projects/{project.id}/datasets"
        self._project = project
        self.__media_type: type[MediaTypeVar] = self.__get_media_type(self._MEDIA_TYPE)
        self._dataset_client = DatasetClient(session=session, project=project, workspace_id=workspace_id)

    def base_url(self, dataset: Dataset) -> str:
        """
        Return the base url for the media endpoint.

        :param dataset: Dataset to retrieve the base url for
        :return: string containing the base url
        """
        return f"{self._base_url}/{dataset.id}/media/{self.plural_media_name}"

    @property
    def plural_media_name(self) -> str:
        """
        Convert the type of the media entities managed by this media client to a
        string in plural form.

        :return:
        """
        return f"{self._MEDIA_TYPE}s"

    @staticmethod
    def __get_media_type(media_type: MediaType) -> type[MediaTypeVar]:
        """
        Get the type of the media entities that are managed by this media client.

        :param media_type: MediaType Enum instance representing the type of media
            entities
        :return: Type of media entities that can be used to instantiate objects of this
            type
        """
        return MEDIA_TYPE_MAPPING[media_type]

    def _get_all(self, dataset: Dataset | None = None) -> MediaList[MediaTypeVar]:
        """
        Get a list holding all media entities of a certain type in the project.

        :param dataset: Optional dataset to retrieve the media for. If no dataset is
            specified, only media from the training dataset will be returned
        :return: MediaList holding all media of a certain type in the project
        """
        if dataset is None:
            dataset = self._project.training_dataset

        url = f"{self._base_url}/{dataset.id}/media:query?limit=100"
        data = {
            "condition": "and",
            "rules": [
                {
                    "field": "MEDIA_TYPE",
                    "operator": "EQUAL",
                    "value": f"{self._MEDIA_TYPE}",
                }
            ],
        }
        response = self.session.get_rest_response(url=url, method="POST", data=data)
        total_number_of_media: int = response[f"total_matched_{self.plural_media_name}"]

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
        return MediaList.from_rest_list(rest_input=raw_media_list, media_type=self.__media_type)

    def _delete_media(self, media_list: Sequence[MediaTypeVar]) -> bool:
        """
        Delete all media entities in `media_list` from the project.

        :param media_list: List of media entities to delete
        :return: True if all items on the media list were deleted successfully,
            False otherwise
        """
        if not isinstance(media_list, MediaList) and isinstance(media_list, Sequence):
            media_list = MediaList(media_list)
        if media_list.media_type == VideoFrame:
            raise ValueError("Unable to delete individual video frames.")
        logging.info(f"Deleting {len(media_list)} {self.plural_media_name} from project '{self._project.name}'...")
        for media_item in media_list:
            try:
                self.session.get_rest_response(url=media_item.base_url, method="DELETE")
            except GetiRequestException as error:
                if error.status_code == 409:
                    logging.info(
                        f"Project '{self._project.name}' is locked for deletion, "
                        f"unable to delete media. Aborting deletion."
                    )
                    return False
                if error.status_code == 404 or error.status_code == 204:
                    # Media item has already been deleted, continue with the rest of
                    # the list
                    continue
        return True

    def _upload_bytes(self, buffer: BinaryIO, dataset: Dataset | None = None) -> dict[str, Any]:
        """
        Upload a buffer representing a media file to the server.

        :param buffer: BinaryIO object representing a media file
        :param dataset: Dataset to upload the media to. If no dataset is passed, the
            media will be uploaded into the default (training) dataset
        :return: Dictionary containing the response of the Intel® Geti™ server, which
            holds the details of the uploaded entity
        """
        if dataset is None:
            dataset = self._project.training_dataset
        return self.session.get_rest_response(
            url=f"{self.base_url(dataset)}",
            method="POST",
            contenttype="multipart",
            data={"file": buffer},
        )

    def _upload(self, filepath: str, dataset: Dataset | None = None) -> dict[str, Any]:
        """
        Upload a media file to the server.

        :param filepath: full path to the media file on disk
        :param dataset: Dataset to upload the media to. If no dataset is passed, the
            media will be uploaded into the default (training) dataset
        :return: Dictionary containing the response of the Intel® Geti™ server, which
            holds the details of the uploaded entity
        """
        with open(filepath, "rb") as f:
            return self._upload_bytes(f, dataset=dataset)

    def _upload_loop(
        self,
        filepaths: list[str],
        skip_if_filename_exists: bool = False,
        dataset: Dataset | None = None,
        max_threads: int = 5,
    ) -> MediaList[MediaTypeVar]:
        """
        Upload media from a list of filepaths. Also checks if media items with the same
        filename exist in the project, to make sure no duplicate items are uploaded.

        :param filepaths: List of full filepaths for media that should be
            uploaded
        :param skip_if_filename_exists: Set to True to skip uploading of a media item
            if a media item with the same filename already exists in the dataset.
            Defaults to False
        :param dataset: Dataset to upload the media to. If no dataset is passed, the
            media will be uploaded into the default (training) dataset
        :param max_threads: Maximum number of threads to use for uploading. Defaults to 5.
            Set to -1 to use all available threads.
        :return: MediaList containing a list of all media entities that were uploaded
            to the project
        """
        if max_threads <= 0:
            # ThreadPoolExecutor will use minimum 5 threads for 1 core cpu
            # and maximum 32 threads for multi-core cpu.
            max_threads = None
        if dataset is None:
            dataset = self._project.training_dataset
        media_in_project = self._get_all(dataset=dataset)
        uploaded_media: MediaList[MediaTypeVar] = MediaList[MediaTypeVar]([])
        upload_count = 0
        skip_count = 0
        logging.info(f"Starting {self._MEDIA_TYPE} upload to dataset '{dataset.name}'...")
        tqdm_prefix = f"Uploading {self.plural_media_name}"

        t_start = time.time()

        def upload_file(filepath: str) -> None:
            nonlocal upload_count, skip_count
            name, ext = os.path.splitext(os.path.basename(filepath))
            if name in media_in_project.names and skip_if_filename_exists:
                skip_count += 1
                return
            try:
                media_dict = self._upload(filepath=filepath, dataset=dataset)
            except GetiRequestException as error:
                if error.status_code == 500:
                    logging.error(f"Failed to upload {self._MEDIA_TYPE} '{name}'. Error message: {error}")
                    return
                raise error
            media_item = MediaRESTConverter.from_dict(input_dict=media_dict, media_type=self.__media_type)
            if isinstance(media_item, Video):
                media_item._data = filepath
            # appends are thread safe
            media_in_project.append(media_item)
            uploaded_media.append(media_item)
            upload_count += 1

        with ThreadPoolExecutor(max_workers=max_threads) as executor, logging_redirect_tqdm(tqdm_class=tqdm):
            list(
                tqdm(  # List unwraps the generator
                    executor.map(upload_file, filepaths),
                    total=len(filepaths),
                    desc=tqdm_prefix,
                )
            )

        t_elapsed = time.time() - t_start
        if upload_count > 0:
            msg = f"Upload complete. Uploaded {upload_count} new {self.plural_media_name} in {t_elapsed:.1f} seconds."
        else:
            msg = f"No new {self.plural_media_name} were uploaded."
        if skip_count > 0:
            msg = (
                msg + f" Found {skip_count} {self.plural_media_name} that already "
                f"existed in project, these {self.plural_media_name} were"
                f" skipped."
            )
        logging.info(msg)
        return uploaded_media

    def _upload_folder(
        self,
        path_to_folder: str,
        n_media: int = -1,
        skip_if_filename_exists: bool = False,
        dataset: Dataset | None = None,
        max_threads: int = 5,
    ) -> MediaList[MediaTypeVar]:
        """
        Upload all media in a folder to the project. Returns the mapping of filenames
        to the unique IDs assigned by Intel Geti.

        :param path_to_folder: Folder with media items to upload
        :param n_media: Number of media to upload from folder
        :param skip_if_filename_exists: Set to True to skip uploading of a media item
            if a media item with the same filename already exists in the project.
            Defaults to False
        :param dataset: Dataset to upload the media to. If no dataset is passed, the
            media will be uploaded into the default (training) dataset
        :param max_threads: Maximum number of threads to use for uploading. Defaults to 5.
            Set to -1 to use all available threads.
        :return: MediaList containing a list of all media entities that were uploaded
            to the project
        """
        media_formats = MEDIA_SUPPORTED_FORMAT_MAPPING[self._MEDIA_TYPE]
        filepaths: list[str] = []
        for media_extension in media_formats:
            filepaths += glob(
                os.path.join(path_to_folder, "**", f"*{media_extension}"),
                recursive=True,
            )
        n_files = len(filepaths)
        if n_media == -1:
            n_to_upload = n_files
        elif n_media < -1:
            raise ValueError(
                f"Number of {self.plural_media_name} to upload must positive, or -1 "
                f"to signify uploading all files in the folder. "
            )
        else:
            n_to_upload = n_files if n_files < n_media else n_media
        return self._upload_loop(
            filepaths=filepaths[0:n_to_upload],
            skip_if_filename_exists=skip_if_filename_exists,
            dataset=dataset,
            max_threads=max_threads,
        )

    def _download_all(
        self,
        path_to_folder: str,
        append_media_uid: bool = False,
        max_threads: int = 10,
        dataset: Dataset | None = None,
    ) -> None:
        """
        Download all media entities in a project to a folder on the local disk.

        :param path_to_folder: path to the folder in which the media should be saved
        :param append_media_uid: True to append the UID of a media item to the
            filename (separated from the original filename by an underscore, i.e.
            '{filename}_{media_id}').
        :param max_threads: Maximum number of threads to use for downloading.
            Defaults to 10. Set to -1 to use all available threads.
        :return:
        """
        datasets = self._dataset_client.get_all_datasets()
        if dataset is not None:
            if dataset not in datasets:
                warnings.warn(
                    f"Dataset '{dataset.name}' not found in project '{self._project.name}'. "
                    f"Skipping download for this dataset."
                )
                return
            datasets = [dataset]
        for dataset in datasets:
            path_to_media_folder = os.path.join(path_to_folder, self.plural_media_name, dataset.name)
            self._download_dataset(
                dataset=dataset,
                path_to_media_folder=path_to_media_folder,
                append_media_uid=append_media_uid,
                max_threads=max_threads,
            )

    def _download_dataset(
        self,
        dataset: Dataset,
        path_to_media_folder: str,
        append_media_uid: bool = False,
        max_threads: int = 10,
    ):
        """
        Download all media items of a single type in the dataset to a folder on disk

        :param dataset: Dataset to download the media for
        :param path_to_media_folder: path to the folder in which the media should be
            saved
        :param append_media_uid: True to append the UID of a media item to the
            filename (separated from the original filename by an underscore, i.e.
            '{filename}_{media_id}').
        :param max_threads: Maximum number of threads to use for downloading.
            Defaults to 10. Set to -1 to use all available threads.
        """
        if max_threads <= 0:
            # ThreadPoolExecutor will use minimum 5 threads for 1 core cpu
            # and maximum 32 threads for multi-core cpu.
            max_threads = None
        media_list = self._get_all(dataset=dataset)
        os.makedirs(path_to_media_folder, exist_ok=True, mode=0o770)
        logging.info(
            f"Downloading {len(media_list)} {self.plural_media_name} from project "
            f"'{self._project.name}' and dataset '{dataset.name}' to folder "
            f"{path_to_media_folder}..."
        )
        t_start = time.time()
        download_count = 0
        existing_count = 0
        tqdm_prefix = f"Downloading {self.plural_media_name}"

        def download_file(media_item: MediaTypeVar) -> None:
            nonlocal download_count, existing_count
            uid_string = ""
            if append_media_uid:
                uid_string = f"_{media_item.id}"
            media_filepath = os.path.join(
                path_to_media_folder,
                os.path.basename(media_item.name) + uid_string + MEDIA_DOWNLOAD_FORMAT_MAPPING[self._MEDIA_TYPE],
            )
            if os.path.exists(media_filepath) and os.path.isfile(media_filepath):
                existing_count += 1
                return
            try:
                response = self.session.get_rest_response(
                    url=media_item.download_url,
                    method="GET",
                    contenttype="jpeg",
                    include_organization_id=False,
                )
            except GetiRequestException as error:
                if error.status_code == 500:
                    logging.error(
                        f"Failed to download {self._MEDIA_TYPE} '{media_item.name}' "
                        f"with ID '{media_item.id}'. Error message: {error}"
                    )
                    return
                raise error

            with open(media_filepath, "wb") as f:
                f.write(response.content)
            if isinstance(media_item, Image | VideoFrame):
                # Set the numpy data attribute if the media item supports it
                media_item._data = numpy_from_buffer(response.content)
            elif isinstance(media_item, Video):
                media_item._data = media_filepath
            download_count += 1

        with ThreadPoolExecutor(max_workers=max_threads) as executor, logging_redirect_tqdm(tqdm_class=tqdm):
            list(
                tqdm(  # List unwraps the generator
                    executor.map(download_file, media_list),
                    total=len(media_list),
                    desc=tqdm_prefix,
                )
            )

        t_elapsed = time.time() - t_start
        if download_count > 0:
            msg = f"Downloaded {download_count} {self.plural_media_name} in {t_elapsed:.1f} seconds."
        else:
            msg = f"No {self.plural_media_name} were downloaded."
        if existing_count > 0:
            msg += (
                f" {existing_count} existing {self.plural_media_name} were found "
                f"in the target folder, download was skipped for these "
                f"{self.plural_media_name}."
            )
        logging.info(msg)
