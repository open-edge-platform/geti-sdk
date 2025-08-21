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
import warnings

from geti_sdk.data_models import (
    Dataset,
    Image,
    Model,
    Project,
    Subset,
    TrainingDatasetStatistics,
    VideoFrame,
)
from geti_sdk.data_models.containers import MediaList
from geti_sdk.http_session import GetiSession
from geti_sdk.http_session.exception import GetiRequestException
from geti_sdk.rest_converters import MediaRESTConverter
from geti_sdk.utils import deserialize_dictionary


class DatasetClient:
    """
    Class to manage datasets for a certain Intel® Geti™ project.
    """

    def __init__(self, workspace_id: str, project: Project, session: GetiSession):
        self.session = session
        self.project = project
        self.workspace_id = workspace_id
        self.base_url = f"workspaces/{workspace_id}/projects/{project.id}/datasets"

    def create_dataset(self, name: str) -> Dataset:
        """
        Create a new dataset named `name` inside the project.

        :param name: Name of the dataset to create
        :return: The created dataset
        """
        request_data = {"name": name}
        response = self.session.get_rest_response(
            url=self.base_url,
            method="POST",
            data=request_data,
        )
        dataset = deserialize_dictionary(response, output_type=Dataset)
        self.project.datasets.append(dataset)
        return dataset

    def delete_dataset(self, dataset: Dataset) -> None:
        """
        Delete provided dataset inside the project.

        :param dataset: Dataset to delete
        """
        try:
            response = self.session.get_rest_response(
                url=self.base_url + f"/{dataset.id}",
                method="DELETE",
            )
        except GetiRequestException as error:
            if error.status_code == 404:
                warnings.warn(
                    f"Dataset with name `{dataset.name}` was not found in the project. "
                    "Please make sure that the dataset you are trying to delete exists."
                )
                return
            raise error
        if isinstance(response, dict) and response.get("result", None) == "success":
            logging.info(f"Dataset `{dataset.name}` was successfully deleted.")
        else:
            logging.error(f"Failed to delete dataset `{dataset.name}`.")

    def get_all_datasets(self) -> list[Dataset]:
        """
        Query the Intel® Geti™ server to retrieve an up to date list of datasets in
        the project.

        :return: List of current datasets in the project
        """
        if self.project.id is None:
            raise ValueError(
                "Project ID is not defined, please make sure that the project you pass contains a valid ID"
            )
        response = self.session.get_rest_response(url=self.base_url, method="GET")
        datasets = [deserialize_dictionary(dataset_dict, Dataset) for dataset_dict in response["datasets"]]
        self.project.datasets = datasets
        return datasets

    def get_dataset_statistics(self, dataset: Dataset) -> dict:
        """
        Retrieve the media and annotation statistics for a particular dataset
        """
        return self.session.get_rest_response(url=f"{self.base_url}/{dataset.id}/statistics", method="GET")

    def get_dataset_by_name(self, dataset_name: str) -> Dataset:
        """
        Retrieve a dataset by name

        :param dataset_name: Name of the dataset to retrieve
        :return: Dataset object
        """
        dataset: Dataset | None = None
        for ds in self.get_all_datasets():
            if ds.name == dataset_name:
                dataset = ds
        if dataset is None:
            raise ValueError(f"Dataset named '{dataset_name}' was not found in project '{self.project.name}'")
        return dataset

    def has_dataset_subfolders(self, path_to_folder: str) -> bool:
        """
        Check if a project folder has it's media folders organized according to the
        datasets in the project

        :param path_to_folder: Path to the project folder to check
        :return: True if the media folders in the project folder tree are organized
            according to the datasets in the project.
        """
        media_folder_names = ["images", "videos"]
        result: bool = True
        for folder_name in media_folder_names:
            full_path = os.path.join(path_to_folder, folder_name)
            if not os.path.exists(full_path):
                continue
            result *= self._media_folder_has_dataset_subfolders(full_path)
        return result

    def _media_folder_has_dataset_subfolders(self, path_to_folder: str) -> bool:
        """
        Check if a folder with media has subfolders organized according to the
        projects' datasets

        :param path_to_folder: Path to the media folder to check for dataset subfolders
        :return: True if the folder contains subfolders named according to the
            datasets in the project, False otherwise
        """
        if not os.path.isdir(path_to_folder):
            return False
        content = os.listdir(path_to_folder)
        for dataset in self.project.datasets:
            if dataset.name not in content:
                return False
            if not os.path.isdir(os.path.join(path_to_folder, dataset.name)):
                return False
        return True

    def get_training_dataset_summary(self, model: Model) -> TrainingDatasetStatistics:
        """
        Return information concerning the training dataset for the `model`.
        This includes the number of images and video frames, and the statistics for
        the subset splitting (i.e. the number of training, test and validation
        images/video frames)

        :param model: Model to get the training dataset for
        :return: A `TrainingDatasetStatistics` object, containing the training dataset
            statistics for the model
        """
        ds_info = model.training_dataset_info
        dataset_storage_id = ds_info.get("dataset_storage_id", None)
        revision_id = ds_info.get("dataset_revision_id", None)
        if dataset_storage_id is None or revision_id is None:
            raise ValueError(
                f"Unable to fetch the required dataset information from the model. "
                f"Expected dataset and revision id's, got {ds_info} instead."
            )
        training_dataset = self.session.get_rest_response(
            url=f"{self.base_url}/{dataset_storage_id}/training_revisions/{revision_id}",
            method="GET",
        )
        return deserialize_dictionary(training_dataset, TrainingDatasetStatistics)

    def get_media_in_training_dataset(self, model: Model, subset: str = "training") -> Subset:
        """
        Return the media in the training dataset for the `model`, for
        the specified `subset`. Subset can be `training`, `validation` or `testing`.

        :param model: Model for which to get the media in the training dataset
        :param subset: The subset for which to return the media items. Can be either
            `training` (the default), `validation` or `testing`
        return: A `Subset` object, containing lists of `images` and `video_frames` in
            the requested `subset`
        :raises: ValueError if the DatasetClient is unable to fetch the required
            dataset information from the model
        """
        ds_info = model.training_dataset_info
        dataset_storage_id = ds_info.get("dataset_storage_id", None)
        revision_id = ds_info.get("dataset_revision_id", None)
        if dataset_storage_id is None or revision_id is None:
            raise ValueError(
                f"Unable to fetch the required dataset information from the model. "
                f"Expected dataset and revision id's, got {ds_info} instead."
            )
        post_data = {
            "condition": "and",
            "rules": [{"field": "subset", "operator": "equal", "value": subset}],
        }

        images: MediaList[Image] = MediaList([])
        video_frames: MediaList[VideoFrame] = MediaList([])

        next_page = f"{self.base_url}/{dataset_storage_id}/training_revisions/{revision_id}/media:query"
        while next_page:
            response = self.session.get_rest_response(url=next_page, method="POST", data=post_data)
            next_page = response.get("next_page", None)
            for item in response["media"]:
                if item["type"] == "image":
                    item.pop("annotation_scene_id", None)
                    item.pop("editor_name", None)
                    item.pop("roi_id", None)
                    image = MediaRESTConverter.from_dict(item, Image)
                    images.append(image)

                if item["type"] == "video":
                    video_id = item["id"]
                    next_frame_page = (
                        f"{self.base_url}/{dataset_storage_id}/training_revisions/"
                        f"{revision_id}/media/videos/{video_id}:query"
                    )
                    while next_frame_page:
                        frames_response = self.session.get_rest_response(
                            url=next_frame_page, method="POST", data=post_data
                        )
                        for frame in frames_response["media"]:
                            frame["video_id"] = video_id
                            video_frame = MediaRESTConverter.from_dict(frame, VideoFrame)
                            video_frames.append(video_frame)
                        next_frame_page = frames_response.get("next_page", None)

        return Subset(images=images, frames=video_frames, purpose=subset)
