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
from typing import ClassVar

import attr

from geti_sdk.data_models.containers.media_list import MediaList
from geti_sdk.data_models.enums import SubsetPurpose
from geti_sdk.data_models.media import Image, VideoFrame
from geti_sdk.data_models.utils import (
    deidentify,
    str_to_datetime,
    str_to_enum_converter,
)


@attr.define
class Dataset:
    """
    Representation of a dataset for a project in Intel® Geti™.

    :var id: Unique database ID of the dataset
    :var name: name of the dataset
    """

    _identifier_fields: ClassVar[str] = ["id", "creation_time"]
    _GET_only_fields: ClassVar[list[str]] = ["use_for_training", "creation_time"]

    name: str
    id: str | None = None
    creation_time: str | None = attr.field(default=None, converter=str_to_datetime)
    use_for_training: bool | None = None

    def deidentify(self) -> None:
        """
        Remove unique database ID from the Dataset.
        """
        deidentify(self)

    def prepare_for_post(self) -> None:
        """
        Set all fields to None that are not valid for making a POST request to the
        /projects endpoint.

        :return:
        """
        for field_name in self._GET_only_fields:
            setattr(self, field_name, None)


@attr.define
class TrainingDatasetStatistics:
    """
    Statistics for a specific dataset that was used for training a model. Note that
    a `dataset` includes both the training, validation and testing set.
    """

    id: str
    creation_time: str = attr.field(converter=str_to_datetime)
    subset_info: dict[str, int]
    dataset_info: dict[str, int]

    @property
    def training_size(self) -> int:
        """Return the number of dataset items in the training set"""
        return self.subset_info["training"]

    @property
    def validation_size(self) -> int:
        """Return the number of dataset items in the validation set"""
        return self.subset_info["validation"]

    @property
    def testing_size(self) -> int:
        """Return the number of dataset items in the testing set"""
        return self.subset_info["testing"]

    @property
    def number_of_videos(self) -> int:
        """Return the total number of videos in the dataset"""
        return self.dataset_info["videos"]

    @property
    def number_of_frames(self) -> int:
        """Return the total number of video frames in the dataset"""
        return self.dataset_info["frames"]

    @property
    def number_of_images(self) -> int:
        """Return the total number of images in the dataset"""
        return self.dataset_info["images"]


@attr.define
class Subset:
    """
    Return the media items for a specific subset (i.e. 'training', 'validation' or
    'testing')

    :var images: List of images in the subset
    :var frames: List of video frames in the subset
    :var purpose: string representing the purpose of the subset. Can be either

    """

    images: MediaList[Image]
    frames: MediaList[VideoFrame]
    purpose: str = attr.field(converter=str_to_enum_converter(SubsetPurpose))

    @property
    def size(self) -> int:
        """Return the total number of items in the subset"""
        return len(self.images) + len(self.frames)
