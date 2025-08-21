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

import os
from abc import abstractmethod
from glob import glob

from geti_sdk.data_models.annotations import Annotation
from geti_sdk.data_models.enums import TaskType
from geti_sdk.data_models.media import MediaInformation


class AnnotationReader:
    """
    Base class for annotation reading, to handle loading and converting annotations
    to Intel Geti format
    """

    def __init__(
        self,
        base_data_folder: str,
        annotation_format: str = ".json",
        task_type: TaskType | str = TaskType.DETECTION,
        anomaly_reduction: bool = False,
    ):
        if task_type is not None and not isinstance(task_type, TaskType):
            task_type = TaskType(task_type)
        self.base_folder = base_data_folder
        self.annotation_format = annotation_format
        self.task_type = task_type
        self.anomaly_reduction = anomaly_reduction

        self._filepaths: list[str] | None = None

    @abstractmethod
    def get_data(
        self,
        filename: str,
        label_name_to_id_mapping: dict,
        media_information: MediaInformation,
        preserve_shape_for_global_labels: bool = False,
    ) -> list[Annotation]:
        """
        Get annotation data for a certain filename
        """
        raise NotImplementedError

    def get_data_filenames(self) -> list[str]:
        """
        Return a list of annotation files found in the `base_data_folder`.

        :return: List of filenames (excluding extension) for all annotation files in
            the data folder
        """
        if self._filepaths is None:
            filepaths = glob(os.path.join(self.base_folder, f"*{self.annotation_format}"))
            self._filepaths = [os.path.splitext(os.path.basename(filepath))[0] for filepath in filepaths]
        return self._filepaths

    @abstractmethod
    def get_all_label_names(self) -> list[str]:
        """
        Return a list of unique label names that were found in the annotation data
        folder belonging to this AnnotationReader instance.
        """
        raise NotImplementedError

    @abstractmethod
    def get_keypoint_joints(self) -> set[tuple[int, int]]:
        """
        Return a list of tuples representing the joints between keypoints
        """
        raise NotImplementedError

    def prepare_and_set_dataset(
        self,
        task_type: TaskType | str,
        previous_task_type: TaskType | None = None,
    ) -> None:
        """
        Prepare a dataset for uploading annotations for a certain task_type.

        :param task_type: TaskType to prepare the dataset for
        :param previous_task_type: Optional type of the (trainable) task preceding
            the current task in the pipeline. This is only used for global tasks
        """
        if not isinstance(task_type, TaskType):
            task_type = TaskType(task_type)
        if task_type in [
            TaskType.DETECTION,
            TaskType.SEGMENTATION,
            TaskType.CLASSIFICATION,
        ]:
            self.task_type = task_type
        else:
            raise ValueError(f"Unsupported task_type {task_type}")

    @property
    def applied_filters(self) -> list[dict[str, list[str] | str]]:
        """
        Return a list of dictionaries representing the filter settings that have
        been applied to the dataset, if any.

        Dictionaries in this list contain two keys:

        - 'labels'      -- List of label names which has been filtered on
        - 'criterion'   -- String representing the criterion that has been used in the
                           filtering. Can be 'OR', 'AND', 'XOR' or 'NOT'.

        :return: List of filter settings that have been applied to the dataset. Returns
            an empty list if no filters have been applied.
        """
        return []
