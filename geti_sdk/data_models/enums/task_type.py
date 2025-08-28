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

from enum import Enum


class TaskType(Enum):
    """
    Enum representing the different task types in Geti™ projects.
    """

    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"
    ANOMALY_CLASSIFICATION = "anomaly_classification"
    ANOMALY_DETECTION = "anomaly_detection"
    ANOMALY_SEGMENTATION = "anomaly_segmentation"
    ANOMALY = "anomaly"
    INSTANCE_SEGMENTATION = "instance_segmentation"
    ROTATED_DETECTION = "rotated_detection"
    KEYPOINT_DETECTION = "keypoint_detection"
    DATASET = "dataset"
    CROP = "crop"

    def __str__(self) -> str:
        """
        Return the string representation of the TaskType instance.

        :return: string containing the task type
        """
        return self.value

    @property
    def is_trainable(self) -> bool:
        """
        Return True if a task of this TaskType is trainable, False otherwise.

        :return:
        """
        return self not in NON_TRAINABLE_TASK_TYPES

    @property
    def is_global(self) -> bool:
        """
        Return True if a task of this TaskType produces global labels, False otherwise.

        :return:
        """
        return self in GLOBAL_TASK_TYPES

    @property
    def is_local(self) -> bool:
        """
        Return True if a task of this TaskType produces local labels, False otherwise.

        :return:
        """
        return self not in GLOBAL_TASK_TYPES and self not in NON_TRAINABLE_TASK_TYPES

    @property
    def is_anomaly(self) -> bool:
        """
        Return True if a task of this TaskType is an anomaly task, False otherwise.

        :return:
        """
        return self in ANOMALY_TASK_TYPES

    @property
    def is_segmentation(self) -> bool:
        """
        Return True if a task of this TaskType is a segmentation task, False otherwise.

        :return:
        """
        return self in SEGMENTATION_TASK_TYPES

    @property
    def is_detection(self) -> bool:
        """
        Return True if a task of this TaskType is a detection task, False otherwise.

        :return:
        """
        return self in DETECTION_TASK_TYPES

    @property
    def is_keypoint_detection(self) -> bool:
        """
        Return True if a task of this TaskType is a keypoint detection task, False otherwise.

        :return:
        """
        return self in KEYPOINT_DETECTION_TASK_TYPES

    @classmethod
    def from_domain(cls, domain):
        """
        Instantiate a :py:class:`~geti_sdk.data_models.enums.task_type.TaskType`
        from a given :py:class:`~geti_sdk.data_models.enums.domain.Domain`.

        :param domain: domain to get the TaskType for
        :return: TaskType instance corresponding to the `domain`
        """
        return cls[domain.name]


NON_TRAINABLE_TASK_TYPES = [TaskType.DATASET, TaskType.CROP]

ANOMALY_TASK_TYPES = [
    TaskType.ANOMALY_CLASSIFICATION,
    TaskType.ANOMALY_DETECTION,
    TaskType.ANOMALY_SEGMENTATION,
    TaskType.ANOMALY,
]

GLOBAL_TASK_TYPES = [
    TaskType.CLASSIFICATION,
    TaskType.ANOMALY_CLASSIFICATION,
    TaskType.ANOMALY,
]

SEGMENTATION_TASK_TYPES = [
    TaskType.SEGMENTATION,
    TaskType.ANOMALY_SEGMENTATION,
    TaskType.INSTANCE_SEGMENTATION,
]

DETECTION_TASK_TYPES = [
    TaskType.DETECTION,
    TaskType.ROTATED_DETECTION,
    TaskType.ANOMALY_DETECTION,
]

KEYPOINT_DETECTION_TASK_TYPES = [
    TaskType.KEYPOINT_DETECTION,
]
