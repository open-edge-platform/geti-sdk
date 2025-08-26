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

from geti_sdk.data_models.enums.task_type import TaskType


class Domain(Enum):
    """
    Enum representing the different task domains in Geti™ projects.
    """

    DETECTION = "DETECTION"
    SEGMENTATION = "SEGMENTATION"
    CLASSIFICATION = "CLASSIFICATION"
    ANOMALY_CLASSIFICATION = "ANOMALY_CLASSIFICATION"
    ANOMALY_DETECTION = "ANOMALY_DETECTION"
    ANOMALY_SEGMENTATION = "ANOMALY_SEGMENTATION"
    ANOMALY = "ANOMALY"
    INSTANCE_SEGMENTATION = "INSTANCE_SEGMENTATION"
    ROTATED_DETECTION = "ROTATED_DETECTION"
    KEYPOINT_DETECTION = "KEYPOINT_DETECTION"

    def __str__(self) -> str:
        """
        Return the string representation of the Domain instance.
        """
        return self.value

    @classmethod
    def from_task_type(cls, task_type: TaskType) -> "Domain":
        """
        Return the Domain corresponding to a certain task type.

        :param task_type: TaskType to retrieve the domain for
        :return: Domain corresponding to task_type
        """
        return cls[task_type.name]
