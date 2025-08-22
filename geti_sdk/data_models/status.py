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
from typing import Any

import attr

from geti_sdk.data_models.performance import Performance


@attr.define(slots=False)
class StatusSummary:
    """
    Summary of the status of a project or task on the GETi cluster.

    NOTE: the 'message' attribute was removed in Geti 1.1

    :var progress: Training progress, if a model is being trained
    :var message: Optional Human readable message describing the status
    """

    progress: float
    message: str | None = None

    def __attrs_post_init__(self):
        """
        Initialize private attributes
        """
        self._user_friendly_message = "Unknown status"

    @classmethod
    def from_dict(cls, status_dict: dict[str, Any]) -> "StatusSummary":
        """
        Create a StatusSummary object from a dictionary.

        :param status_dict: Dictionary representing a status, as returned by the GETi
            /status and /jobs endpoints
        :return: StatusSummary object holding the status data contained in `status_dict`
        """
        return cls(**status_dict)

    @property
    def user_friendly_message(self) -> str:
        """
        Return a message describing the status in a human readable format
        """
        if self.message is not None:
            return self.message
        return self._user_friendly_message

    @user_friendly_message.setter
    def user_friendly_message(self, message: str):
        """
        Set the user friendly message to describe the status
        """
        self._user_friendly_message = message


@attr.define
class LabelAnnotationRequirements:
    """
    Detailed information regarding the required number of annotations for a
    specific label.

    :var id: Unique database ID of the label
    :var label_name: Name of the label
    :var label_color: Color of the label
    :var value: Required number of annotations for this label
    """

    id: str
    label_name: str
    label_color: str
    value: int


@attr.define
class AnnotationRequirements:
    """
    Container holding the required number of annotations before auto-training can be
    triggered for a task in GETi.

    :var value: Total number of required annotations for the task
    :var details: Required annotations per label
    """

    details: list[LabelAnnotationRequirements]
    value: int


@attr.define
class TaskStatus:
    """
    Status of a single task in GETi.

    :var id: Unique database ID of the task
    :var is_training: True if a training job is currently running for the task
    :var required_annotations: AnnotationRequirements object that holds details
        related to the required number of annotations before auto-training will be
        started for the task
    :var status: StatusSummary object that contains (among others) a human readable
        message describing the status of the task
    :var title: Title of the taks
    :var n_new_annotations: Number of new annotations that have been made for this
        task since its last training round. Only used in Geti v1.1 and up
    """

    id: str
    is_training: bool
    required_annotations: AnnotationRequirements
    status: StatusSummary
    title: str
    n_new_annotations: int | None = None  # Added in Geti v1.1
    ready_to_train: bool | None = None  # Added in Geti v1.4

    def __attrs_post_init__(self):
        """
        Make sure task status message is set correctly
        """
        if self.is_training:
            self.status.user_friendly_message = "Training"
        elif self.required_annotations.value != 0:
            self.status.user_friendly_message = "Waiting for user annotations"


@attr.define
class ProjectStatus:
    """
    Status of a project in GETi.

    :param is_training: True if a training job is currently running for any of the
        tasks in the project
    :var n_required_annotations: Total number of required annotations for the project,
        before auto-training can be started
    :var status: StatusSummary object that contains (among others) a human readable
        message describing the status of the project
    :var tasks: List of TaskStatus objects, detailing the status of each task in the
        project
    :var n_new_annotations: Only used in Geti v1.1
    """

    is_training: bool
    n_required_annotations: int
    status: StatusSummary
    tasks: list[TaskStatus]
    n_new_annotations: int | None = None  # Added in Geti v1.1
    project_performance: Performance | None = None
    n_running_jobs: int | None = None
    n_running_jobs_project: int | None = None

    def __attrs_post_init__(self):
        """
        Make sure task status message is set correctly
        """
        if self.is_training:
            self.status.user_friendly_message = "Training"
        elif self.n_required_annotations != 0:
            self.status.user_friendly_message = "Waiting for user annotations"

    @property
    def summary(self) -> str:
        """
        Return a string that gives a very brief summary of the project status.

        :return: String holding a brief summary of the project status
        """
        summary_str = f"Project status:\n  {self.status.user_friendly_message}\n"
        for task in self.tasks:
            summary_str += f"    Task: {task.title}\n      State: {task.status.user_friendly_message}\n"
            if task.is_training or task.status.progress != -1.0:
                summary_str += f"      Progress: {task.status.progress:.1f}%\n"
        if self.project_performance.score is not None:
            summary_str += f"  Latest score: {self.project_performance.score * 100:.1f}%\n"
        return summary_str
