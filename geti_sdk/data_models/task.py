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

import copy
from pprint import pformat
from typing import Any, ClassVar

import attr

from geti_sdk.data_models.enums.task_type import (
    ANOMALY_TASK_TYPES,
    GLOBAL_TASK_TYPES,
    NON_TRAINABLE_TASK_TYPES,
    TaskType,
)
from geti_sdk.data_models.label import Label
from geti_sdk.data_models.utils import (
    attr_value_serializer,
    deidentify,
    remove_null_fields,
    str_to_task_type,
)


@attr.define
class Task:
    """
    Representation of a Task in GETi.

    :var title: Title of the task
    :var task_type: Type of the task
    :var labels: List of labels belonging to the task
    :var id: unique database ID of the task
    :var label_schema_id: unique database ID of the label schema for the task
    :var keypoint_structure: The keypoint structure for the keypoint detection task
    """

    _identifier_fields: ClassVar[list[str]] = ["id", "label_schema_id"]

    title: str
    task_type: str = attr.field(converter=str_to_task_type)
    labels: list[Label] | None = None
    label_schema_id: str | None = None
    id: str | None = None
    keypoint_structure: dict | None = None

    @property
    def type(self) -> TaskType:
        """
        Return the type of the task.

        This property is here to make sure that the type of task_type is derived
        correctly. The attribute self.task_type can be instantiated both as a string
        and as a TaskType

        :return: type of the task
        """
        task_type = self.task_type
        if not isinstance(task_type, TaskType):
            task_type = TaskType(task_type)
        return task_type

    @property
    def is_trainable(self) -> bool:
        """
        Return True if this Task represents a trainable task.

        :return: True if the task is trainable, False otherwise
        """
        return self.type not in NON_TRAINABLE_TASK_TYPES

    @property
    def is_global(self) -> bool:
        """
        Return True if this Task represents a trainable task that produces global
        labels, False otherwise.

        :return: True if the task produces global labels, False otherwise
        """
        return self.type in GLOBAL_TASK_TYPES

    @property
    def is_anomaly(self) -> bool:
        """
        Return True if this task is an anomaly task.

        :return: True if task is an anomaly type task, False otherwise
        """
        return self.type in ANOMALY_TASK_TYPES

    def deidentify(self) -> None:
        """
        Set all identifier fields for the task to None.
        """
        deidentify(self)
        if self.is_trainable:
            for label in self.labels:
                deidentify(label)

    def get_label_names(self, include_empty: bool = True) -> list[str]:
        """
        Return a list of label names for the task.

        :param include_empty: True to include the empty label (if present), False to
            exclude it
        :return: List of label names for the task
        """
        if include_empty:
            labels = [label.name for label in self.labels]
        else:
            labels = [label.name for label in self.labels if not label.is_empty]
        return labels

    def to_dict(self) -> dict[str, Any]:
        """
        Return the dictionary representation of the task.

        :return:
        """
        return attr.asdict(self, recurse=True, value_serializer=attr_value_serializer)

    @property
    def overview(self) -> str:
        """
        Return a string that shows an overview of the task. This still shows all
        the detailed information of the task. If less details are required, please
        use the `summary` property

        :return: String holding an overview of the project
        """
        deidentified = copy.deepcopy(self)
        deidentified.deidentify()
        overview_dict = deidentified.to_dict()
        remove_null_fields(overview_dict)
        return pformat(overview_dict)

    @property
    def summary(self) -> str:
        """
        Return a string that gives a very brief summary of the task. This is the
        least detailed representation of the task, if more details are required
        please use the `overview` property

        :return: String holding a brief summary of the task
        """
        summary_str = f"Task: {self.title}\n  Type: {self.type} \n  Labels:\n"
        for label in self.labels:
            summary_str += f"    Name: {label.name},  Group: {label.group},  Parent: {label.parent_id}\n"
        return summary_str

    def prepare_for_post(self) -> None:
        """
        Set all fields to None that are not valid for making a POST request to the
        /projects endpoint.

        :return:
        """
        if self.labels is not None:
            labels_indices_to_pop: list[int] = []
            for ii, label in enumerate(self.labels):
                if label.is_empty:
                    labels_indices_to_pop.append(ii)
                label.prepare_for_post()
            for index in labels_indices_to_pop:
                # Empty labels are not allowed to be specified explicitly in a POST
                # request
                self.labels.pop(index)
