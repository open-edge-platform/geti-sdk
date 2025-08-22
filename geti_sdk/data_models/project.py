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

from .dataset import Dataset
from .label import Label
from .performance import Performance
from .task import Task
from .utils import (
    attr_value_serializer,
    deidentify,
    remove_null_fields,
    str_to_datetime,
)


@attr.define
class Connection:
    """
    Representation of a connection between two tasks in GETi.

    :var to: Name of the task to which the connection is leading
    :var from_: Name of the task from which the connection originates
    """

    to: str
    from_: str


@attr.define
class Pipeline:
    """
    Representation of a pipeline for a project in GETi.

    :var tasks: List of tasks in the pipeline
    :var connections: List of connections between the tasks in the pipeline
    """

    tasks: list[Task]
    connections: list[Connection]

    @property
    def trainable_tasks(self) -> list[Task]:
        """
        Return an ordered list of trainable tasks.

        :return: List of trainable tasks in the pipeline
        """
        return [task for task in self.tasks if task.is_trainable]

    def resolve_connections(self):
        """
        Replace the task id's in the connections attribute of the pipeline with the
        actual task titles.

        :return:
        """
        task_id_to_name = {task.id: task.title for task in self.tasks}
        for connection in self.connections:
            if connection.to in task_id_to_name:
                connection.to = task_id_to_name[connection.to]
            if connection.from_ in task_id_to_name:
                connection.from_ = task_id_to_name[connection.from_]

    def resolve_parent_labels(self):
        """
        Replace the parent_id's in the pipeline labels by the actual label names.

        :return:
        """
        for task in self.trainable_tasks:
            for label in task.labels:
                if label.parent_id in self.label_id_to_name_mapping:
                    label.parent_id = self.label_id_to_name_mapping[label.parent_id]

    @property
    def label_id_to_name_mapping(self) -> dict[str, str]:
        """
        Return a mapping of label ID's to label names for all labels in the pipeline.

        :return: dictionary containing the label ID's as keys and the label names as
            values
        """
        label_mapping: dict[str, str] = {}
        for task in self.trainable_tasks:
            label_mapping.update({label.id: label.name for label in task.labels})
        return label_mapping

    def get_labels_per_task(self, include_empty: bool = True) -> list[list[Label]]:
        """
        Return a nested list of labels for each task in the pipeline.

        Each entry in the outermost list corresponds to a task in the pipeline. The
        innermost list holds the labels for that specific task

        :param include_empty: True to include empty labels in the output
        :return: nested list of labels for each task in the pipeline
        """
        outer_list: list[list[Label]] = []
        for task in self.trainable_tasks:
            task_labels = [label for label in task.labels if not label.is_empty] if not include_empty else task.labels
            outer_list.append(task_labels)
        return outer_list

    def get_all_labels(self) -> list[Label]:
        """
        Return a list of all labels in the pipeline.

        :return: List of all labels for every task in the pipeline
        """
        label_list: list[Label] = []
        for task in self.trainable_tasks:
            label_list.extend(task.labels)
        return label_list

    def deidentify(self) -> None:
        """
        Remove all unique database ID's from the tasks and labels in the pipeline.
        """
        self.resolve_connections()
        self.resolve_parent_labels()
        for task in self.tasks:
            task.deidentify()

    def prepare_for_post(self) -> None:
        """
        Set all fields to None that are not valid for making a POST request to the
        /projects endpoint.

        :return:
        """
        for task in self.tasks:
            task.prepare_for_post()


@attr.define
class Project:
    """
    Representation of a project in Intel® Geti™.

    :var id: Unique database ID of the project
    :var name: Name of the project
    :var creation_time: Time at which the project was created
    :var pipeline: Pipeline for the project
    :var datasets: List of datasets belonging to the project
    :var score: Score achieved by the AI assistant for the project
    :var thumbnail: URL at which a thumbnail for the project can be obtained
    :var storage_info: Storage information for the project can be obtained
    """

    _identifier_fields: ClassVar[list[str]] = [
        "id",
        "thumbnail",
        "creation_time",
        "creator_id",
    ]
    _GET_only_fields: ClassVar[list[str]] = [
        "thumbnail",
        "score",
        "performance",
        "creator_id",
        "creation_time",
        "storage_info",
    ]

    name: str
    pipeline: Pipeline
    datasets: list[Dataset] | None = None  # `datasets` was removed from project listing in Geti v1.15
    score: float | None = None  # 'score' is removed in v1.1
    performance: Performance | None = None
    creation_time: str | None = attr.field(default=None, converter=str_to_datetime)
    id: str | None = None
    thumbnail: str | None = None
    creator_id: str | None = None
    storage_info: dict | None = None

    def get_trainable_tasks(self) -> list[Task]:
        """
        Return an ordered list of trainable tasks.

        :return: List of trainable tasks in the project
        """
        return self.pipeline.trainable_tasks

    @property
    def project_type(self) -> str:
        """
        Return a string containing the project type. The type is constructed by
        concatenating the types of tasks in the task chain together, separated by a
        '_to_' sequence.

        For example, a project with a detection task followed by a segmentation task
        will have it's project type set as: 'detection_to_segmentation'

        :return: string containing the project type
        """
        tasks = self.get_trainable_tasks()
        project_type = ""
        for index, task in enumerate(tasks):
            if index > 0:
                project_type += f"_to_{task.task_type}"
            else:
                project_type += f"{task.task_type}"
        return project_type

    def deidentify(self) -> None:
        """
        Remove all unique database ID's from the project, pipeline and datasets.

        :return:
        """
        deidentify(self)
        self.pipeline.deidentify()
        for dataset in self.datasets:
            dataset.deidentify()

    def prepare_for_post(self) -> None:
        """
        Set all fields to None that are not valid for making a POST request to the
        /projects endpoint.

        :return:
        """
        for field_name in self._GET_only_fields:
            setattr(self, field_name, None)
        self.pipeline.prepare_for_post()
        for dataset in self.datasets:
            dataset.prepare_for_post()

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the project to a dictionary representation.

        :return: Dictionary holding the project data
        """
        output_dict = attr.asdict(self, value_serializer=attr_value_serializer)
        for connection in output_dict["pipeline"]["connections"]:
            from_value = connection.pop("from_")
            connection["from"] = from_value
        return output_dict

    def get_labels_per_task(self, include_empty: bool = True) -> list[list[dict[str, Any]]]:
        """
        Return a nested list containing the labels for each task in the project.
        Each entry in the outermost list corresponds to a trainable task in the project.

        :param include_empty: True to include empty labels in the output, False to
            exclude them
        :return:
        """
        return [
            [attr.asdict(label) for label in task_labels]
            for task_labels in self.pipeline.get_labels_per_task(include_empty=include_empty)
        ]

    def get_parameters(self) -> dict[str, str | list[str]]:
        """
        Return the parameters used to create the project.

        :return: Dictionary containing the keys `project_name`, `project_type` and
            `labels`
        """
        parameter_dict = {
            "project_name": self.name,
            "project_type": self.project_type,
            "labels": self.get_labels_per_task(include_empty=False),
        }
        remove_null_fields(parameter_dict)
        return parameter_dict

    def get_all_labels(self) -> list[Label]:
        """
        Return a list of all labels in the project.

        :return: List of all labels in the project
        """
        return self.pipeline.get_all_labels()

    @property
    def overview(self) -> str:
        """
        Return a string that shows an overview of the project. This still shows all
        the detailed information of the project. If less details are required, please
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
        Return a string that gives a very brief summary of the project. This is the
        least detailed representation of the project, if more details are required
        please use the `overview` property

        :return: String holding a brief summary of the project
        """
        summary_str = f"Project: {self.name}\n"
        for task_index, (task, labels) in enumerate(
            zip(self.get_trainable_tasks(), self.pipeline.get_labels_per_task())
        ):
            summary_str += f"  Task {task_index + 1}: {task.title}\n"
            if labels is not None:
                summary_str += f"    Labels: {[label.name for label in labels]}\n"
        return summary_str

    @property
    def training_dataset(self) -> Dataset:
        """
        Return a training dataset for the project.

        :return: Training dataset for the project
        """
        return next(dataset for dataset in self.datasets if dataset.use_for_training)
