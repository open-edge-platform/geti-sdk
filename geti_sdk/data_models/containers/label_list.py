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

from collections import UserList
from collections.abc import Sequence
from typing import Any

from geti_sdk.data_models.label import Label, ScoredLabel
from geti_sdk.data_models.project import Project
from geti_sdk.utils.serialization_helpers import deserialize_dictionary


class LabelList(UserList):
    """
    A list containing labels for an Intel® Geti™ inference model.
    """

    def __init__(self, data: Sequence[Label] | None = None):
        self.data: list[Label] = []
        if data is not None:
            super().__init__(list(data))

        self._id_mapping: dict[str, Label] = {}
        self._name_mapping: dict[str, Label] = {}
        self._generate_indices()
        self._empty_label = next((x for x in self.data if x.is_empty), None)

    @property
    def has_empty_label(self) -> bool:
        """
        Return True if the list of Labels contains an empty label
        """
        return self._empty_label is not None

    def _generate_indices(self):
        """
        Map names and ID's to Label objects to enable quick label retrieval
        """
        self._id_mapping = {x.id: x for x in self.data}
        self._name_mapping = {x.name: x for x in self.data}

    def get_by_id(self, id: str) -> Label:  # noqa: A002
        """
        Return the Label object with ID corresponding to `id`
        """
        label = self._id_mapping.get(id, None)
        if label is None:
            raise KeyError(f"Label with id `{id}` was not found in the LabelList")
        return label

    def get_by_name(self, name: str) -> Label:
        """
        Return the Label object named `name`
        """
        label = self._name_mapping.get(name, None)
        if label is None:
            raise KeyError(f"Label named `{name}` was not found in the LabelList")
        return label

    @classmethod
    def from_json(cls, input_json: list[dict[str, Any]]) -> "LabelList":
        """
        Create a LabelList object from json input. Input should be formatted as a list
        of dictionaries, each representing a single Label
        """
        label_list: list[Label] = []
        for item in input_json:
            label_list.append(deserialize_dictionary(item, Label))
        return cls(label_list)

    def create_scored_label(self, id_or_name: str, score: float) -> ScoredLabel:
        """
        Return a ScoredLabel object corresponding to the label identified by
        `id_or_name`, and with an assigned probability score of `score`

        :param id_or_name: ID or name of the Label to assign
        :param score: probability score of the label
        """
        try:
            label = self.get_by_id(id_or_name)
        except KeyError:
            label = self.get_by_name(id_or_name)
        return ScoredLabel.from_label(label, probability=score)

    def get_empty_label(self) -> Label | None:
        """
        Return the empty label, if the LabelList contains one. If not, return None
        """
        return self._empty_label

    def get_non_empty_labels(self) -> "LabelList":
        """
        Return all non-empty labels
        """
        return LabelList([x for x in self.data if not x.is_empty])

    @classmethod
    def from_project(cls, project: Project, task_index: int = 0) -> "LabelList":
        """
        Create a LabelList object for the 'project', corresponding to the trainable
        task addressed by `task_index`

        :param project: Project for which to get the list of labels
        :param task_index: Index of the task for which to get the list of labels.
            Defaults to 0, i.e. the first trainable task in the project
        """
        task = project.pipeline.trainable_tasks[task_index]
        return cls(task.labels)

    def sort_by_ids(self, label_ids: list[str]):
        """
        Sort the labels in the LabelList by their ID, according to the order defined
        in `label_ids`
        """
        new_data: list[Label] = []
        for label_id in label_ids:
            if label_id is None:
                # Certain models have the label id as 'None' to signal an empty label
                if self.get_empty_label() is not None:
                    new_data.append(self.get_empty_label())
                else:
                    continue
            new_data.append(self.get_by_id(label_id))
        self.data = new_data
