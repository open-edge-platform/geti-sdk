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

from collections.abc import Sequence
from typing import ClassVar

import attr

from geti_sdk.data_models.label import Label, ScoredLabel
from geti_sdk.data_models.shapes import (
    Ellipse,
    Keypoint,
    Polygon,
    Rectangle,
    RotatedRectangle,
)
from geti_sdk.data_models.utils import deidentify, str_to_datetime


@attr.define
class Annotation:
    """
    Representation of a single annotation for a media item on the Intel® Geti™ platform.

    :var labels: List of labels belonging to the annotation
    :var modified: Date and time of the last modification made to this annotation
    :var shape: Shape of the annotation
    :var id: Unique database ID assigned to the annotation
    :var labels_to_revisit: Optional list of database ID's of the labels that may not
        be up to date and therefore need to be revisited for this annotation
    """

    _identifier_fields: ClassVar[str] = ["id", "modified"]

    labels: list[ScoredLabel]
    shape: Rectangle | Ellipse | Polygon | RotatedRectangle | Keypoint
    modified: str | None = attr.field(converter=str_to_datetime, default=None)
    id: str | None = None
    labels_to_revisit: list[str] | None = None

    def deidentify(self) -> None:
        """
        Remove all unique database ID's from the annotation and the entities it
        contains
        """
        deidentify(self)
        for label in self.labels:
            deidentify(label)

    @property
    def label_names(self) -> list[str]:
        """
        Return a list of label names for this Annotation.
        """
        return [label.name for label in self.labels]

    def append_label(self, label: ScoredLabel) -> None:
        """
        Add a label to the list of labels belonging to this annotation.

        :param label:
        :return:
        """
        self.labels.append(label)

    def extend_labels(self, labels: list[ScoredLabel]) -> None:
        """
        Add a list of labels to the labels already attached to this annotation.

        :param labels: List of ScoredLabels to add
        :return:
        """
        self.labels.extend(labels)

    def pop_label_by_name(self, label_name: str) -> None:
        """
        Remove a label from the list of labels belonging to this annotation.

        :param label_name: Name of the label to remove from the list
        :return:
        """
        index = None
        for index, label in enumerate(self.labels):
            if label.name == label_name:
                break
        if index is not None:
            self.labels.pop(index)

    def map_labels(self, labels: Sequence[ScoredLabel | Label]) -> "Annotation":
        """
        Attempt to map the labels found in `labels` to those in the Annotation
        instance. Labels are matched by name. This method will return a new
        Annotation object.

        :param labels: Labels to which the existing labels should be mapped
        :return: Annotation with updated labels, corresponding to those found in
            the `project` (if matching labels were found)
        """
        mapped_label_names = [label.name for label in labels]
        mapped_label_ids = [label.id for label in labels]

        new_labels: list[ScoredLabel] = []
        for label in self.labels:
            if label.name in mapped_label_names:
                label_index = mapped_label_names.index(label.name)
                new_labels.append(
                    ScoredLabel(
                        probability=label.probability,
                        name=label.name,
                        color=label.color,
                        id=mapped_label_ids[label_index],
                        source=label.source,
                    )
                )
        new_labels_to_revisit = [new_label.id for new_label in new_labels if new_label.id in self.labels_to_revisit]
        return Annotation(
            labels=new_labels,
            shape=self.shape,
            modified=self.modified,
            id=None,
            labels_to_revisit=new_labels_to_revisit,
        )
