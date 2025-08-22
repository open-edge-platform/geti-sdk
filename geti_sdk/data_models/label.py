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
from typing import ClassVar

import attr

from geti_sdk.data_models.enums.domain import Domain


@attr.define
class LabelSource:
    """
    Representation of a source for a ScoredLabel in GETi

    :var user_id: ID of the user who assigned the label, if any
    :var model_id: ID of the model which generated the label, if any
    :var model_storage_id: ID of the model storage to which the model belongs
    """

    user_id: str | None = None
    model_id: str | None = None
    model_storage_id: str | None = None


@attr.define
class Label:
    """
    Representation of a Label in GETi.

    :var name: Name of the label
    :var id: Unique database ID of the label
    :var color: Color (in hex representation) of the label
    :var group: Name of the label group to which the label belongs
    :var is_empty: True if the label represents an empty label, False otherwise
    :var parent_id: Optional name of the parent label, if any
    """

    _identifier_fields: ClassVar[list[str]] = ["id", "hotkey"]
    _GET_only_fields: ClassVar[list[str]] = ["is_empty", "is_anomalous"]

    name: str
    color: str
    group: str
    is_empty: bool
    hotkey: str = ""
    domain: Domain | None = None
    id: str | None = None
    parent_id: str | None = None
    is_anomalous: bool | None = None
    is_background: bool | None = None

    def __key(self) -> tuple[str, str]:
        """
        Return a tuple representing the key of the label.

        The key is a tuple containing the name and color of the label.

        :return: A tuple representing the key of the label.
        """
        return (self.name, self.color)

    def __hash__(self) -> int:
        """
        Calculate the hash value of the object.

        :return: The hash value of the object.
        """
        return hash(self.__key())

    def prepare_for_post(self) -> None:
        """
        Set all fields to None that are not valid for making a POST request to the
        /projects endpoint.

        :return:
        """
        for field_name in self._GET_only_fields:
            setattr(self, field_name, None)

    @property
    def color_tuple(self) -> tuple[int, int, int]:
        """
        Return the color of the label as an RGB tuple.

        :return:
        """
        hex_color_str = copy.deepcopy(self.color).strip("#")
        return tuple(int(hex_color_str[i : i + 2], 16) for i in (0, 2, 4))


@attr.define
class ScoredLabel:
    """
    Representation of a Label with an assigned probability in GETi.

    :var name: Name of the label
    :var id: Unique database ID of the label
    :var color: Color (in hex representation) of the label
    :var probability:
    :var source:
    """

    _identifier_fields: ClassVar[list[str]] = ["id"]

    probability: float = attr.field(converter=float)  # float converter here to make
    # sure we're storing probability
    # as a float64 dtype
    name: str | None = None
    color: str | None = None
    id: str | None = None
    source: LabelSource | None = None

    @property
    def color_tuple(self) -> tuple[int, int, int]:
        """
        Return the color of the label as an RGB tuple.

        :return:
        """
        hex_color_str = copy.deepcopy(self.color).strip("#")
        return tuple(int(hex_color_str[i : i + 2], 16) for i in (0, 2, 4))

    @classmethod
    def from_label(cls, label: Label, probability: float) -> "ScoredLabel":
        """
        Create a ScoredLabel instance from an input Label and probability score.

        :param label: Label to convert to ScoredLabel
        :param probability: probability score for the label
        :return: ScoredLabel instance corresponding to `label` and `probability`
        """
        probability = 0 if label.is_empty else probability
        return ScoredLabel(name=label.name, probability=probability, color=label.color, id=label.id)

    def __key(self) -> tuple[str, str]:
        """
        Return a tuple representing the key of the ScoredLabel.

        The key is a tuple containing the name and color of the scored label.

        :return: A tuple representing the key of the label.
        """
        return (self.name, self.color)

    def __hash__(self) -> int:
        """
        Calculate the hash value of the object.

        :return: The hash value of the object.
        """
        return hash(self.__key())
