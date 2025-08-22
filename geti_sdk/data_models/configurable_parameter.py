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

from pprint import pformat
from typing import Any, ClassVar

import attr

from geti_sdk.data_models.enums.configuration_enums import (
    ParameterDataType,
    ParameterInputType,
)
from geti_sdk.data_models.utils import (
    attr_value_serializer,
    remove_null_fields,
    str_to_enum_converter,
)

DEPRECATED_PARAMETERS = ["auto_hpo_state", "auto_hpo_value"]


@attr.define
class ConfigurableParameter:
    """
    Representation of a generic configurable parameter in GETi.

    :var data_type: Data type for the parameter. Can be integer, float, string or
        boolean
    :var default_value: Default value for the parameter
    :var description: Human readable description of the parameter
    :var editable: Boolean indicating whether this parameter is editable (True) or not
        (False)
    :var header: Human readable name for the parameter
    :var name: system name for the parameter
    :var template_type: Indicates whether the parameter takes free input (`input`)
        or the value has to be selected from a list of options (`selectable`)
    :var value: The current value for the parameter
    :var ui_rules: Dictionary representing rules for logic processing in the UI,
        based on parameter values
    :var warning: Optional warning message pointing out possible risks of changing the
        parameter
    """

    _identifier_fields: ClassVar[list[str]] = []
    _non_minimal_fields: ClassVar[list[str]] = [
        "default_value",
        "description",
        "editable",
        "header",
        "warning",
        "ui_rules",
    ]

    name: str
    value: str | bool | float | int
    data_type: str | None = attr.field(default=None, converter=str_to_enum_converter(ParameterDataType))
    default_value: str | bool | float | int | None = None
    description: str | None = None
    editable: bool | None = None
    header: str | None = None
    template_type: str | None = attr.field(default=None, converter=str_to_enum_converter(ParameterInputType))
    ui_rules: dict[str, Any] | None = None
    warning: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Return the dictionary representation of the ConfigurableParameter object.
        """
        return attr.asdict(self, recurse=True, value_serializer=attr_value_serializer)

    @property
    def summary(self) -> str:
        """
        Return a string containing a very brief summary of the ConfigurableParameter
        object.

        :return: string holding a very short summary of the ConfigurableParameter
        """
        return f"{self.header} -- Name: {self.name} -- Value: {self.value}"

    @property
    def overview(self) -> str:
        """
        Return a string that shows an overview of the configurable parameter. This
        still shows all the metadata of the parameter. If less details are required,
        please use the `summary` property

        :return: String holding an overview of the configurable parameter
        """
        overview_dict = self.to_dict()
        remove_null_fields(overview_dict)
        overview_dict.pop("ui_rules")
        return pformat(overview_dict)


@attr.define
class ConfigurableBoolean(ConfigurableParameter):
    """
    Representation of a configurable boolean in GETi.
    """

    default_value: bool | None = attr.field(default=None, kw_only=True)
    value: bool = attr.field(kw_only=True)


@attr.define
class ConfigurableInteger(ConfigurableParameter):
    """
    Representation of a configurable integer in GETi.

    :var min_value: Minimum value allowed to be set for the configurable integer
    :var max_value: Maximum value allowed to be set for the configurable integer
    """

    default_value: int | None = attr.field(default=None, kw_only=True)
    value: int = attr.field(kw_only=True)
    min_value: int | None = attr.field(default=None, kw_only=True)
    max_value: int | None = attr.field(default=None, kw_only=True)


@attr.define
class ConfigurableFloat(ConfigurableParameter):
    """
    Representation of a configurable float in GETi.

    :var min_value: Minimum value allowed to be set for the configurable float
    :var max_value: Maximum value allowed to be set for the configurable float
    """

    default_value: float | None = attr.field(kw_only=True, default=None)
    value: float = attr.field(kw_only=True)
    min_value: float = attr.field(kw_only=True)
    max_value: float = attr.field(kw_only=True)
    step_size: float | None = attr.field(kw_only=True, default=None)  # Added in Geti v1.6


@attr.define
class SelectableFloat(ConfigurableParameter):
    """
    Representation of a float selectable configurable parameter in GETi.

    :var options: List of options that the selectable float is allowed to take
    """

    default_value: float | None = attr.field(kw_only=True, default=None)
    value: float = attr.field(kw_only=True)
    options: list[float] = attr.field(kw_only=True)


@attr.define
class SelectableString(ConfigurableParameter):
    """
    Representation of a string selectable configurable parameter in GETi.

    :var options: List of options that the selectable string is allowed to take
    """

    default_value: str | None = attr.field(kw_only=True, default=None)
    enum_name: str = attr.field(kw_only=True)
    value: str = attr.field(kw_only=True)
    options: list[str] = attr.field(kw_only=True)
