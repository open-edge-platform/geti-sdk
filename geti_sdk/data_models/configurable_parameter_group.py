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
from collections.abc import Sequence
from typing import Any, ClassVar, Optional, Union, get_args

import attr

from geti_sdk.data_models.configurable_parameter import (
    DEPRECATED_PARAMETERS,
    ConfigurableBoolean,
    ConfigurableFloat,
    ConfigurableInteger,
    ConfigurableParameter,
    SelectableFloat,
    SelectableString,
)
from geti_sdk.data_models.enums.configuration_enums import (
    ConfigurableParameterType,
    ParameterDataType,
    ParameterInputType,
)
from geti_sdk.data_models.utils import (
    attr_value_serializer,
    deidentify,
    str_to_enum_converter,
)

PARAMETER_TYPES = Union[  # noqa: UP007
    SelectableFloat,
    SelectableString,
    ConfigurableFloat,
    ConfigurableInteger,
    ConfigurableBoolean,
]


def _parameter_dicts_to_list(
    parameter_dicts: list[dict[str, Any] | PARAMETER_TYPES],
) -> list[PARAMETER_TYPES]:
    """
    Convert a list of dictionary representations of configurable parameters to a
    list of ConfigurableParameter objects.

    :param parameter_dicts: List of dictionaries, each entry representing a single
        configurable parameter
    :return: List of corresponding ConfigurableParameter objects
    """
    parameters: list[PARAMETER_TYPES] = []
    for parameter in parameter_dicts:
        if isinstance(parameter, get_args(PARAMETER_TYPES)):
            parameters.append(parameter)
            continue
        for deprecated_key in DEPRECATED_PARAMETERS:
            parameter.pop(deprecated_key, None)
        data_type = parameter.get("data_type", None)
        template_type = parameter.get("template_type", None)
        if data_type is None or template_type is None:
            raise ValueError(
                f"Unable to reconstruct ParameterGroup object from input: "
                f"{parameter_dicts}. No data or template type found for "
                f"parameter {parameter}"
            )
        data_type = ParameterDataType(data_type)
        template_type = ParameterInputType(template_type)
        parameter_type: type[ConfigurableParameter]
        if data_type == ParameterDataType.STRING and template_type == ParameterInputType.SELECTABLE:
            parameter_type = SelectableString
        elif data_type == ParameterDataType.INTEGER and template_type == ParameterInputType.INPUT:
            parameter_type = ConfigurableInteger
        elif data_type == ParameterDataType.BOOLEAN and template_type == ParameterInputType.INPUT:
            parameter_type = ConfigurableBoolean
        elif data_type == ParameterDataType.FLOAT and template_type == ParameterInputType.INPUT:
            parameter_type = ConfigurableFloat
        elif data_type == ParameterDataType.FLOAT and template_type == ParameterInputType.SELECTABLE:
            parameter_type = SelectableFloat
        else:
            raise ValueError(
                f"Unable to reconstruct ParameterGroup object from input: "
                f"{parameter_dicts}. Invalid data or template type found "
                f"for parameter {parameter}"
            )
        parameters.append(parameter_type(**parameter))
    return parameters


@attr.define
class ParameterGroup:
    """
    Representation of a group of configurable parameters in GETi, as returned by the
    /configuration endpoints.

    :var header: Human readable name for the parameter group
    :var type: Type of the parameter group
    :var name: name by which the parameter group is identified in the system
    :var parameters: List of configurable parameters
    :var groups: List of parameter groups
    """

    _non_minimal_fields: ClassVar[list[str]] = ["description"]

    header: str
    type: str = attr.field(converter=str_to_enum_converter(ConfigurableParameterType))
    description: str | None = None
    parameters: Sequence[PARAMETER_TYPES] | None = None
    name: str | None = None
    groups: list["ParameterGroup"] | None = None

    def to_dict(self) -> dict[str, Any]:
        """
        Return the dictionary representation of the ParameterGroup
        """
        return attr.asdict(self, recurse=True, value_serializer=attr_value_serializer)

    @classmethod
    def from_dict(cls, input_dict: dict[str, Any]) -> "ParameterGroup":
        """
        Create a ParameterGroup object from an input dictionary.

        :param input_dict: Dictionary to create the parameter group from
        :return: ParameterGroup instance corresponding to the data received in
            input_dict
        """
        input_copy = copy.deepcopy(input_dict)
        for deprecated_key in DEPRECATED_PARAMETERS:
            input_copy.pop(deprecated_key, None)
        parameter_dicts: list[dict[str, Any] | ConfigurableParameter] = input_copy.pop("parameters", [])
        group_dicts: list[dict[str, Any] | ParameterGroup] = input_copy.pop("groups", [])

        parameters = _parameter_dicts_to_list(parameter_dicts=parameter_dicts)
        groups: list[ParameterGroup] = []
        for group_dict in group_dicts:
            if isinstance(group_dict, ParameterGroup):
                groups.append(group_dict)
                continue
            groups.append(ParameterGroup.from_dict(group_dict))
        return cls(**input_copy, parameters=parameters, groups=groups)

    def deidentify(self) -> None:
        """
        Remove all identifier fields from the ParameterGroup
        """
        for parameter in self.parameters:
            deidentify(parameter)
        for group in self.groups:
            group.deidentify()

    def parameter_names(self, get_nested: bool = True) -> list[str]:
        """
        Return a list of names of all parameters in the ParameterGroup.

        :param get_nested: True to include parameter names in nested ParameterGroups
            belonging to this group as well. False to only consider this groups'
            parameters
        :return: List of parameter names
        """
        parameter_names = [parameter.name for parameter in self.parameters]
        if get_nested:
            for group in self.groups:
                parameter_names.extend(group.parameter_names(get_nested=True))
        return parameter_names

    def get_parameter_group_by_name(self, name: str) -> Optional["ParameterGroup"]:
        """
        Return the parameter group named `name`, if it belongs to this
        ParameterGroup or any of it's nested groups.

        If no group by `name` is found, this method returns None

        :param name: Name of the parameter group to look for
        :return: ParameterGroup named `name`, if any group by that name is found.
            None otherwise
        """
        all_groups = []
        for group in self.groups:
            all_groups.extend(group.groups)
        return next((group for group in all_groups if group.name == name), None)

    def get_parameter_by_name(self, name: str, group_name: str | None = None) -> PARAMETER_TYPES | None:
        """
        Get the data for the configurable parameter named `name` from the
        ParameterGroup. This method returns None if no parameter by that name was found

        :param name: Name of the parameter to get
        :param group_name: Optional name of the parameter group to which the parameter
            belongs. If None is specified (the default), this method looks in all
            parameter groups belonging to this group.
        :return: ConfigurableParameter object representing the parameter named `name`
        """
        if name not in self.parameter_names(get_nested=True):
            return None

        if group_name is None:
            parameters = [parameter for parameter in self.parameters if parameter.name == name]
            for group in self.groups:
                parameters.extend([parameter for parameter in group.parameters if parameter.name == name])
            if len(parameters) != 1:
                raise ValueError(
                    f"Found multiple parameters named {name}, please specify a "
                    f"group_name to identify the parameter unambiguously."
                )
            parameter = parameters[0]
        else:
            group = self.get_parameter_group_by_name(group_name)
            parameter = group.get_parameter_by_name(name) if group is not None else None
        return parameter

    def get_group_containing(self, parameter_name: str) -> Optional["ParameterGroup"]:
        """
        Return the parameter group that contains the parameter specified in
        `parameter_name`. If no group containing the parameter is found, this method
        returns None

        :param parameter_name: Name of the parameter for which to retrieve the group
        :return: ParameterGroup containing the parameter by name `parameter_name`
        """
        if not self.groups or self.get_parameter_by_name(parameter_name) is None:
            return None
        group_names = [group.name for group in self.groups]
        for group_name in group_names:
            if self.get_parameter_by_name(parameter_name, group_name) is not None:
                return self.get_parameter_group_by_name(group_name)
        return None

    @property
    def summary(self) -> str:
        """
        Return a string containing a very brief summary of the ParameterGroup.

        :return: string holding a very short summary of the ParameterGroup
        """
        summary_str = f"{self.header}"
        for parameter in self.parameters:
            summary_str += f"\n  {parameter.summary}"
        for group in self.groups:
            summary_str += f"\n  {group.summary}"
        summary_str += "\n"
        return summary_str
