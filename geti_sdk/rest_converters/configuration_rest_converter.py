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
from typing import Any

import attr

from geti_sdk.data_models.configurable_parameter_group import ParameterGroup
from geti_sdk.data_models.configuration import (
    ConfigurableParameters,
    FullConfiguration,
    GlobalConfiguration,
    TaskConfiguration,
)
from geti_sdk.data_models.configuration_identifiers import (
    ComponentEntityIdentifier,
    EntityIdentifier,
    HyperParameterGroupIdentifier,
)
from geti_sdk.data_models.enums.configuration_enums import ConfigurationEntityType
from geti_sdk.data_models.utils import remove_null_fields
from geti_sdk.utils import deprecate


@deprecate(
    since="2.12",
    use="geti_sdk.rest_converters.configurable_parameters_rest_converter",
    reason=(
        "This legacy converter is deprecated in favor of ConfigurableParametersRESTConverter "
        "which provides support for the new project and training configuration REST endpoints, "
        "introduced in Geti server version 2.12 and above."
    ),
)
class ConfigurationRESTConverter:
    """
    Class that handles conversion of Intel® Geti™ REST output for configurable parameter
    entities to objects and vice versa.
    """

    @staticmethod
    def entity_identifier_from_dict(input_dict: dict[str, Any]) -> EntityIdentifier:
        """
        Create an EntityIdentifier object from an input dictionary.

        :param input_dict: Dictionary representing an EntityIdentifier in Intel® Geti™
        :return: EntityIdentifier object corresponding to the data in `input_dict`
        """
        identifier_type = input_dict.get("type")
        if isinstance(identifier_type, str):
            identifier_type = ConfigurationEntityType(identifier_type)
        if identifier_type == ConfigurationEntityType.HYPER_PARAMETER_GROUP:
            identifier_class = HyperParameterGroupIdentifier
        elif identifier_type == ConfigurationEntityType.COMPONENT_PARAMETERS:
            identifier_class = ComponentEntityIdentifier
        else:
            raise ValueError(
                f"Invalid entity identifier type found: Entity identifier of type {identifier_type} is not supported."
            )
        return identifier_class(**input_dict)

    @staticmethod
    def from_dict(input_dict: dict[str, Any]) -> ConfigurableParameters:
        """
        Create a ConfigurableParameters object holding the configurable parameters
        for an entity in the Intel® Geti™ platform, from a dictionary returned by the
        /configuration REST endpoints.

        :param input_dict: Dictionary containing the configurable parameters
        :return: ConfigurableParameters instance holding the parameter data
        """
        input_copy = copy.deepcopy(input_dict)
        entity_identifier = input_copy.pop("entity_identifier")

        if not isinstance(entity_identifier, EntityIdentifier):
            entity_identifier = ConfigurationRESTConverter.entity_identifier_from_dict(input_dict=entity_identifier)

        input_copy["entity_identifier"] = entity_identifier
        return ConfigurableParameters.from_dict(input_dict=input_copy)

    @staticmethod
    def _rest_components_to_objects(
        input_list: list[dict[str, Any]],
    ) -> list[ConfigurableParameters]:
        """
        Create a list of configurable parameters from a list of dictionaries received
        by the Intel® Geti™ /configuration endpoints.

        :param input_list: List of dictionaries to convert
        :return: List of ConfigurableParameters instances
        """
        component_objects: list[ConfigurableParameters] = []
        for component in input_list:
            if not isinstance(component, ConfigurableParameters):
                component_objects.append(ConfigurationRESTConverter.from_dict(component))
            else:
                component_objects.append(component)
        return component_objects

    @staticmethod
    def _remove_non_minimal_fields(configurable_parameters: ParameterGroup):
        """
        For all parameters in the parameter group, set the fields that are not part
        of the minimal representation of the parameter to 'None'.

        NOTE: This method modifies the input in place

        :param configurable_parameters: ConfigurableParameters instance to convert to
            minimal representation
        :return:
        """
        # Set non minimal fields of ConfigurableParameters object to None
        for field in attr.fields(type(configurable_parameters)):
            if field.name in configurable_parameters._non_minimal_fields:
                setattr(configurable_parameters, field.name, None)
        # Set non minimal fields of individual parameters to None
        for parameter in configurable_parameters.parameters:
            for field in attr.fields(type(parameter)):
                if field.name in parameter._non_minimal_fields:
                    setattr(parameter, field.name, None)
        # Set groups to None if no groups, else remove redundant fields from all groups
        if not configurable_parameters.groups:
            configurable_parameters.groups = None
        else:
            for group in configurable_parameters.groups:
                ConfigurationRESTConverter._remove_non_minimal_fields(group)

    @staticmethod
    def task_configuration_from_dict(input_dict: dict[str, Any]) -> TaskConfiguration:
        """
        Create a TaskConfiguration object holding all configurable parameters for a
        task in an Intel�� Geti™ project, from a dictionary returned by the
        /configuration REST endpoints.

        :param input_dict: Dictionary containing the configurable parameters for the
            task
        :return: TaskConfiguration instance holding the parameter data
        """
        input_copy = copy.deepcopy(input_dict)
        components = input_copy.pop("components")
        component_objects = ConfigurationRESTConverter._rest_components_to_objects(components)
        input_copy.update({"components": component_objects})
        return TaskConfiguration(**input_copy)

    @staticmethod
    def configuration_to_minimal_dict(
        configuration: TaskConfiguration | GlobalConfiguration | FullConfiguration,
        deidentify: bool = True,
    ) -> dict[str, Any]:
        """
        Convert a TaskConfiguration, GlobalConfiguration or FullConfiguration into a
        dictionary, removing fields that are None or are only relevant to the
        Intel® Geti™ UI.

        :param configuration: TaskConfiguration or GlobalConfiguration to convert
        :param deidentify: True to remove all unique database identifiers, False to
            preserve identifiers. Defaults to True
        :return: Dictionary representation of the configuration
        """
        input_copy = copy.deepcopy(configuration)
        if deidentify:
            input_copy.deidentify()
        if not isinstance(configuration, FullConfiguration):
            for config in input_copy.components:
                ConfigurationRESTConverter._remove_non_minimal_fields(config)
        else:
            for config in input_copy.global_.components:
                ConfigurationRESTConverter._remove_non_minimal_fields(config)
            for task_config in input_copy.task_chain:
                for config in task_config.components:
                    ConfigurationRESTConverter._remove_non_minimal_fields(config)
        result = input_copy.to_dict()
        remove_null_fields(result)
        return result

    @staticmethod
    def global_configuration_from_rest(
        input_: list[dict[str, Any]] | dict[str, Any],
    ) -> GlobalConfiguration:
        """
        Create a GlobalConfiguration object holding the configurable parameters
        for all project-wide components in the Intel® Geti™ project, from input from the
        /configuration/global REST endpoint.

        :param input_: REST response holding the serialized configurable parameters
        :return:
        """
        input_copy = copy.deepcopy(input_)
        input_list = input_copy if isinstance(input_copy, list) else input_copy.pop("items")
        component_objects = ConfigurationRESTConverter._rest_components_to_objects(input_list)
        if isinstance(input_copy, list):
            return GlobalConfiguration(components=component_objects)
        input_copy.update({"components": component_objects})
        return GlobalConfiguration(**input_copy)

    @staticmethod
    def full_configuration_from_rest(input_dict: dict[str, Any]) -> FullConfiguration:
        """
        Convert a dictionary holding the full configuration for an Intel® Geti™
        project, as returned by the /configuration endpoint, to an object
        representation.

        :param input_dict: Dictionary representing the full project configuration
        :return: FullConfiguration instance holding the global and task chain
            configuration
        """
        global_dict = input_dict.pop("global")
        task_chain_list = input_dict.pop("task_chain")
        global_config = ConfigurationRESTConverter.global_configuration_from_rest(global_dict)
        task_chain_config = [
            ConfigurationRESTConverter.task_configuration_from_dict(task_config) for task_config in task_chain_list
        ]
        return FullConfiguration(global_=global_config, task_chain=task_chain_config)

    @staticmethod
    def configurable_parameter_list_to_rest(
        configurable_parameter_list: list[ConfigurableParameters],
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Convert a list of model hyper parameters to a dictionary that can be sent to
        the /configuration POST endpoints.

        :param configurable_parameter_list: List of ConfigurableParameter instances
        :return: Dictionary containing:
            - 'components': list of dictionaries representing configurable parameters,
                            that are conforming to the /configuration REST endpoints
        """
        rest_parameters: list[dict[str, Any]] = []
        for parameter_set in configurable_parameter_list:
            parameter_copy = copy.deepcopy(parameter_set)
            ConfigurationRESTConverter._remove_non_minimal_fields(parameter_copy)
            parameter_dict = parameter_copy.to_dict()
            remove_null_fields(parameter_dict)
            rest_parameters.append(parameter_dict)
        return {"components": rest_parameters}
