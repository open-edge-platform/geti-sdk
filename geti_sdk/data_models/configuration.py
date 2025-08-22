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

from typing import Any, ClassVar

import attr

from geti_sdk.data_models import Algorithm
from geti_sdk.data_models.configurable_parameter import DEPRECATED_PARAMETERS
from geti_sdk.data_models.configurable_parameter_group import (
    PARAMETER_TYPES,
    ParameterGroup,
)
from geti_sdk.data_models.configuration_identifiers import (
    ComponentEntityIdentifier,
    HyperParameterGroupIdentifier,
)
from geti_sdk.data_models.utils import attr_value_serializer, deidentify


@attr.define
class ConfigurableParameters(ParameterGroup):
    """
    Representation of configurable parameters in GETi, as returned by the
    /configuration endpoint

    :var entity_identifier: Identification information for the entity to which the
        configurable parameters apply
    :var id: Unique database ID of the configurable parameters
    """

    _identifier_fields: ClassVar[str] = ["id"]

    entity_identifier: HyperParameterGroupIdentifier | ComponentEntityIdentifier = attr.field(kw_only=True)
    id: str | None = attr.field(default=None, kw_only=True)

    def deidentify(self) -> None:
        """
        Remove all unique database ID's from the configurable parameters
        """
        super().deidentify()
        deidentify(self)
        deidentify(self.entity_identifier)


@attr.define(slots=False)
class Configuration:
    """
    Representation of a set of configurable parameters in GETi, that apply to a project
    or task.
    """

    _identifier_fields: ClassVar[list[str]] = []

    components: list[ConfigurableParameters]

    def __attrs_post_init__(self):
        """
        Set configurable parameters as Configuration attributes
        """
        for parameter_name in self.get_all_parameter_names():
            setattr(self, parameter_name, self.get_parameter_by_name(parameter_name))

    def to_dict(self) -> dict[str, Any]:
        """
        Return the dictionary representation of the Configuration.
        """
        return attr.asdict(self, recurse=True, value_serializer=attr_value_serializer)

    def deidentify(self) -> None:
        """
        Remove all unique database ID's from the Configuration
        """
        deidentify(self)
        for config in self.components:
            config.deidentify()

    def __iter__(self):
        """
        Iterate over all parameters in the configuration.
        """
        parameter_names = self.get_all_parameter_names()
        for parameter_name in parameter_names:
            yield self.get_parameter_by_name(parameter_name)

    def get_all_parameter_names(self) -> list[str]:
        """
        Return a list of names of all configurable parameters within the task
        configuration.
        """
        parameters: list[str] = []
        for config in self.components:
            parameters.extend(config.parameter_names())
        return parameters

    def _set_parameter_value(
        self,
        parameter_name: str,
        value: bool | float | int | str,
        group_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Prepare a dictionary that can be used for setting a parameter value in GETi.

        :param parameter_name: Name of the parameter to set
        :param value: Value to set for the parameter
        :param group_name: Optional name of the parameter group in which the parameter
            lives. If left as None (the default), this method will attempt to find the
            group automatically
        :return: Dictionary that can be passed directly to the
            /configuration POST endpoints to set the parameter
            value
        """
        if parameter_name not in self.get_all_parameter_names():
            raise ValueError(
                f"Parameter named '{parameter_name}' was not found in the configuration"
                f" {self}. Unable to prepare data to set "
                f"parameter value"
            )
        result: dict[str, Any] = {}
        for config in self.components:
            parameter = config.get_parameter_by_name(parameter_name, group_name)
            if parameter is not None:
                result.update({"entity_identifier": config.entity_identifier.to_dict()})
                parameter_value_dict = {"name": parameter_name, "value": value}
                parameter.value = value
                if config.groups:
                    group = config.get_group_containing(parameter_name)
                    if group is not None:
                        result.update(
                            {
                                "groups": [
                                    {
                                        "name": group.name,
                                        "parameters": [parameter_value_dict],
                                    }
                                ]
                            }
                        )
                        break
                result.update({"parameters": [parameter_value_dict]})
                break
        return result

    def get_parameter_by_name(self, name: str) -> PARAMETER_TYPES | None:
        """
        Return the configurable parameter named `name`. If no parameter by that name
        is found within the Configuration, this method returns None

        :param name: Name of the parameter to get
        :return: ConfigurableParameter object
        """
        if name in DEPRECATED_PARAMETERS:
            return None

        parameters = [config.get_parameter_by_name(name) for config in self.components]
        parameters = [parameter for parameter in parameters if parameter is not None]
        if len(parameters) == 0:
            return None
        if len(parameters) > 1:
            raise ValueError(
                f"Multiple parameters named '{name}' were found in the configuration "
                f"{self}. Unable to unambiguously retrieve "
                f"parameter."
            )
        return parameters[0]

    @property
    def component_configurations(self) -> list[ConfigurableParameters]:
        """
        Return all configurable parameters that are component-related.

        :return:
        """
        return [config for config in self.components if isinstance(config.entity_identifier, ComponentEntityIdentifier)]

    def get_component_configuration(self, component: str) -> ConfigurableParameters | None:
        """
        Return the configurable parameters for a certain component. If
        no configuration is found for the specified component, this method returns None

        :param component: Name of the component to get the configuration for
        :return: ConfigurableParameters for the component
        """
        return next(
            (config for config in self.component_configurations if config.entity_identifier.component == component),
            None,
        )


@attr.define(slots=False)
class GlobalConfiguration(Configuration):
    """
    Representation of the project-wide configurable parameters for a project in GETi.
    """

    def set_parameter_value(
        self,
        parameter_name: str,
        value: bool | float | int | str,
        group_name: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Prepare a dictionary that can be used for setting a parameter value in GETi.

        :param parameter_name: Name of the parameter to set
        :param value: Value to set for the parameter
        :param group_name: Optional name of the parameter group in which the parameter
            lives. If left as None (the default), this method will attempt to find the
            group automatically
        :return: Dictionary that can be passed directly to the
            /configuration POST endpoints to set the parameter
            value
        """
        result = self._set_parameter_value(parameter_name, value, group_name)
        return [result]

    def apply_identifiers(self, workspace_id: str, project_id: str):
        """
        Apply the unique database identifiers passed in `workspace_id`
        and `project_id` to all configurable parameters in the GlobalConfiguration.

        :param workspace_id: Workspace ID to assign
        :param project_id: Project ID to assign
        :return:
        """
        for config in self.components:
            config.entity_identifier.workspace_id = workspace_id
            config.entity_identifier.project_id = project_id

    @property
    def summary(self) -> str:
        """
        Return a string containing a very brief summary of the GlobalConfiguration.

        :return: string holding a very short summary of the GlobalConfiguration
        """
        summary_str = "Configuration for global components:\n"
        for configurable_parameters in self.components:
            summary_str += f"  {configurable_parameters.summary}\n"
        return summary_str


@attr.define(slots=False)
class TaskConfiguration(Configuration):
    """
    Representation of the configurable parameters for a task in GETi.
    """

    _identifier_fields: ClassVar[list[str]] = ["task_id"]

    task_id: str | None = attr.field(default=None, kw_only=True)
    task_title: str = attr.field(kw_only=True)

    @property
    def model_configurations(self) -> list[ConfigurableParameters]:
        """
        Return all configurable parameters that are model-related.

        :return: List of configurable parameters
        """
        return [
            config for config in self.components if isinstance(config.entity_identifier, HyperParameterGroupIdentifier)
        ]

    def set_parameter_value(
        self,
        parameter_name: str,
        value: bool | float | int | str,
        group_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Prepare a dictionary that can be used for setting a configurable parameter
        value in GETi.

        :param parameter_name: Name of the parameter to set
        :param value: Value to set for the parameter
        :param group_name: Optional name of the parameter group in which the parameter
            lives. If left as None (the default), this method will attempt to find the
            group automatically
        :return: Dictionary that can be passed directly to the
            /configuration POST endpoints to set the parameter
            value
        """
        result = self._set_parameter_value(parameter_name, value, group_name)
        return {"components": [result]}

    def resolve_algorithm(self, algorithm: Algorithm):
        """
        Resolve the algorithm name and id of the model template for all hyper
        parameter groups in the task configuration.

        :param algorithm: Algorithm instance to which the hyper parameters belong
        :return:
        """
        for config in self.model_configurations:
            config.entity_identifier.resolve_algorithm(algorithm=algorithm)

    def apply_identifiers(self, workspace_id: str, project_id: str, task_id: str, model_storage_id: str):
        """
        Apply the unique database identifiers passed in `workspace_id`,
        `project_id`, `task_id` and `model_storage_id` to all configurable parameters
        in the TaskConfiguration.

        :param workspace_id: Workspace ID to assign
        :param project_id: Project ID to assign
        :param task_id: Task ID to assign
        :param model_storage_id: Model storage ID to assign
        :return:
        """
        self.task_id = task_id
        for config in self.component_configurations:
            config.entity_identifier.workspace_id = workspace_id
            config.entity_identifier.project_id = project_id
            config.entity_identifier.task_id = task_id
        for config in self.model_configurations:
            config.entity_identifier.workspace_id = workspace_id
            config.entity_identifier.model_storage_id = model_storage_id

    @property
    def summary(self) -> str:
        """
        Return a string containing a very brief summary of the TaskConfiguration.

        :return: string holding a very short summary of the TaskConfiguration
        """
        summary_str = f"Configuration for {self.task_title}:\n"
        for configurable_parameters in self.components:
            summary_str += f"  {configurable_parameters.summary}\n"
        return summary_str


@attr.define
class FullConfiguration:
    """
    Representation of the full configuration (both global and task-chain) for a
    project in GETi.
    """

    global_: GlobalConfiguration
    task_chain: list[TaskConfiguration]

    def deidentify(self) -> None:
        """
        Remove all unique database ID's from the Configuration.
        """
        self.global_.deidentify()
        for task_config in self.task_chain:
            task_config.deidentify()

    def to_dict(self) -> dict[str, Any]:
        """
        Return the dictionary representation of the FullConfiguration.
        """
        result = attr.asdict(self, recurse=True, value_serializer=attr_value_serializer)
        global_dict = result.pop("global_")
        result.update({"global": global_dict["components"]})
        return result

    @property
    def summary(self) -> str:
        """
        Return a string containing a very brief summary of the FullConfiguration.

        :return: string holding a very short summary of the FullConfiguration
        """
        summary_str = "Full project configuration:\n"
        summary_str += f"  {self.global_.summary}\n"
        for task_configuration in self.task_chain:
            summary_str += f"  {task_configuration.summary}\n"
        return summary_str
