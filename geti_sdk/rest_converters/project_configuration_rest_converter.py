# Copyright (C) 2025 Intel Corporation
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
from copy import deepcopy

from geti_sdk.data_models.configuration_models import ProjectConfiguration, TaskConfig
from geti_sdk.rest_converters.configurable_parameters_rest_converter import (
    ConfigurableParametersRESTConverter,
)


class ProjectConfigurationRESTConverter(ConfigurableParametersRESTConverter):
    """
    Converter class for transforming ProjectConfiguration objects to/from REST API format.

    This class provides methods to convert between ProjectConfiguration domain objects and their
    corresponding REST representations, handling task configurations and auto-training settings.
    """

    @classmethod
    def task_config_to_rest(cls, task_config: TaskConfig) -> dict:
        """
        Convert a TaskConfig object to its REST API representation.

        Transforms a TaskConfig domain object into a dictionary structure expected by the
        REST API, including task_id and configurable parameters for training and auto_training.

        :param task_config: TaskConfig object containing training and auto-training configuration
        :return: Dictionary containing task_id, training parameters, and auto_training parameters
                in REST API format
        """
        return {
            "task_id": task_config.task_id,
            "training": cls.configurable_parameters_to_rest(task_config.training),
            "auto_training": cls.configurable_parameters_to_rest(task_config.auto_training),
        }

    @classmethod
    def project_configuration_to_rest(cls, project_configuration: ProjectConfiguration) -> dict:
        """
        Convert a ProjectConfiguration object to its REST API representation.

        Transforms a ProjectConfiguration domain object into a dictionary structure expected
        by the REST API, converting all task configurations to their REST format.

        :param project_configuration: ProjectConfiguration object containing all task configurations
        :return: Dictionary containing task_configs array with each task configuration converted
                to REST format
        """
        return {
            "task_configs": [
                cls.task_config_to_rest(task_config) for task_config in project_configuration.task_configs
            ],
        }

    @classmethod
    def project_configuration_from_rest(cls, rest_input: dict) -> ProjectConfiguration:
        """
        Create a ProjectConfiguration object from REST API input.

        Parses REST API dictionary and converts it into a validated ProjectConfiguration
        domain object, transforming each task configuration from REST format to domain format.

        :param rest_input: REST API dictionary containing task_configs array and other
                          project configuration parameters
        :return: Validated ProjectConfiguration object with all task configurations converted
                from REST format
        """
        rest_input = deepcopy(rest_input)
        task_configs = []
        for task_data in rest_input.pop("task_configs", {}):
            task_configs.append(cls.configurable_parameters_from_rest(task_data))

        return ProjectConfiguration.model_validate({"task_configs": task_configs} | rest_input)
