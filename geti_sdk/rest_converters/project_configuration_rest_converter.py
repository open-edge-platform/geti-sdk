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
    Converters between ProjectConfiguration models and their corresponding REST views
    """

    @classmethod
    def task_config_to_rest(cls, task_config: TaskConfig) -> dict:
        """
        Get the REST view of a task configuration

        :param task_config: Task configuration object
        :return: REST view of the task configuration
        """
        return {
            "task_id": task_config.task_id,
            "training": cls.configurable_parameters_to_rest(task_config.training),
            "auto_training": cls.configurable_parameters_to_rest(
                task_config.auto_training
            ),
        }

    @classmethod
    def project_configuration_to_rest(
        cls, project_configuration: ProjectConfiguration
    ) -> dict:
        """
        Get the REST view of a project configuration

        :param project_configuration: Project configuration object
        :return: REST view of the project configuration
        """
        rest_view = {
            "task_configs": [
                cls.task_config_to_rest(task_config)
                for task_config in project_configuration.task_configs
            ],
        }
        return rest_view

    @classmethod
    def project_configuration_from_rest(cls, rest_input: dict) -> ProjectConfiguration:
        """
        Convert a REST input to a ProjectConfiguration object.

        :param rest_input: REST input dictionary
        :return: ProjectConfiguration object
        """
        rest_input = deepcopy(rest_input)
        task_configs = []
        for task_data in rest_input.pop("task_configs", {}):
            task_configs.append(cls.configurable_parameters_from_rest(task_data))

        return ProjectConfiguration.model_validate(
            {"task_configs": task_configs} | rest_input
        )
