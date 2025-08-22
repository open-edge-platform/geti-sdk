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

from geti_sdk.data_models import Project
from geti_sdk.data_models.configuration_models.project_configuration import (
    ProjectConfiguration,
)
from geti_sdk.http_session import GetiSession
from geti_sdk.rest_converters.project_configuration_rest_converter import (
    ProjectConfigurationRESTConverter,
)


class ProjectConfigurationClient:
    """
    REST client for managing project-level configuration settings.

    This client provides methods to retrieve and update project configuration,
    including task configurations and auto-training settings for all tasks
    within a project.
    """

    def __init__(self, workspace_id: str, project: Project, session: GetiSession):
        self.session = session
        project_id = project.id
        self.project = project
        self.workspace_id = workspace_id
        self.base_url = f"workspaces/{workspace_id}/projects/{project_id}/project_configuration"

    def get_configuration(self) -> ProjectConfiguration:
        """
        Retrieve the complete project configuration.

        Fetches the current project configuration including all task configurations,
        auto-training settings, and other project-level configuration parameters.

        :return: ProjectConfiguration object containing all project-level settings
        """
        config_rest = self.session.get_rest_response(url=self.base_url, method="GET")
        return ProjectConfigurationRESTConverter.project_configuration_from_rest(config_rest)

    def set_project_auto_train(self, auto_train: bool = False) -> None:
        """
        Enable or disable auto-training for all tasks in the project.

        Updates the auto_training.enable setting for every task configuration
        in the project to the specified value. This is a convenience method
        to bulk update auto-training settings across all tasks.

        :param auto_train: True to enable auto-training for all tasks, False to disable
        :return: None
        """
        project_configuration = self.get_configuration()
        for task_config in project_configuration.task_configs:
            task_config.auto_training.enable = auto_train
        self.set_configuration(project_configuration)

    def set_configuration(self, configuration: ProjectConfiguration) -> None:
        """
        Update the project configuration with new settings.

        Applies the provided project configuration to the project, updating all
        configuration parameters including task configurations and auto-training
        settings. Changes will affect the behavior of all tasks in the project.

        :param configuration: ProjectConfiguration object containing the new configuration
                            settings to apply to the project
        :return: None
        """
        config_rest = ProjectConfigurationRESTConverter.project_configuration_to_rest(configuration)
        self.session.get_rest_response(url=self.base_url, method="PATCH", data=config_rest)
