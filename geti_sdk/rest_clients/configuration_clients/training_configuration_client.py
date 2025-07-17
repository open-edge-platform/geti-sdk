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

from geti_sdk.data_models.configuration_models.training_configuration import (
    TrainingConfiguration,
)
from geti_sdk.http_session import GetiSession

from geti_sdk.rest_converters.training_configuration_rest_converter import (
    TrainingConfigurationRESTConverter,
)


class TrainingConfigurationClient:
    """
    Class to manage configuration for an algorithm.
    """

    def __init__(self, workspace_id: str, project: Project, session: GetiSession):
        self.session = session
        project_id = project.id
        self.project = project
        self.workspace_id = workspace_id
        self.base_url = (
            f"workspaces/{workspace_id}/projects/{project_id}/training_configuration"
        )

    def get_configuration(self, model_manifest_id: str) -> TrainingConfiguration:
        """Return the project configuration."""
        url = f"{self.base_url}?model_manifest_id={model_manifest_id}"
        config_rest = self.session.get_rest_response(url=url, method="GET")
        config_rest["model_manifest_id"] = model_manifest_id
        return TrainingConfigurationRESTConverter.training_configuration_from_rest(
            config_rest
        )

    def set_configuration(self, configuration: TrainingConfiguration) -> None:
        """
        Set the configuration for the project. This method accepts either a
        FullConfiguration, TaskConfiguration or GlobalConfiguration object

        :param configuration: Configuration to set
        :return:
        """
        config_rest = TrainingConfigurationRESTConverter.training_configuration_to_rest(
            configuration
        )
        self.session.get_rest_response(
            url=self.base_url, method="PATCH", data=config_rest
        )
