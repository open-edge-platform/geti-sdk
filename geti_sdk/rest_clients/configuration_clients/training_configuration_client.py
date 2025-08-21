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
import logging

from geti_sdk.data_models import Project
from geti_sdk.data_models.configuration_models import Hyperparameters
from geti_sdk.data_models.configuration_models.training_configuration import (
    TrainingConfiguration,
)
from geti_sdk.http_session import GetiRequestException, GetiSession
from geti_sdk.rest_converters.training_configuration_rest_converter import (
    TrainingConfigurationRESTConverter,
)


class TrainingConfigurationClient:
    """
    REST client for managing training configurations for machine learning models within a project.

    This client provides methods to retrieve and update training configurations, including
    full configurations and hyperparameters-only configurations for specific models.
    """

    def __init__(self, workspace_id: str, project: Project, session: GetiSession):
        self.session = session
        project_id = project.id
        self.project = project
        self.workspace_id = workspace_id
        self.base_url = f"workspaces/{workspace_id}/projects/{project_id}/training_configuration"

    def get_configuration(self, model_manifest_id: str) -> TrainingConfiguration:
        """
        Retrieve the complete training configuration for a specific model manifest (AKA algorithm).

        Fetches the full training configuration including global parameters, hyperparameters,
        and all configuration sections (dataset preparation, training, evaluation) for the
        specified model manifest.

        :param model_manifest_id: Unique identifier of the model manifest to get configuration for
        :return: Complete TrainingConfiguration object containing all configuration parameters
        """
        url = f"{self.base_url}?model_manifest_id={model_manifest_id}"
        config_rest = self.session.get_rest_response(url=url, method="GET")
        config_rest["model_manifest_id"] = model_manifest_id
        return TrainingConfigurationRESTConverter.training_configuration_from_rest(config_rest)

    def get_hyperparameters_by_model_id(self, model_id: str, task_id: str | None) -> Hyperparameters:
        """
        Retrieve only the hyperparameters for a specific trained model.

        Fetches hyperparameters configuration for a specific model instance. This method
        returns only the hyperparameters portion of the configuration, excluding global
        parameters and other configuration metadata.

        :param model_id: Unique identifier of the trained model to get hyperparameters for
        :param task_id: Unique identifier of the task to get hyperparameters for
        :return: Hyperparameters object containing training, dataset preparation, and evaluation hyperparameters
        """
        url = f"{self.base_url}?model_id={model_id}"
        if task_id is not None:
            url += f"&task_id={task_id}"
        try:
            config_rest = self.session.get_rest_response(url=url, method="GET")
        except GetiRequestException as exc:
            if task_id is None:
                logging.error(
                    "Failed to get hyperparameters by model_id. Please note that for task-chain projects, "
                    "it's required to provide the task_id."
                )
            raise exc
        # when querying by model_id, the rest response contains only hyperparameters
        return TrainingConfigurationRESTConverter.hyperparameters_from_rest(config_rest)

    def set_configuration(self, configuration: TrainingConfiguration) -> None:
        """
        Update the training configuration for the project.

        Applies the provided training configuration to the project, updating all configuration
        parameters including global parameters and hyperparameters. The configuration will
        affect future training jobs in the project.

        :param configuration: TrainingConfiguration object containing the new configuration
                            parameters to apply to the project
        :return: None
        """
        config_rest = TrainingConfigurationRESTConverter.training_configuration_to_rest(configuration)
        self.session.get_rest_response(url=self.base_url, method="PATCH", data=config_rest)
