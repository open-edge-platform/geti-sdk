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

import json
import logging
import os
import time

from geti_sdk.data_models import (
    Algorithm,
    FullConfiguration,
    GlobalConfiguration,
    Project,
    Task,
    TaskConfiguration,
)
from geti_sdk.data_models.configurable_parameter_group import PARAMETER_TYPES
from geti_sdk.http_session import GetiRequestException, GetiSession
from geti_sdk.rest_converters import ConfigurationRESTConverter
from geti_sdk.utils import deprecate, get_supported_algorithms


@deprecate(
    since="2.12",
    use="geti_sdk.rest_clients.configuration_clients",
    reason=(
        "This legacy client is deprecated in favor of the new ProjectConfigurationClient and "
        "TrainingConfigurationClient classes that provide improved functionality with new project and training "
        "configuration endpoints in Geti server version 2.12 and above"
    ),
)
class ConfigurationClient:
    """
    Class to manage configuration for a certain project.
    """

    def __init__(self, workspace_id: str, project: Project, session: GetiSession):
        self.session = session
        project_id = project.id
        self.project = project
        self.workspace_id = workspace_id
        self.task_ids = [task.id for task in project.get_trainable_tasks()]
        self.base_url = f"workspaces/{workspace_id}/projects/{project_id}/configuration"
        self.supported_algos = get_supported_algorithms(
            rest_session=session, project=project, workspace_id=workspace_id
        )

        # Query the project status to make sure that the project is loaded. Then we
        # can safely fetch the configuration later on, even for newly created projects
        self.session.get_rest_response(url=f"workspaces/{workspace_id}/projects/{project_id}/status", method="GET")
        # Hack: Wait for some time to make sure that the configurations are
        # initialized properly
        time.sleep(1)

    def get_task_configuration(self, task_id: str, algorithm_name: str | None = None) -> TaskConfiguration:
        """
        Get the configuration for the task with id `task_id`.

        :param task_id: ID of the task to get configurations for
        :param algorithm_name: Optional name of the algorithm to get configuration for.
            If an algorithm name is passed, the returned TaskConfiguration will contain
            only the hyper parameters for that algorithm, and won't hold any
            component parameters
        :return: TaskConfiguration holding all component parameters and hyper parameters
            for the task
        """
        url = f"{self.base_url}/task_chain/{task_id}"
        if algorithm_name is not None:
            url += f"?algorithm_name={algorithm_name}"
        config_data = self.session.get_rest_response(url=url, method="GET")
        return ConfigurationRESTConverter.task_configuration_from_dict(config_data)

    def get_global_configuration(self) -> GlobalConfiguration:
        """
        Get the project-wide configurable parameters.

        :return: GlobalConfiguration instance holding the configurable parameters for
            all project-wide components
        """
        config_data = self.session.get_rest_response(url=f"{self.base_url}/global", method="GET")
        return ConfigurationRESTConverter.global_configuration_from_rest(config_data)

    def _set_task_configuration(self, task_id: str, config: dict):
        """
        Update the configuration for a task.

        :param task_id: ID of the task to set the configuration for
        :param config: Dictionary containing the updated configuration values
        :return: Response of the configuration POST endpoint.
        """
        return self.session.get_rest_response(url=f"{self.base_url}/task_chain/{task_id}", method="POST", data=config)

    def set_project_auto_train(self, auto_train: bool = False) -> None:
        """
        Set the `auto_train` parameter for all tasks in the project.

        :param auto_train: True to enable auto_training, False to disable
        """
        for task_id in self.task_ids:
            config = self.get_task_configuration(task_id=task_id)
            config_data = config.set_parameter_value("auto_training", value=auto_train)
            self._set_task_configuration(task_id=task_id, config=config_data)

    def set_project_num_iterations(self, value: int = 50):
        """
        Set the number of iterations to train for each task in the project.

        :param value: Number of iterations to set
        """
        iteration_names = ["num_iters", "max_num_epochs"]
        for task_id in self.task_ids:
            config = self.get_task_configuration(task_id=task_id)
            parameter: PARAMETER_TYPES | None = None
            for parameter_name in iteration_names:
                parameter = config.get_parameter_by_name(parameter_name)
                if parameter is not None:
                    self._set_task_configuration(
                        task_id=task_id,
                        config=config.set_parameter_value(parameter_name, value=value),
                    )
                    break
            if parameter is None:
                raise ValueError(
                    f"No iteration parameters were found for task {config.task_title}. "
                    f"Unable to set number of iterations"
                )

    def set_project_parameter(
        self,
        parameter_name: str,
        value: bool | str | float | int,
        parameter_group_name: str | None = None,
    ):
        """
        Set the value for a parameter with `parameter_name` that lives in the
        group `parameter_group_name`. The parameter is set for all tasks in the project

        The `parameter_group_name` can be left as None, in that case this method will
        attempt to determine the appropriate parameter group automatically.

        :param parameter_name: Name of the parameter
        :param parameter_group_name: Optional name of the parameter group name to
            which the parameter belongs. If left as None (the default), this method will
            attempt to determine the correct parameter group automatically, if needed.
        :param value: Value to set for the parameter
        """
        for index, task_id in enumerate(self.task_ids):
            config = self.get_task_configuration(task_id)
            config_data = config.set_parameter_value(
                parameter_name=parameter_name,
                value=value,
                group_name=parameter_group_name,
            )
            self._set_task_configuration(task_id=task_id, config=config_data)

    def get_full_configuration(self) -> FullConfiguration:
        """
        Return the full configuration for a project (for both global and task_chain).

        :return: FullConfiguration object holding the global and task chain
            configuration
        """
        data = self.session.get_rest_response(url=self.base_url, method="GET")
        return ConfigurationRESTConverter.full_configuration_from_rest(data)

    def get_for_task_and_algorithm(self, task: Task, algorithm: Algorithm):
        """
        Get the hyper parameters for a specific task and algorithm.

        :param task: Task to get hyper parameters for
        :param algorithm: Algorithm to get hyper parameters for
        :return: TaskConfiguration holding only the model hyper parameters for the
            specified algorithm
        """
        if algorithm not in self.supported_algos.get_by_task_type(task.type):
            raise ValueError(
                f"The requested algorithm '{algorithm.name}' is not "
                f"supported for a task of type '{task.type}'. Unable to retrieve "
                f"configuration."
            )
        return self.get_task_configuration(task_id=task.id, algorithm_name=algorithm.model_manifest_id)

    def download_configuration(self, path_to_folder: str) -> FullConfiguration:
        """
        Retrieve the full configuration for a project from the cluster and save it to
        a file `configuration.json` in the folder specified at `path_to_folder`.

        :param path_to_folder: Folder to save the configuration to
        :return:
        """
        config = self.get_full_configuration()
        config_data = ConfigurationRESTConverter.configuration_to_minimal_dict(config)
        os.makedirs(path_to_folder, exist_ok=True, mode=0o770)
        configuration_path = os.path.join(path_to_folder, "configuration.json")
        with open(configuration_path, "w") as file:
            json.dump(config_data, file, indent=4)
        logging.info(f"Project parameters for project '{self.project.name}' were saved to file {configuration_path}.")
        return config

    def apply_from_object(self, configuration: FullConfiguration) -> FullConfiguration | None:
        """
        Attempt to apply the configuration values passed in as `configuration` to
        the project managed by this instance of the ConfigurationClient.

        :param configuration: FullConfiguration to be applied
        :return:
        """
        global_config = configuration.global_
        global_config.apply_identifiers(self.workspace_id, self.project.id)

        project_tasks = self.project.get_trainable_tasks()
        if len(project_tasks) != len(configuration.task_chain):
            raise ValueError(
                f"Structure of the configuration in: '{configuration}' does not match "
                f"that of the project. Unable to set configuration"
            )
        for task, task_config in zip(project_tasks, configuration.task_chain):
            current_task_config = self.get_task_configuration(task_id=task.id)
            model_storage_ids = [
                config.entity_identifier.model_storage_id for config in current_task_config.model_configurations
            ]

            task_config.apply_identifiers(
                workspace_id=self.workspace_id,
                project_id=self.project.id,
                task_id=task.id,
                model_storage_id=model_storage_ids[0],
            )
        data = ConfigurationRESTConverter.configuration_to_minimal_dict(configuration=configuration, deidentify=False)
        try:
            result = self.session.get_rest_response(url=self.base_url, method="POST", data=data)
        except GetiRequestException:
            failed_parameters: list[dict[str, str]] = []
            global_config = configuration.global_
            task_chain_config = configuration.task_chain
            for parameter in global_config:
                config_data = global_config.set_parameter_value(parameter.name, parameter.value)
                try:
                    self.session.get_rest_response(url=f"{self.base_url}/global", method="POST", data=config_data)
                except GetiRequestException:
                    failed_parameters.append({"global": parameter.name})
            for task_config in task_chain_config:
                for parameter in task_config:
                    config_data = task_config.set_parameter_value(parameter.name, parameter.value)
                    try:
                        self._set_task_configuration(task_id=task_config.task_id, config=config_data)
                    except GetiRequestException:
                        failed_parameters.append({task_config.task_title: parameter.name})
            logging.warning(
                f"Setting configuration failed for the following parameters: "
                f"{failed_parameters}. All other parameters were set successfully."
            )
            result = None
        if result:
            return configuration
        return None

    def apply_from_file(self, path_to_folder: str, filename: str | None = None) -> FullConfiguration | None:
        """
        Attempt to apply a configuration from a file on disk. The
        parameter `path_to_folder` is mandatory and should point to the folder in which
        the configuration file to upload lives. The parameter `filename` is optional,
        when left as `None` this method will look for a file `configuration.json` in
        the specified folder.

        :param path_to_folder: Path to the folder in which the configuration file to
            apply lives
        :param filename: Optional filename for the configuration file to apply
        :return:
        """
        if filename is None:
            filename = "configuration.json"
        path_to_config = os.path.join(path_to_folder, filename)
        if not os.path.isfile(path_to_config):
            raise ValueError(
                f"Unable to find configuration file at {path_to_config}. Please "
                f"provide a valid path to the folder holding the configuration data."
            )
        with open(path_to_config) as file:
            data = json.load(file)
        config = ConfigurationRESTConverter.full_configuration_from_rest(data)
        return self.apply_from_object(config)

    def set_configuration(
        self,
        configuration: FullConfiguration | GlobalConfiguration | TaskConfiguration,
    ):
        """
        Set the configuration for the project. This method accepts either a
        FullConfiguration, TaskConfiguration or GlobalConfiguration object

        :param configuration: Configuration to set
        :return:
        """
        if isinstance(configuration, FullConfiguration):
            full_configuration = configuration
        elif isinstance(configuration, TaskConfiguration):
            full_configuration = self.get_full_configuration()
            if configuration.task_id is None:
                raise ValueError(
                    "Cannot set a TaskConfiguration without a task_id. Please make sure to set a valid task_id."
                )
            # Find index of configuration in the task chain
            task_index: int | None = None
            for ti, task_configuration in enumerate(full_configuration.task_chain):
                if task_configuration.task_id == configuration.task_id:
                    task_index = ti
                    break
            if task_index is None:
                raise ValueError(
                    f"Unable to find task with id {configuration.task_id} in the "
                    f"project. Please make sure that the task_id for the configuration "
                    f"you have provided is valid"
                )
            full_configuration.task_chain[task_index] = configuration
        elif isinstance(configuration, GlobalConfiguration):
            full_configuration = self.get_full_configuration()
            full_configuration.global_ = configuration
        else:
            raise TypeError(
                f"Invalid configuration of type '{type(configuration)}' received. Unable to set configuration."
            )
        self.apply_from_object(full_configuration)

    def get_for_model(self, task_id: str, model_id: str) -> TaskConfiguration:
        """
        Get the hyper parameters for the model with id `model_id`. Note that the model
        has to be trained within the task with id `task_id` in order for the
        parameters to be retrieved successfully.

        :param task_id: ID of the task to get configurations for
        :param model_id: ID of the model to get the hyper parameters for
        :return: TaskConfiguration holding all hyper parameters for the model
        """
        url = f"{self.base_url}/task_chain/{task_id}?model_id={model_id}"
        config_data = self.session.get_rest_response(url=url, method="GET")
        return ConfigurationRESTConverter.task_configuration_from_dict(config_data)
