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
from typing import TypeVar

from requests import Response

from geti_sdk.data_models import (
    Algorithm,
    Job,
    Model,
    ModelGroup,
    ModelSummary,
    OptimizedModel,
    Project,
    Task,
)
from geti_sdk.data_models.enums import JobState, JobType, OptimizationType
from geti_sdk.http_session import GetiSession
from geti_sdk.rest_converters import ModelRESTConverter
from geti_sdk.utils import get_supported_algorithms
from geti_sdk.utils.job_helpers import get_job_with_timeout, monitor_job

ModelType = TypeVar("ModelType", Model, OptimizedModel)


class ModelClient:
    """
    Class to manage the models and model groups for a certain project
    """

    def __init__(self, workspace_id: str, project: Project, session: GetiSession):
        self.session = session
        project_id = project.id
        self.project = project
        self.workspace_id = workspace_id
        self.task_ids = [task.id for task in project.get_trainable_tasks()]
        self.base_url = f"workspaces/{workspace_id}/projects/{project_id}/model_groups"
        self.supported_algos = get_supported_algorithms(
            rest_session=self.session, project=project, workspace_id=workspace_id
        )

    def get_all_model_groups(self) -> list[ModelGroup]:
        """
        Return a list of all model groups in the project.

        :return: List of model groups in the project
        """
        response = self.session.get_rest_response(url=self.base_url, method="GET")
        response_array = response["model_groups"]
        model_groups = [ModelRESTConverter.model_group_from_dict(group) for group in response_array]
        # Update algorithm details
        for group in model_groups:
            if group.learning_approach == "fully_supervised":  # only fully supervised models have a manifest
                group.algorithm = self.supported_algos.get_by_model_manifest_id(
                    model_manifest_id=group.model_template_id
                )
            for model in group.models:
                # set the model storage id, to link models to their parent group
                model.model_storage_id = group.id
        return model_groups

    def get_latest_model_for_all_model_groups(self) -> list[Model]:
        """
        Return the latest trained models for each model group in the project.

        :return: List of models, one for each trained algorithm in the project.
        """
        model_groups = self.get_all_model_groups()
        latest_models: list[Model] = []
        for model_group in model_groups:
            lm = model_group.get_latest_model()
            latest_models.append(self._get_model_detail(group_id=model_group.id, model_id=lm.id))
        return latest_models

    def get_model_group_by_algo_name(self, algorithm_name: str) -> ModelGroup | None:
        """
        Return the model group for the algorithm named `algorithm_name`, if any. If
        no model group for this algorithm is found in the project, this method returns
        None

        :param algorithm_name: Name of the algorithm
        :return: ModelGroup instance corresponding to this algorithm
        """
        model_groups = self.get_all_model_groups()
        return next(
            (group for group in model_groups if group.algorithm.name == algorithm_name),
            None,
        )

    def get_latest_model_by_algo_name(self, algorithm_name: str) -> Model | None:
        """
        Return the latest model for a specific algorithm. If no model has been trained
        for the algorithm, this method returns None.

        :param algorithm_name: Name fo the algorithm for which to return the model
        :return: Model object respresenting the model.
        """
        model_group = self.get_model_group_by_algo_name(algorithm_name)
        if model_group:
            model_summary = model_group.get_latest_model()
            return self._get_model_detail(group_id=model_group.id, model_id=model_summary.id)
        return None

    def get_latest_optimized_model(
        self,
        algorithm_name: str,
        optimization_type: str = "MO",
        precision: str = "FP16",
        require_xai: bool = False,
    ) -> OptimizedModel:
        """
        Return the optimized model for the latest trained model for a specified
        algorithm. Additional parameters allow filtering on the optimization type
        (e.g. 'nncf', 'pot', 'mo', 'onnx'), precision ('int8', 'fp16', 'fp32') and
        whether or not the model includes an XAI head for saliency map generation.

        If no optimized model for the specified criteria can be found, this method
        raises an error

        :param algorithm_name: Name of the algorithm to retrieve the model for
        :param optimization_type: Optimization type to select. Options are 'mo',
            'nncf', 'pot', 'onnx'. Case insensitive. Defaults to 'MO'
        :param precision: Model precision to select. Options are 'INT8', 'FP16', 'FP32'.
            Defaults to 'FP16'
        :param require_xai: If True, only select models that include an XAI head.
            Defaults to False
        """
        base_model = self.get_latest_model_by_algo_name(algorithm_name=algorithm_name)
        if base_model is None:
            raise RuntimeError(f"No trained model was found for algorithm `{algorithm_name}`")
        n_optimized_models = len(base_model.optimized_models)
        logging.info(
            f"{n_optimized_models} optimized models were found for algorithm "
            f"`{algorithm_name}`. Finding the most recent optimized model with "
            f"precision {precision} and optimization type {optimization_type}."
        )
        if require_xai:
            opt_models = [m for m in base_model.optimized_models if m.has_xai_head]
        else:
            opt_models = base_model.optimized_models
        if n_optimized_models != 0 and len(opt_models) == 0:
            raise RuntimeError(
                f"Algorithm {algorithm_name} has a trained base model and "
                f"{n_optimized_models} optimized models, but no optimized model with "
                f"XAI head was found."
            )
        cap_prec = precision.upper()
        supported_precisions = ["FP32", "FP16", "INT8"]
        if cap_prec not in supported_precisions:
            raise ValueError(
                f"Invalid target precision specified: {precision}. Supported options are: {supported_precisions}"
            )
        opt_models_precision = [om for om in opt_models if cap_prec in om.name]
        cap_opt_type = optimization_type.upper()
        opt_type = OptimizationType(cap_opt_type)
        opt_models_prec_type = [om for om in opt_models_precision if om.optimization_type == opt_type]
        if len(opt_models_prec_type) == 0:
            raise RuntimeError(
                f"Algorithm {algorithm_name} has a trained base model and "
                f"{n_optimized_models} optimized models, but no optimized model "
                f"matches the required optimization type and precision."
            )
        if len(opt_models_prec_type) == 1:
            return opt_models_prec_type[0]
        logging.info(
            f"Found {len(opt_models_prec_type)} models that match the selection "
            f"criteria. Returning the most recently created matching model."
        )
        creation_dates = [om.creation_date for om in opt_models_prec_type]
        max_index = creation_dates.index(max(creation_dates))
        return opt_models_prec_type[max_index]

    def get_model_by_algorithm_task_and_version(
        self,
        algorithm: Algorithm,
        version: int | None = None,
        task: Task | None = None,
    ) -> Model | None:
        """
        Retrieve a Model from the Intel® Geti™ server, corresponding to a specific
        algorithm and model version. If no version is passed, this method will
        retrieve the latest model for the algorithm.

        If no model for the algorithm is available in the project, this method returns
        None

        :param algorithm: Algorithm for which to get the model
        :param version: Version of the model to retrieve. If left as None, returns the
            latest version
        :param task: Task for which to get the model. If left as None, this method
            searches for models for `algorithm` in all tasks in the project
        :return: Model object corresponding to `algorithm` and `version`, for a
            specific `task`, if any. If no model is found by those parameters, this
            method returns None
        """
        if task is not None and algorithm.task != task.type:
            raise ValueError(
                f"Unable to retrieve model. The algorithm {algorithm} is not available for the task {task}"
            )
        model_groups = self.get_all_model_groups()
        model_group: ModelGroup | None = None
        for group in model_groups:
            if group.algorithm == algorithm:
                if task is None:
                    model_group = group
                    break
                if group.task_id == task.id:
                    model_group = group
                    break
        if model_group is None:
            return None
        if version is not None:
            try:
                model_summary = model_group.get_model_by_version(version=version)
            except ValueError:
                return None
        else:
            model_summary = model_group.get_latest_model()
        return self._get_model_detail(model_group.id, model_id=model_summary.id)

    def _get_model_detail(self, group_id: str, model_id: str) -> Model:
        """
        Return the Model object holding detailed information about the model with
        `model_id` living in the model group with `group_id`.

        :param group_id: Unique database ID of the model group to which the model to
            get belongs
        :param model_id: Unique database ID of the model to get
        :return: Model instance holding detailed information about the model
        """
        model_detail = self.session.get_rest_response(url=f"{self.base_url}/{group_id}/models/{model_id}", method="GET")
        model = ModelRESTConverter.model_from_dict(model_detail)
        model.model_group_id = group_id
        model.base_url = self.base_url
        return model

    def update_model_detail(self, model: Model | ModelSummary) -> Model:
        """
        Update the model such that its details are up to date. This includes updating
        the list of available optimized models for the model.

        :param model: Model or ModelSummary object, representing the model to update
        :return: Model object containing the up to date details of the model
        """
        if isinstance(model, ModelSummary):
            group_id = model.model_storage_id
        elif isinstance(model, Model):
            group_id = model.model_group_id
        else:
            raise TypeError(
                f"Invalid type {type(model)}. Argument `model` must be either a Model or ModelSummary object"
            )
        model_detail = self.session.get_rest_response(url=f"{self.base_url}/{group_id}/models/{model.id}", method="GET")
        updated_model = ModelRESTConverter.model_from_dict(model_detail)
        updated_model.model_group_id = group_id
        updated_model.base_url = self.base_url
        return updated_model

    def set_active_model(
        self,
        model: Model | ModelSummary | None = None,
        algorithm: Algorithm | str | None = None,
    ) -> None:
        """
        Set the model as the active model.

        :param model: Model or ModelSummary object representing the model to set as active
        :param algorithm: Algorithm or algorithm name for which to set the model as active
        :raises ValueError: If neither `model` nor `algorithm` is specified,
            If the algorithm is not supported in the project,
            If unable to set the active model
        """
        # First we determine the algorithm name
        if model is not None:
            # Update the model details to make sure we have the latest information
            model = self.update_model_detail(model)
            algorithm_name = model.architecture
        elif algorithm is not None:
            if isinstance(algorithm, str):
                algorithm_name = algorithm
            elif isinstance(algorithm, Algorithm):
                algorithm_name = algorithm.name
            else:
                raise ValueError(
                    f"Invalid type {type(algorithm)}. Argument `algorithm` must be "
                    "either a string representing the algorith name or an Algorithm object"
                )
        else:
            raise ValueError("Either `model` or `algorithm` must be specified to set the active model")
        # Now we make sure that the algorithm is supported in the project
        algorithms_supported_in_the_project = {
            algorithm.name
            for task in self.project.get_trainable_tasks()
            for algorithm in self.supported_algos.get_by_task_type(task.type)
        }
        if algorithm_name not in algorithms_supported_in_the_project:
            raise ValueError(f"Algorithm `{algorithm_name}` is not supported in the project {self.project.name}.")
        # We get a model group for the algorithm
        model_group = self.get_model_group_by_algo_name(algorithm_name=algorithm_name)
        model_group_id = model_group.id if model_group is not None else None
        if model_group_id is None:
            raise ValueError("Unable to set the active model. Train a model first")
        # Fire a request to the server to set a model from the group as active
        url = f"{self.base_url}/{model_group_id}:activate"
        _ = self.session.get_rest_response(url=url, method="POST")
        logging.info(f"{algorithm_name} model set as active successfully")

    def get_active_model_for_task(self, task: Task) -> Model | None:
        """
        Return the Model details for the currently active model, for a task if any.
        If the task does not have any trained models, this method returns None

        :param task: Task object containing details of the task to get the model for
        :return: Model object representing the currently active model in the
            Intel® Geti™ project, if any
        """
        model_groups = self.get_all_model_groups()
        model_id: str | None = None
        group_id: str | None = None
        for group in model_groups:
            if not group.has_trained_models:
                continue
            if group.algorithm.task != task.type:
                continue
            model_summary = group.get_latest_model()
            if model_summary is not None and model_summary.active_model:
                model_id = model_summary.id
                group_id = group.id
                break
        if model_id is not None:
            return self._get_model_detail(group_id=group_id, model_id=model_id)
        # Sometimes the `active_model` flag is not set by the server, even though
        # there is a model available for the task. In that case we fall back to
        # returning the latest model in the available model group
        if len(model_groups) == 1:
            model_summary_no_active_check = model_groups[0].get_latest_model()
            if model_summary_no_active_check is not None:
                model_id = model_summary_no_active_check.id
                group_id = model_groups[0].id
                if model_id is not None:
                    return self._get_model_detail(group_id=group_id, model_id=model_id)
        return None

    def _download_model(self, model: ModelType, path_to_folder: str) -> ModelType:
        """
        Download a Model or OptimizedModel.

        :param model: Model or OptimizedModel to download
        :return: Model or OptimizedModel object holding the details of the downloaded
            model
        """
        if isinstance(model, Model):
            url = f"{self.base_url}/{model.model_group_id}/models/{model.id}/export"
            filename = f"{model.name}_base.zip"
        elif isinstance(model, OptimizedModel):
            url = (
                f"{self.base_url}/{model.model_group_id}/models/"
                f"{model.previous_trained_revision_id}/optimized_models/"
                f"{model.id}/export"
            )
            filename = f"{model.name}_{model.optimization_type}_optimized.zip"
        else:
            raise ValueError(f"Invalid model type: `{type(model)}. Unable to download model data.")
        response = self.session.get_rest_response(url=url, method="GET", contenttype="zip")
        model_folder = os.path.join(path_to_folder, "models")
        os.makedirs(model_folder, exist_ok=True, mode=0o770)
        model_filepath = os.path.join(model_folder, filename)
        with open(model_filepath, "wb") as f:
            f.write(response.content)
        return model

    def download_active_model_for_task(self, path_to_folder: str, task: Task) -> Model | None:
        """
        Download the currently active model for the task.
        If the task does not have an active model yet, this method returns None

        This method will create a directory 'models' in the path specified in
        `path_to_folder`

        :param path_to_folder: Path to the target folder in which to save the active
            model, and all optimized models derived from it.
        :param task: Task object containing details of the task to download the model
            for
        :return: Model instance holding the details of the active model
        """
        model = self.get_active_model_for_task(task=task)
        if model is None:
            logging.info(
                f"Project '{self.project.name} does not have any trained models yet, unable to download active model."
            )
            return None
        model_filepath = os.path.join(path_to_folder, "models")
        logging.info(
            f"Downloading active model for task {task.title} in project "
            f"{self.project.name} to folder {model_filepath}..."
        )
        self._download_model(model, path_to_folder=path_to_folder)
        for optimized_model in model.optimized_models:
            self._download_model(optimized_model, path_to_folder=path_to_folder)
        model_info_filepath = os.path.join(model_filepath, f"{task.type}_model_details.json")
        with open(model_info_filepath, "w") as f:
            json.dump(model.to_dict(), f, indent=4)
        return model

    def get_all_active_models(self) -> list[Model | None]:
        """
        Return the Model details for the active model for all tasks in the project,
        if the tasks have any.

        This method returns a list of Models, where the index of the Model in the list
        corresponds to the index of the task in list of trainable tasks for the project.

        If any of the tasks do not have a trained model, the entry corresponding to
        the index of that task will be None

        :return: Model object representing the currently active model for the task in
            the Intel® Geti™ project, if any
        """
        return [self.get_active_model_for_task(task=task) for task in self.project.get_trainable_tasks()]

    def download_all_active_models(self, path_to_folder: str) -> list[Model | None]:
        """
        Download the active models for all tasks in the project.

        This method will create a directory 'models' in the path specified in
        `path_to_folder`

        :param path_to_folder: Path to the target folder in which to save the active
            models, and all optimized models derived from them.
        :return: List of Model objects representing the currently active models
            (if any) for all tasks in the Intel® Geti™ project. The index of the
            Model in the list corresponds to the index of the task in the list of
            trainable tasks for the project.
        """
        return [
            self.download_active_model_for_task(path_to_folder=path_to_folder, task=task)
            for task in self.project.get_trainable_tasks()
        ]

    def get_model_for_job(self, job: Job, check_status: bool = True) -> Model:
        """
        Return the model that was created by the `job` from the Intel® Geti™ server.

        :param job: Job to retrieve the model for
        :param check_status: True to first update the status of the job, to make sure
            it is finished. Setting this to False will not update the job status.
        :return: Model produced by the job
        """
        if check_status:
            job.update(self.session)
        job_pid = job.metadata.project.id
        if job_pid != self.project.id:
            raise ValueError(
                f"Cannot get model for job `{job.description}`. This job does not "
                f"belong to the project managed by this ModelClient instance."
            )
        if job.state != JobState.FINISHED:
            raise ValueError(
                f"Job `{job.description}` is not finished yet, unable to retrieve "
                f"model for the job. Please wait until job is finished"
            )
        metadata = job.metadata
        task_data = metadata.task
        version = task_data.model_version
        algorithm = self.supported_algos.get_by_model_manifest_id(task_data.model_template_id)
        if hasattr(task_data, "name") and job.type in (
            JobType.TRAIN,
            JobType.INFERENCE,
            JobType.EVALUATE,
        ):
            task_name = task_data.name
            task = next(task for task in self.project.get_trainable_tasks() if task.title == task_name)
            return self.get_model_by_algorithm_task_and_version(algorithm=algorithm, version=version, task=task)
        if job.type == JobType.OPTIMIZATION:
            model_group_id, optimized_model_id = None, None
            if hasattr(metadata, "model_group_id"):
                model_group_id = metadata.model_group_id
            if hasattr(metadata, "optimized_model_id"):
                optimized_model_id = metadata.optimized_model_id
            if model_group_id is not None and optimized_model_id is not None:
                return self._get_model_detail(model_group_id, optimized_model_id)
        raise ValueError(
            f"Unable to retrieve model for job {job.name} of type {job.type}. Getting "
            f"the model for this job type is not supported. "
        )

    def get_task_for_model(self, model: Model | OptimizedModel) -> Task:
        """
        Return the task to which a certain model belongs, if possible. This method only
        works when the model identifiers are still in place, if they have been stripped
        it will raise a ValueError.

        If the model does not match any task in the project, this method will raise an
        error.

        :param model: Model or OptimizedModel to find the task for
        :return: Task for which the model was trained
        """
        project_model_groups = self.get_all_model_groups()
        tasks = self.project.get_trainable_tasks()
        model_group: ModelGroup | None = None
        error_msg = f"Unable to match model '{model}' to any task in project {self.project}. "
        if model.model_group_id is None:
            raise ValueError(error_msg + "The model does not contain a model group identifier")
        for group in project_model_groups:
            if group.id == model.model_group_id:
                model_group = group
                break
        if model_group is None:
            raise ValueError(error_msg + "The model does not belong to any of the model groups in the project.")
        task_id = model_group.task_id
        model_task: Task | None = None
        for task in tasks:
            if task.id == task_id:
                model_task = task
                break
        if model_task is None:
            raise ValueError(
                error_msg + f"Model was found on the server but could not be linked "
                f"to specific task for 'task_id={model_group.task_id}'."
            )
        return model_task

    def optimize_model(self, model: Model, optimization_type: str = "pot") -> Job:
        """
        Start an optimization job for the specified `model`.

        :param model: Model to optimize
        :param optimization_type: DEPRECATED. Type of optimization to run. Currently
            supported values: ["pot"]. Case insensitive. Defaults to "pot"
        :return: Job object referring to the optimization job running on the
            Intel® Geti™ server.
        """
        if isinstance(model, OptimizedModel):
            raise ValueError(
                f"Model {model.name} is already optimized, please specify a base model for optimization instead."
            )
        valid_optimization_types = ["pot"]
        optimization_type = optimization_type.lower()
        if optimization_type not in valid_optimization_types:
            raise ValueError(
                f"Invalid optimization type specified: `{optimization_type}`. Valid "
                f"options are: {valid_optimization_types}"
            )
        optimize_model_url = model.base_url + ":optimize"
        payload = {}
        response = self.session.get_rest_response(url=optimize_model_url, method="POST", data=payload)
        return get_job_with_timeout(
            job_id=response["job_id"],
            session=self.session,
            workspace_id=self.workspace_id,
            job_type="optimization",
        )

    def purge_model(self, model: Model | ModelSummary) -> None:
        """
        Purge the model from the Intel® Geti™ server.

        This will permanently delete all the files related to the model including base model weights,
        optimized model weights and exportable code for the Intel® Geti™ server.

        :param model: Model to archive. Only base models are accepted, not optimized models.
            Note: the model must not be the latest in the model group or be the active model.
        :raises ValueError: If the model does not have a base_url, meaning it cannot be purged
            from the remote server.
        """
        model = self.update_model_detail(model)
        if model.base_url is None:
            raise ValueError(f"Model {model.name} does not have a base_url. Unable to purge the model.")
        purge_model_url = model.base_url + ":purge"
        response = self.session.get_rest_response(
            url=purge_model_url,
            method="POST",
        )
        if type(response) is Response and response.status_code == 204:
            logging.info(f"Model {model.name} was successfully purged.")
        else:
            logging.error(f"Failed to purge model {model.name}.")

    def monitor_job(self, job: Job, timeout: int = 10000, interval: int = 15) -> Job:
        """
        Monitor and print the progress of a `job`. Program execution is
        halted until the job has either finished, failed or was cancelled.

        Progress will be reported in 15s intervals

        :param job: job to monitor
        :param timeout: Timeout (in seconds) after which to stop the monitoring
        :param interval: Time interval (in seconds) at which the ModelClient polls
            the server to update the status of the jobs. Defaults to 15 seconds
        :return: job with it's status updated
        """
        return monitor_job(session=self.session, job=job, timeout=timeout, interval=interval)
