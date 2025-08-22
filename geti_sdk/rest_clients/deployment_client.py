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
import logging
import os
import tempfile
import zipfile
from collections.abc import Sequence

from geti_sdk.data_models import Project
from geti_sdk.data_models.code_deployment_info import (
    DeploymentModelIdentifier,
)
from geti_sdk.data_models.enums import OptimizationType
from geti_sdk.data_models.model import Model, OptimizedModel
from geti_sdk.deployment import Deployment
from geti_sdk.http_session import GetiSession
from geti_sdk.rest_clients import (
    ConfigurationClient,
    TrainingConfigurationClient,
)
from geti_sdk.rest_clients.model_client import ModelClient
from geti_sdk.rest_clients.prediction_client import PredictionClient
from geti_sdk.rest_converters import ConfigurationRESTConverter
from geti_sdk.utils import get_supported_algorithms


class DeploymentClient:
    """
    Class to manage model deployment for a certain Intel® Geti™ project.
    """

    def __init__(self, workspace_id: str, project: Project, session: GetiSession):
        self.session = session
        self.project = project
        self.workspace_id = workspace_id
        self.base_url = f"workspaces/{workspace_id}/projects/{project.id}"
        self.supported_algos = get_supported_algorithms(
            rest_session=session, project=project, workspace_id=workspace_id
        )

        self._model_client = ModelClient(workspace_id=workspace_id, project=project, session=session)
        self._prediction_client = PredictionClient(workspace_id=workspace_id, project=project, session=session)
        self._training_configuration_client = TrainingConfigurationClient(
            workspace_id=workspace_id, project=project, session=session
        )
        # legacy configuration client for backward compatibility
        self._configuration_client = ConfigurationClient(workspace_id=workspace_id, project=project, session=session)

    @property
    def deployment_package_url(self) -> str:
        """
        Return the base URL for the deployment group of endpoints

        :return: URL for the deployment endpoints for the Intel® Geti™ project
        """
        return self.base_url + "/deployment_package"

    @property
    def ready_to_deploy(self) -> bool:
        """
        Return True when the project is ready for deployment, False otherwise.

        A project is ready for deployment when it contains at least one trained model
        for each task.

        :return: True when the project is ready for deployment, False otherwise
        """
        return self._prediction_client.ready_to_predict and all(
            model is not None for model in self._model_client.get_all_active_models()
        )

    def _fetch_deployment(self, model_identifiers: list[DeploymentModelIdentifier]) -> Deployment:
        """
        Download and return the deployment.

        :param model_identifiers: list of models to include in the deployment
        :return: `Deployment` object containing the deployment
        """

        data = {
            "package_type": "geti_sdk",
            "models": [model.to_dict() for model in model_identifiers],
        }
        deployment_tempdir = tempfile.mkdtemp()
        zipfile_path = os.path.join(deployment_tempdir, "deployment.zip")
        response = self.session.get_rest_response(
            url=self.deployment_package_url + ":download", data=data, method="POST"
        )
        logging.info("Downloading project deployment archive...")
        with open(zipfile_path, "wb") as f:
            f.write(response.content)
        with zipfile.ZipFile(zipfile_path, "r") as zipped_deployment:
            zipped_deployment.extractall(deployment_tempdir)
        logging.info(f"Deployment for project '{self.project.name}' downloaded and extracted successfully.")
        if os.path.exists(zipfile_path):
            os.remove(zipfile_path)
        deployment = Deployment.from_folder(path_to_folder=os.path.join(deployment_tempdir, "deployment"))
        deployment._path_to_temp_resources = deployment_tempdir
        deployment._requires_resource_cleanup = True
        return deployment

    def deploy_project(
        self,
        output_folder: str | os.PathLike | None = None,
        models: Sequence[Model | OptimizedModel] | None = None,
        enable_explainable_ai: bool = False,
        prepare_for_ovms: bool = False,
    ) -> Deployment:
        """
        Deploy a project by creating a Deployment instance. The Deployment contains
        the optimized active models for each task in the project, and can be loaded
        with OpenVINO to run inference locally.

        The `models` parameter can be used in two ways:

          - If `models` is left as None this method will create a deployment containing
            the current active model for each task in the project.

          - If a list of models is passed, then it should always contain a number of
            Models less than or equal to the amount of tasks in the project. For
            instance, for a task-chain project with two tasks a list of at most two
            models could be passed (one for each task).
            If in this case a list of only one model is passed, that model will be
            used for its corresponding task and the method will resort to using the
            active model for the other task in the project.

        Optionally, configuration files for deploying models to OpenVINO Model Server
        (OVMS) can be generated. If you want this, make sure to pass
        `prepare_for_ovms=True` as input parameter. In this case, you MUST specify an
        `output_folder` to save the deployment and OVMS configuration files.
        Note that this does not change the resulting deployment in itself, it can
        still be used locally as well.

        :param output_folder: Path to a folder on local disk to which the Deployment
            should be downloaded. If no path is specified, the deployment will not be
            saved to disk directly. Note that it is always possible to save the
            deployment once it has been created, using the `deployment.save` method.
        :param models: Optional list of models to use in the deployment. If no list is
            passed, this method will create a deployment using the currently active
            model for each task in the project.
        :param enable_explainable_ai: True to include an Explainable AI head in
            the deployment. This will add an Explainable AI head to the model for each
            task in the project, allowing for the generation of saliency maps.
        :param prepare_for_ovms: True to prepare the deployment to be hosted on a
            OpenVINO model server (OVMS). Passing True will create OVMS configuration
            files for the model(s) in the project and instructions with sample code
            showing on how to launch an OVMS container including the deployed models.
        :return: Deployment for the project
        """
        if not self.ready_to_deploy:
            raise ValueError(
                f"Project '{self.project.name}' is not ready for deployment, please "
                f"make sure that each task in the project has at least one model "
                f"trained before deploying the project."
            )
        if prepare_for_ovms and output_folder is None:
            raise ValueError("An `output_folder` must be specified when setting `prepare_for_ovms=True`.")
        deployment = self._backend_deploy_project(
            output_folder=output_folder,
            models=models,
            require_xai=enable_explainable_ai,
        )
        if prepare_for_ovms:
            deployment.generate_ovms_config(output_folder=output_folder)
        return deployment

    def _backend_deploy_project(
        self,
        output_folder: str | os.PathLike | None = None,
        models: Sequence[Model | OptimizedModel] | None = None,
        require_xai: bool = False,
    ) -> Deployment:
        """
        Create a deployment for a project, using the /deployment_package:download
        endpoint from the backend.

        :param output_folder: Path to a folder on local disk to which the Deployment
            should be downloaded. If no path is specified, the deployment will not be
            saved to disk directly. Note that it is always possible to save the
            deployment once it has been created, using the `deployment.save` method.
        :param models: Optional list of models to use in the deployment. If no list is
            passed, this method will create a deployment using the currently active
            model for each task in the project.
        :param require_xai: True to include an Explainable AI head in the deployment.
        :return: Deployment for the project
        """
        # Get the models to deploy
        tasks = self.project.get_trainable_tasks()
        if models is None:
            models = self._model_client.get_all_active_models()

        tasks_with_model = [self._model_client.get_task_for_model(model) for model in models]

        if len(models) < len(tasks):
            tasks_without_model = [task for task in tasks if task not in tasks_with_model]
            active_models = [self._model_client.get_active_model_for_task(task) for task in tasks_without_model]

            models = models + active_models

            model_id_to_task_id = {
                model.id: task.id for model, task in zip(models, tasks_with_model + tasks_without_model)
            }
        else:
            model_id_to_task_id = {model.id: task.id for model, task in zip(models, tasks_with_model)}

        # Fetch the optimized model for each model, if it is not optimized already
        optimized_models: list[OptimizedModel] = []
        for model in models:
            if not isinstance(model, OptimizedModel):
                current_optim_models = model.optimized_models
                optimization_types = {op_model.optimization_type for op_model in current_optim_models}
                preferred_model = None
                for optimization_type in OptimizationType:
                    # The order of fields in OptimizationType gives us priority order for optimization types
                    if optimization_type not in optimization_types or (
                        optimization_type in (OptimizationType.ONNX, OptimizationType.NONE)
                    ):
                        continue
                    for op_model in current_optim_models:
                        if op_model.optimization_type == optimization_type and require_xai == op_model.has_xai_head:
                            preferred_model = op_model
                            break
                if preferred_model is None:
                    raise ValueError(
                        f"Could not find an optimized model for model '{model.name}' "
                        f"with optimization type '{optimization_type}' and explainable ai "
                        + ("enabled." if require_xai else "disabled.")
                    )
                optimized_models.append(preferred_model)

                # Update the task id for the model
                task_id_for_model = model_id_to_task_id[model.id]
                model_id_to_task_id.update({preferred_model.id: task_id_for_model})
            else:
                optimized_models.append(model)

        model_identifiers = [DeploymentModelIdentifier.from_model(model) for model in optimized_models]
        # Make sure models are sorted according to task order
        sorted_optimized_models: list[OptimizedModel] = []
        for task in tasks:
            for model in optimized_models:
                model_task_id = model_id_to_task_id[model.id]
                if model_task_id == task.id:
                    sorted_optimized_models.append(model)
                    break

        # Retrieve hyper parameters for the models
        hyper_parameters = []
        for model in sorted_optimized_models:
            if self.session.version.is_configuration_revamped:
                hyperparams_dict = self._training_configuration_client.get_hyperparameters_by_model_id(
                    model_id=model.id, task_id=model_id_to_task_id[model.id]
                ).model_dump()
            else:
                hyperparams = self._configuration_client.get_for_model(
                    model_id=model.id, task_id=model_id_to_task_id[model.id]
                )
                hyperparams_dict = ConfigurationRESTConverter.configuration_to_minimal_dict(hyperparams)
            hyper_parameters.append(hyperparams_dict)

        # Make the request to prepare and fetch the deployment package
        deployment = self._fetch_deployment(model_identifiers=model_identifiers)

        # Attach the appropriate hyper parameters and model_group_id to the deployed
        # models
        for index, model in enumerate(deployment.models):
            model.hyper_parameters = hyper_parameters[index]
            model.model_group_id = sorted_optimized_models[index].model_group_id

        # Save the deployment, if needed
        if output_folder is not None:
            deployment.save(path_to_folder=output_folder)
        return deployment
