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
import datetime
import json
import logging
import os
import shutil
from collections.abc import Callable, Sequence
from typing import Any

import attr
import numpy as np
from model_api.models import Model as OMZModel

from geti_sdk.data_models import Label, Prediction, Project, Task
from geti_sdk.deployment.data_models import ROI, IntermediateInferenceResult
from geti_sdk.rest_converters import ProjectRESTConverter

from .deployed_model import DeployedModel
from .inference_hook_interfaces import PostInferenceHookInterface
from .utils import (
    OVMS_README_PATH,
    assign_empty_label,
    flow_control,
    generate_ovms_model_name,
    pipeline_callback_factory,
)


@attr.define(slots=False)
class Deployment:
    """
    Representation of a deployed Intel® Geti™ project that can be used to run
    inference locally
    """

    project: Project
    models: list[DeployedModel]

    def __attrs_post_init__(self):
        """
        Initialize private attributes.
        """
        self._is_single_task: bool = len(self.project.get_trainable_tasks()) == 1
        self._are_models_loaded: bool = False
        self._inference_converters: dict[str, Any] = {}
        self._path_to_temp_resources: str | None = None
        self._requires_resource_cleanup: bool = False
        self._post_inference_hooks: list[PostInferenceHookInterface] = []
        self._empty_label: Label | None = None
        self._asynchronous_mode: bool = False
        self._async_callback_defined: bool = False

    @property
    def is_single_task(self) -> bool:
        """
        Return True if the deployment represents a project with only a single task.

        :return: True if the deployed project contains only one trainable task, False
            if it is a pipeline project
        """
        return self._is_single_task

    @property
    def are_models_loaded(self) -> bool:
        """
        Return True if all inference models for the Deployment are loaded and ready
        to infer.

        :return: True if all inference models for the deployed project are loaded in
            memory and ready for inference
        """
        return self._are_models_loaded

    @property
    def asynchronous_mode(self) -> bool:
        """
        Return True if the deployment is configured for asynchronous inference execution.

        Asynchronous execution can result in a large increase in throughput for
        certain applications, for example video processing. However, it requires
        slightly more configuration compared to synchronous (the default) mode. For a
        more detailed overview of the differences between synchronous and
        asynchronous execution, please refer to the OpenVINO documentation at
        https://docs.openvino.ai/2024/notebooks/115-async-api-with-output.html

        :return: True if the deployment is set in asynchronous execution mode, False
            if it is in synchronous mode.
        """
        return self._async_callback_defined

    @asynchronous_mode.setter
    def asynchronous_mode(self, mode: bool):
        """
        Set the inference mode for the deployment

        :param mode: False to set the deployment to synchronous mode. Removes all
            asynchronous callbacks for the models in the deployment.
        """
        if mode:
            if not self._async_callback_defined:
                raise ValueError(
                    "Please use the `set_asynchronous_callback` method to switch a "
                    "deployment to asynchronous inference mode."
                )
            logging.debug("Deployment is already in asynchronous mode.")
        else:
            self._async_callback_defined = False
            for model in self.models:
                model.asynchronous_mode = False
            logging.info("Asynchronous inference mode disabled. All callbacks removed.")

    def save(self, path_to_folder: str | os.PathLike) -> bool:
        """
        Save the Deployment instance to a folder on local disk.

        :param path_to_folder: Folder to save the deployment to
        :return: True if the deployment was saved successfully, False otherwise
        """
        project_dict = ProjectRESTConverter.to_dict(self.project, deidentify=False)
        deployment_folder = os.path.join(path_to_folder, "deployment")

        os.makedirs(deployment_folder, exist_ok=True, mode=0o770)

        # Save model for each task
        for task, model in zip(self.project.get_trainable_tasks(), self.models):
            model_dir = os.path.join(deployment_folder, task.title)
            os.makedirs(model_dir, exist_ok=True, mode=0o770)
            success = model.save(model_dir)
            if not success:
                logging.exception(f"Saving model '{model.name}' failed. Unable to save deployment.")
                return False

        # Save project data
        project_filepath = os.path.join(deployment_folder, "project.json")
        with open(project_filepath, "w") as project_file:
            json.dump(project_dict, project_file, indent=4)

        # Save post inference hooks, if any
        if self.post_inference_hooks:
            hook_config_file = os.path.join(deployment_folder, "hook_config.json")
            hook_configs: list[dict[str, Any]] = []
            for hook in self.post_inference_hooks:
                hook_configs.append(hook.to_dict())
            with open(hook_config_file, "w") as file:
                json.dump({"post_inference_hooks": hook_configs}, file)

        # Clean up temp resources if needed
        if self._requires_resource_cleanup:
            self._remove_temporary_resources()
            self._requires_resource_cleanup = False

        return True

    @classmethod
    def from_folder(cls, path_to_folder: str | os.PathLike) -> "Deployment":
        """
        Create a Deployment instance from a specified `path_to_folder`.

        :param path_to_folder: Path to the folder containing the Deployment data
        :return: Deployment instance corresponding to the deployment data in the folder
        """
        deployment_folder = path_to_folder
        if not isinstance(path_to_folder, str):
            path_to_folder = str(path_to_folder)
        if not path_to_folder.endswith("deployment"):
            if "deployment" in os.listdir(path_to_folder):
                deployment_folder = os.path.join(path_to_folder, "deployment")
            else:
                raise ValueError(
                    f"No `deployment` folder found in the directory at `{path_to_folder}`. Unable to load Deployment."
                )
        project_filepath = os.path.join(deployment_folder, "project.json")
        with open(project_filepath) as project_file:
            project_dict = json.load(project_file)
        project = ProjectRESTConverter.from_dict(project_dict)
        task_folder_names = [task.title for task in project.get_trainable_tasks()]
        models: list[DeployedModel] = []
        for task_folder in task_folder_names:
            models.append(DeployedModel.from_folder(os.path.join(deployment_folder, task_folder)))
        deployment = cls(models=models, project=project)

        # Load post inference hooks, if any
        hook_config_file = os.path.join(deployment_folder, "hook_config.json")
        if os.path.isfile(hook_config_file):
            available_hooks = {subcls.__name__: subcls for subcls in PostInferenceHookInterface.__subclasses__()}
            with open(hook_config_file) as file:
                hook_dict = json.load(file)
            for hook_data in hook_dict["post_inference_hooks"]:
                for hook_name, hook_args in hook_data.items():
                    target_hook = available_hooks[hook_name]
                    hook = target_hook.from_dict(hook_args)
                deployment.add_post_inference_hook(hook)
        return deployment

    def load_inference_models(
        self,
        device: str | Sequence[str] = "CPU",
        max_async_infer_requests: int | Sequence[int] | None = None,
        openvino_configuration: dict[str, str] | None = None,
    ):
        """
        Load the inference models for the deployment to the specified device.

        Note: For a list of devices that are supported for OpenVINO inference, please see:
        https://docs.openvino.ai/latest/openvino_docs_OV_UG_supported_plugins_Supported_Devices.html

        :param device: Device to load the inference models to (e.g. 'CPU', 'GPU',
            'AUTO', etc).

            **NOTE**: For task chain deployments, it is possible to pass a list of device names
            instead. Each entry in the list is the target device for the model
            corresponding to it's index. I.e. the first entry is applied for the
            first model, the second entry for the second model.
        :param max_async_infer_requests: Maximum number of infer requests to use in
            asynchronous mode. This parameter only takes effect when the asynchronous
            inference mode is used. It controls the maximum number of request that
            will be handled in parallel. When set to 0, OpenVINO will attempt to
            determine the optimal number of requests for your system automatically.
            When left as None (the default), a single infer request per model will be
            used to conserve memory

            **NOTE**: For task chain deployments, it is possible to pass a list of integers.
            Each entry in the list is the maximum number of infer requests for the model
            corresponding to it's index. I.e. the first number is applied for the
            first model, the second number for the second model.
        :param openvino_configuration: Configuration for the OpenVINO execution mode
            and plugins. This can include for example specific performance hints. For
            further details, refer to the OpenVINO documentation here:
            https://docs.openvino.ai/2022.3/openvino_docs_OV_UG_Performance_Hints.html#doxid-openvino-docs-o-v-u-g-performance-hints
        """
        if max_async_infer_requests is None:
            max_async_infer_requests = 1
        elif max_async_infer_requests == 0 and device == "CPU":
            max_async_infer_requests = os.cpu_count()

        for idx, model in enumerate(self.models):
            dev = device[idx] if isinstance(device, Sequence) and not isinstance(device, str) else device
            if isinstance(max_async_infer_requests, Sequence):
                max_requests = max_async_infer_requests[idx]
            else:
                max_requests = max_async_infer_requests
            model.load_inference_model(
                device=dev,
                project=self.project,
                max_async_infer_requests=max_requests,
                plugin_configuration=openvino_configuration,
                task_index=idx,
            )

        # Extract empty label for the upstream task
        self._empty_label = self.models[0].labels.get_empty_label()

        self._are_models_loaded = True
        logging.info(f"Inference models loaded on device `{device}` successfully.")

    def infer(self, image: np.ndarray, name: str | None = None) -> Prediction:
        """
        Run inference on an image for the full model chain in the deployment.

        :param image: Image to run inference on, as a numpy array containing the pixel
            data. The image is expected to have dimensions [height x width x channels],
            with the channels in RGB order
        :param name: Optional name for the image, if specified this will be used in
            any post inference hooks belonging to the deployment.
        :return: inference results
        """
        self._check_models_loaded()

        if self.asynchronous_mode:
            raise RuntimeError(
                "Unable to use synchronous inference while the deployment is set to "
                "asynchronous execution mode. Please use the `.infer_async` method "
                "instead."
            )

        # Single task inference
        if self.is_single_task:
            prediction = self._infer_task(image, task=self.project.get_trainable_tasks()[0], explain=False)
        # Multi-task inference
        else:
            prediction = self._infer_pipeline(image=image, explain=False)

        # Empty label is not generated by prediction postprocessing correctly, append
        # it here if there are no other predictions
        assign_empty_label(prediction=prediction, image=image, empty_label=self._empty_label)

        self._execute_post_inference_hooks(image=image, prediction=prediction, name=name)

        return prediction

    def infer_async(self, image: np.ndarray, runtime_data: Any | None = None) -> None:
        """
        Perform asynchronous inference on the `image`.

        **NOTE**: Inference results are not returned directly! Instead, a
        post-inference callback should be defined to handle results, using the
        `.set_asynchronous_callback` method.

        :param image: numpy array representing an image
        :param runtime_data: An optional object containing any additional data
            that should be passed to the asynchronous callback for each infer request.
            This can for example be a timestamp or filename for the image to infer.
            Passing complex objects like a tuple/list or dictionary is also supported.
        """
        self._check_models_loaded()

        if not self.asynchronous_mode:
            raise RuntimeError(
                "No callback function defined to handle asynchronous inference, "
                "please make sure to define a callback using "
                "the `.set_asynchronous_callback()` method of the deployment, "
                "otherwise your inference results will be lost."
            )

        runtime_data_tuple = (image, runtime_data)

        # Note that this method is the same for both single task and pipeline projects
        # Infer requests for downstream models in a pipeline are handled via callbacks
        self._infer_task_async(
            image,
            task=self.project.get_trainable_tasks()[0],
            explain=False,
            runtime_data=runtime_data_tuple,
        )

    def explain(self, image: np.ndarray, name: str | None = None) -> Prediction:
        """
        Run inference on an image for the full model chain in the deployment. The
        resulting prediction will also contain saliency maps and the feature vector
        for the input image.

        :param image: Image to run inference on, as a numpy array containing the pixel
            data. The image is expected to have dimensions [height x width x channels],
            with the channels in RGB order
        :param name: Optional name for the image, if specified this will be used in
            any post inference hooks belonging to the deployment.
        :return: inference results
        """
        self._check_models_loaded()

        if self.asynchronous_mode:
            raise RuntimeError(
                "Unable to use synchronous inference while the deployment is set to "
                "asynchronous execution mode. Please use the `.explain_async` method "
                "instead."
            )

        # Single task inference
        if self.is_single_task:
            prediction = self._infer_task(image, task=self.project.get_trainable_tasks()[0], explain=True)
        # Multi-task inference
        else:
            prediction = self._infer_pipeline(image=image, explain=True)

        self._execute_post_inference_hooks(image=image, prediction=prediction, name=name)
        return prediction

    def explain_async(self, image: np.ndarray, runtime_data: Any | None = None) -> None:
        """
        Perform asynchronous inference on the `image`, and generate saliency maps and
        feature vectors

        **NOTE**: Inference results are not returned directly! Instead, a
        post-inference callback should be defined to handle results, using the
        `.set_asynchronous_callback` method.

        :param image: numpy array representing an image
        :param runtime_data: An optional object containing any additional data
            that should be passed to the asynchronous callback for each infer request.
            This can for example be a timestamp or filename for the image to infer.
            Passing complex objects like a tuple/list or dictionary is also supported.
        """
        self._check_models_loaded()

        if not self.asynchronous_mode:
            raise RuntimeError(
                "No callback function defined to handle asynchronous inference, "
                "please make sure to define a callback using "
                "the `.set_asynchronous_callback()` method of the deployment, "
                "otherwise your inference results will be lost."
            )

        runtime_data_tuple = (image, runtime_data)

        # Note that this method is the same for both single task and pipeline projects
        # Infer requests for downstream models in a pipeline are handled via callbacks
        self._infer_task_async(
            image,
            task=self.project.get_trainable_tasks()[0],
            explain=True,
            runtime_data=runtime_data_tuple,
        )

    def _check_models_loaded(self) -> None:
        """
        Check if models are loaded and ready for inference.

        :raises: ValueError in case models are not loaded
        """
        if not self.are_models_loaded:
            raise ValueError(
                f"Deployment '{self}' is not ready to infer, the inference models are "
                f"not loaded. Please call 'load_inference_models' first."
            )

    def _infer_task(self, image: np.ndarray, task: Task, explain: bool = False) -> Prediction:
        """
        Run pre-processing, inference, and post-processing on the input `image`, for
        the model associated with the `task`.

        :param image: Image to run inference on
        :param task: Task to run inference for
        :param explain: True to get additional outputs for model explainability,
            including saliency maps and the feature vector for the image
        :return: Inference result
        """
        model = self._get_model_for_task(task)
        return model.infer(image, explain)

    def _infer_task_async(
        self,
        image: np.ndarray,
        task: Task,
        explain: bool = False,
        runtime_data: Any | None = None,
    ) -> None:
        """
        Perform asynchronous inference on the `image`.

        **NOTE**: Inference results are not returned directly! Instead, a
        post-inference callback should be defined to handle results, using the
        `.set_asynchronous_callback` method.

        :param image: numpy array representing an image
        :param task: Task to run inference for
        :param explain: True to include saliency maps and feature maps in the returned
            Prediction. Note that these are only available if supported by the model.
        :param runtime_data: An optional object containing any additional data
            that should be passed to the asynchronous callback for each infer request.
            This can for example be a timestamp or filename for the image to infer.
            Passing complex objects like a tuple/list or dictionary is also supported.
        """
        model = self._get_model_for_task(task)
        model.infer_async(image=image, explain=explain, runtime_data=runtime_data)

    def _infer_pipeline(self, image: np.ndarray, explain: bool = False) -> Prediction:
        """
        Run pre-processing, inference, and post-processing on the input `image`, for
        all models in the task chain associated with the deployment.

        Note: If `explain=True`, a saliency map, feature vector and active score for
        the first task in the pipeline will be included in the prediction output

        :param image: Image to run inference on
        :param explain: True to get additional outputs for model explainability,
            including saliency maps and the feature vector for the image
        :return: Inference result
        """
        previous_labels: list[Label] | None = None
        intermediate_result: IntermediateInferenceResult | None = None
        rois: list[ROI] | None = None
        image_views: list[np.ndarray] | None = None

        # Pipeline inference
        for task in self.project.pipeline.tasks[1:]:
            # First task in the pipeline generates the initial result and ROIs
            if task.is_trainable and previous_labels is None:
                task_prediction = self._infer_task(image, task=task, explain=explain)
                rois: list[ROI] | None = None
                if not task.is_global:
                    rois = [ROI.from_annotation(annotation) for annotation in task_prediction.annotations]
                intermediate_result = IntermediateInferenceResult(image=image, prediction=task_prediction, rois=rois)
                previous_labels = [label for label in task.labels if not label.is_empty]

            # Downstream trainable tasks
            elif task.is_trainable:
                if rois is None or image_views is None or intermediate_result is None:
                    raise NotImplementedError(
                        "Unable to run inference for the pipeline in the deployed "
                        "project: A flow control task is required between each "
                        "trainable task in the pipeline."
                    )
                new_rois: list[ROI] = []
                for roi, view in zip(rois, image_views):
                    view_prediction = self._infer_task(view, task)
                    if task.is_global:
                        # Global tasks add their labels to the existing shape in the ROI
                        intermediate_result.extend_annotations(view_prediction.annotations, roi=roi)
                    else:
                        # Local tasks create new shapes in the image coordinate system,
                        # and generate ROI's corresponding to the new shapes
                        for annotation in view_prediction.annotations:
                            intermediate_result.append_annotation(annotation, roi=roi)
                            new_rois.append(ROI.from_annotation(annotation))
                        intermediate_result.rois = [
                            new_roi.to_absolute_coordinates(parent_roi=roi) for new_roi in new_rois
                        ]
                previous_labels = [label for label in task.labels if not label.is_empty]

            # Downstream flow control tasks
            else:
                if previous_labels is None:
                    raise NotImplementedError(
                        f"Unable to run inference for the pipeline in the deployed "
                        f"project: First task in the pipeline after the DATASET task "
                        f"has to be a trainable task, found task of type {task.type} "
                        f"instead."
                    )
                rois, image_views = flow_control(intermediate_result=intermediate_result, task=task)
        return intermediate_result.prediction

    def _get_model_for_task(self, task: Task) -> DeployedModel:
        """
        Get the DeployedModel instance corresponding to the input `task`.

        :param task: Task to get the model for
        :return: DeployedModel corresponding to the task
        """
        try:
            task_index = self.project.get_trainable_tasks().index(task)
        except ValueError as error:
            raise ValueError(
                f"Task {task.title} is not in the list of trainable tasks for project {self.project.name}."
            ) from error
        return self.models[task_index]

    def _remove_temporary_resources(self) -> bool:
        """
        If necessary, clean up any temporary resources associated with the deployment.

        :return: True if temp files have been deleted successfully
        """
        if self._path_to_temp_resources is not None and os.path.isdir(self._path_to_temp_resources):
            try:
                shutil.rmtree(self._path_to_temp_resources)
            except PermissionError:
                logging.warning(
                    f"Unable to remove temporary files for deployment at path "
                    f"`{self._path_to_temp_resources}` because the files are in "
                    f"use by another process. "
                )
                return False
        else:
            logging.debug(
                f"Unable to clean up temporary resources for deployment {self}, "
                f"because the resources were not found on the system. Possibly "
                f"they were already deleted."
            )
            return False
        return True

    def __del__(self):
        """
        If necessary, clean up any temporary resources associated with the deployment.
        This method is called when the Deployment instance is deleted.
        """
        if self._requires_resource_cleanup:
            self._remove_temporary_resources()

    def generate_ovms_config(self, output_folder: str | os.PathLike) -> None:
        """
        Generate the configuration files needed to push the models for the
        `Deployment` instance to OVMS.

        :param output_folder: Target folder to save the configuration files to
        """
        # First prepare the model config list
        if os.path.basename(output_folder) != "ovms_models":
            ovms_models_dir = os.path.join(output_folder, "ovms_models")
        else:
            ovms_models_dir = output_folder
            output_folder = os.path.dirname(ovms_models_dir)
        os.makedirs(ovms_models_dir, exist_ok=True)

        model_configs: list[dict[str, dict[str, Any]]] = []
        for model in self.models:
            # Create configuration entry for model
            model_name = generate_ovms_model_name(project=self.project, model=model, omit_version=True)
            config = {
                "name": model_name,
                "base_path": f"/models/{model_name}",
                "plugin_config": {"PERFORMANCE_HINT": "LATENCY"},
            }
            model_configs.append({"config": config})

            # Copy IR model files to the expected OVMS format
            if model.version is not None:
                model_version = str(model.version)
            else:
                # Fallback to version 1 if no version info is available
                model_version = "1"

            ovms_model_dir = os.path.join(ovms_models_dir, model_name, model_version)

            # Load the model to embed preprocessing for inference with OVMS adapter
            embedded_model = OMZModel.create_model(model=os.path.join(model.model_data_path, "model.xml"))
            embedded_model.save(
                xml_path=os.path.join(ovms_model_dir, "model.xml"),
                bin_path=os.path.join(ovms_model_dir, "model.bin"),
            )
            logging.info(f"Model `{model.name}` prepared for OVMS inference.")

        # Save model configurations
        ovms_config_list = {"model_config_list": model_configs}
        config_target_filepath = os.path.join(ovms_models_dir, "ovms_model_config.json")
        with open(config_target_filepath, "w") as file:
            json.dump(ovms_config_list, file)

        # Copy resource files
        shutil.copy2(OVMS_README_PATH, os.path.join(output_folder))

        logging.info(
            f"Configuration files for OVMS model deployment have been generated in "
            f"directory '{output_folder}'. This folder contains a `OVMS_README.md` "
            f"file with instructions on how to launch OVMS, connect to it and run "
            f"inference. Please follow the instructions outlined there to get started."
        )

    def infer_queue_full(self) -> bool:
        """
        Return True if the queue for asynchronous infer requests is full, False
        otherwise

        :return: True if the infer queue is full, False otherwise
        """
        if not self.asynchronous_mode:
            logging.warning("Method `infer_queue_full()` has no effect in synchronous execution mode")
        # Check the infer queue of the first model in the deployment
        return self.models[0].infer_queue_full()

    def await_all(self) -> None:
        """
        Block execution until all asynchronous infer requests have finished
        processing.

        This means that program execution will resume once the infer queue is empty

        This is a flow control function, it is only applicable when using
        asynchronous inference.
        """
        if not self.asynchronous_mode:
            logging.warning("Method `await_all()` has no effect in synchronous execution mode")
        # Wait until the infer queue of the last model in the task chain is empty
        for model in self.models:
            model.await_all()

    def await_any(self) -> None:
        """
        Block execution until any of the asynchronous infer requests currently in
        the infer queue completes processing

        This means that program execution will resume once a single spot becomes
        available in the infer queue

        This is a flow control function, it is only applicable when using
        asynchronous inference.
        """
        if not self.asynchronous_mode:
            logging.warning("Method `await_any()` has no effect in synchronous execution mode")
        self.models[0].await_any()

    def set_asynchronous_callback(
        self,
        callback_function: Callable[[np.ndarray, Prediction, Any | None], None] | None = None,
    ) -> None:
        """
        Set the callback function to handle asynchronous inference results. This
        function is called whenever a result for an asynchronous inference request
        comes available.

        **NOTE**: Calling this method enables asynchronous inference mode for the
                  deployment. The regular synchronous inference method will no longer
                  be available, unless the deployment is reloaded.

        :param callback_function: Function that should be called to handle
            asynchronous inference results. The function should take the following
            input parameters:

             1. The image/video frame. This is the original image to infer
             2. The inference results (the Prediction). This is the model output for
                the image
             2. Any additional data that will be passed with the infer
                request at runtime. For example, this could be a timestamp for the
                frame, or a title/filepath, etc. This can be in the form of any object:
                You can for instance pass a dictionary, or a tuple/list of multiple
                objects

            **NOTE**: It is possible to call this method without specifying any
                      callback function. In that case, the deployment will be switched
                      to asynchronous mode but only the post-inference hooks will be
                      executed after each infer request
        """

        def post_inference_hook_callback(image: np.ndarray, prediction: Prediction, runtime_data: Any | None):
            name = runtime_data.get("name", None) if isinstance(runtime_data, dict) else None
            self._execute_post_inference_hooks(image=image, prediction=prediction, name=name)

        if callback_function is None:
            final_callback_function = post_inference_hook_callback
        else:

            def callback_and_hook(image, prediction, runtime_data):
                post_inference_hook_callback(image, prediction, runtime_data)
                callback_function(image, prediction, runtime_data)

            final_callback_function = callback_and_hook

        if self.is_single_task:

            def full_callback(result: Prediction, runtime_data: tuple[np.ndarray, Any]):
                image, runtime_user_data = runtime_data
                assign_empty_label(prediction=result, image=image, empty_label=self._empty_label)

                # User defined callback to further process the prediction results
                final_callback_function(image, result, runtime_user_data)

            self.models[0].set_asynchronous_callback(full_callback)
        else:
            # Pipeline case, this is a general implementation accounting for
            # pipelines with more than 2 trainable tasks!
            callbacks: list[Callable[[Prediction, Any], None]] = []
            trainable_tasks = self.project.get_trainable_tasks()

            for task_idx, task in enumerate(self.project.pipeline.tasks):
                if not task.is_trainable:
                    # No callbacks are added for flow control tasks
                    continue

                trainable_task_idx = trainable_tasks.index(task)
                is_first_task = trainable_task_idx == 0
                is_final_task = trainable_task_idx == len(trainable_tasks) - 1

                next_task: Task | None = None
                next_trainable_task: Task | None = None
                add_flow_control = False
                if not is_final_task:
                    next_task = self.project.pipeline.tasks[task_idx + 1]
                    next_trainable_task = trainable_tasks[trainable_task_idx + 1]
                    add_flow_control = next_task.type.is_trainable

                callbacks.append(
                    pipeline_callback_factory(
                        is_first_task=is_first_task,
                        is_final_task=is_final_task,
                        add_flow_control=add_flow_control,
                        current_task=task,
                        deployment=self,
                        next_trainable_task=next_trainable_task,
                        next_task=next_task,
                        final_callback=final_callback_function,
                    )
                )
            for task, callback in zip(trainable_tasks, callbacks):
                model = self._get_model_for_task(task)
                model.set_asynchronous_callback(callback)

        self._async_callback_defined = True
        logging.info("Asynchronous inference mode enabled.")

    @property
    def post_inference_hooks(self) -> list[PostInferenceHookInterface]:
        """
        Return the currently active post inference hooks for the deployment

        :return: list of PostInferenceHook objects
        """
        return self._post_inference_hooks

    def clear_inference_hooks(self) -> None:
        """
        Remove all post inference hooks for the deployment
        """
        n_hooks = len(self.post_inference_hooks)
        self._post_inference_hooks = []
        if n_hooks != 0:
            logging.info(f"Post inference hooks cleared. {n_hooks} hooks were removed successfully")

    def add_post_inference_hook(self, hook: PostInferenceHookInterface) -> None:
        """
        Add a post inference hook, which will be executed after each call to
        `Deployment.infer`

        :param hook: PostInferenceHook to be added to the deployment
        """
        self._post_inference_hooks.append(hook)
        logging.info(f"Hook `{hook}` added.")
        logging.info(f"Deployment now contains {len(self.post_inference_hooks)} post inference hooks.")

    def _execute_post_inference_hooks(self, image: np.ndarray, prediction: Prediction, name: str | None = None) -> None:
        """
        Execute all post inference hooks

        :param image: Numpy image which was inferred
        :param prediction: Prediction for the image
        :param name: Optional name for the image
        """
        timestamp = datetime.datetime.now()
        for hook in self._post_inference_hooks:
            hook.run(image, prediction, name, timestamp)
