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
import shutil
import tempfile
import time
import zipfile
from collections.abc import Callable
from copy import deepcopy
from typing import Any

import attr
import defusedxml.ElementTree as ET
import numpy as np
from model_api.adapters import OpenvinoAdapter, OVMSAdapter
from model_api.models import Model as model_api_Model
from model_api.tilers import DetectionTiler, InstanceSegmentationTiler, Tiler
from openvino import Core

from geti_sdk.data_models import OptimizedModel, Project
from geti_sdk.data_models.containers import LabelList
from geti_sdk.data_models.enums.domain import Domain
from geti_sdk.data_models.predictions import Prediction, ResultMedium
from geti_sdk.deployment.predictions_postprocessing.results_converter.results_to_prediction_converter import (
    ConverterFactory,
    InferenceResultsToPredictionConverter,
)
from geti_sdk.http_session import GetiSession
from geti_sdk.rest_converters import ModelRESTConverter

from .utils import (
    generate_ovms_model_address,
    generate_ovms_model_name,
    target_device_is_ovms,
)

MODEL_DIR_NAME = "model"
PYTHON_DIR_NAME = "python"

SALIENCY_KEY = "saliency_map"
ANOMALY_SALIENCY_KEY = "anomaly_map"
SEGMENTATION_SALIENCY_KEY = "soft_prediction"
FEATURE_VECTOR_KEY = "feature_vector"

TILER_MAPPING = {
    Domain.DETECTION: DetectionTiler,
    Domain.INSTANCE_SEGMENTATION: InstanceSegmentationTiler,
}

OVMS_TIMEOUT = 10  # Max time to wait for OVMS models to become available


@attr.define
class DeployedModel(OptimizedModel):
    """
    Representation of an Intel® Geti™ model that has been deployed for inference. It
    can be loaded onto a device to generate predictions.
    """

    hyper_parameters: dict | None = attr.field(kw_only=True, repr=False, default=None)

    def __attrs_post_init__(self):
        """
        Initialize private attributes
        """
        super().__attrs_post_init__()
        self._domain: Domain | None = None
        self._model_data_path: str | None = None
        self._model_python_path: str | None = None
        self._needs_tempdir_deletion: bool = False
        self._tempdir_path: str | None = None
        self._labels: LabelList | None = None

        # Attributes related to model explainability
        self._saliency_key: str | None = None
        self._saliency_location: str | None = None
        self._feature_vector_key: str | None = None
        self._feature_vector_location: str | None = None

        self._converter: InferenceResultsToPredictionConverter | None = None
        self._async_callback_defined: bool = False
        self._tiling_enabled: bool = False
        self._tiler: Tiler | None = None

    @property
    def model_data_path(self) -> str:
        """
        Return the path to the raw model data

        :return: path to the directory containing the raw model data
        """
        if self._model_data_path is None:
            raise ValueError("Model data path has not been set yet, location of binary model data is unknown.")
        return self._model_data_path

    def get_data(self, source: str | os.PathLike | GetiSession):
        """
        Load the model weights from a data source. The `source` can be one of the
        following:

          1. The Intel® Geti™ platform (if an GetiSession instance is passed). In this
            case the weights will be downloaded, and extracted to a temporary directory
          2. A zip file on local disk, in this case the weights will be extracted to a
             temporary directory
          3. A folder on local disk containing the .xml and .bin file for the model

        :param source: Data source to load the weights from
        """
        if isinstance(source, os.PathLike | str):
            if os.path.isfile(source) and os.path.splitext(source)[1] == ".zip":
                # Extract zipfile into temporary directory
                if self._model_data_path is None:
                    temp_dir = tempfile.mkdtemp()
                    self._needs_tempdir_deletion = True
                    self._tempdir_path = temp_dir
                else:
                    temp_dir = self._model_data_path

                with zipfile.ZipFile(source, "r") as zipped_source_model:
                    zipped_source_model.extractall(temp_dir)

                # _model_data_path contains the model structure and weights
                # _model_python_path contains the custom model wrappers
                self._model_data_path = os.path.join(temp_dir, MODEL_DIR_NAME)
                self._model_python_path = os.path.join(temp_dir, PYTHON_DIR_NAME)

                self.get_data(temp_dir)

            elif os.path.isdir(source):
                source_contents = os.listdir(source)
                model_dir = os.path.join(source, MODEL_DIR_NAME) if MODEL_DIR_NAME in source_contents else source
                model_dir_contents = os.listdir(model_dir)
                if "model.bin" in model_dir_contents and "model.xml" in model_dir_contents:
                    self._model_data_path = model_dir
                else:
                    raise ValueError(
                        f"Unable to load model data from path '{model_dir}'. Model "
                        f"file 'model.xml' and weights file 'model.bin' were not found "
                        f"at the path specified. "
                    )

                self._model_python_path = os.path.join(source, PYTHON_DIR_NAME)

        elif isinstance(source, GetiSession):
            if self.base_url is None:
                raise ValueError(
                    f"Insufficient data to retrieve data for model {self}. Please set a base_url for the model first."
                )
            response = source.get_rest_response(url=self.base_url + "/export", method="GET", contenttype="zip")
            filename = f"{self.name}_{self.optimization_type}_optimized.zip"
            model_dir = tempfile.mkdtemp()
            model_filepath = os.path.join(model_dir, filename)
            with open(model_filepath, "wb") as f:
                f.write(response.content)
            self._model_data_path = model_dir
            self._needs_tempdir_deletion = True
            self._tempdir_path = model_dir
            self.get_data(source=model_filepath)

    def __del__(self):
        """
        Clean up the temporary directory created to store the model data (if any). This
        method is called when the OptimizedModel object is deleted.
        """
        if self._needs_tempdir_deletion and self._tempdir_path is not None and os.path.exists(self._tempdir_path):
            shutil.rmtree(self._tempdir_path)

    def load_inference_model(
        self,
        device: str = "CPU",
        configuration: dict[str, Any] | None = None,
        project: Project | None = None,
        plugin_configuration: dict[str, str] | None = None,
        max_async_infer_requests: int = 0,
        task_index: int = 0,
    ) -> None:
        """
        Load the actual model weights to a specified device.

        :param device: Device (CPU or GPU) to load the model to. Defaults to 'CPU'
        :param configuration: Optional dictionary holding additional configuration
            parameters for the model
        :param project: Optional project to which the model belongs.
            This is only used when the model is run on OVMS, in that case the
            project is needed to identify the correct model
        :param plugin_configuration: Configuration for the OpenVINO execution mode
            and plugins. This can include for example specific performance hints. For
            further details, refer to the OpenVINO documentation here:
            https://docs.openvino.ai/2022.3/openvino_docs_OV_UG_Performance_Hints.html#doxid-openvino-docs-o-v-u-g-performance-hints
        :param max_async_infer_requests: Maximum number of asynchronous infer request
            that can be processed in parallel. This depends on the properties of the
            target device. If left to 0 (the default), the optimal number of requests
            will be selected automatically.
        :param task_index: Index of the task within the project for which the model is
            trained.
        :return: OpenVino inference engine model that can be used to make predictions
            on images
        """
        core = Core()
        if not target_device_is_ovms(device=device):
            # Run the model locally
            model_adapter = OpenvinoAdapter(
                core,
                model=os.path.join(self.model_data_path, "model.xml"),
                weights_path=os.path.join(self.model_data_path, "model.bin"),
                device=device,
                plugin_config=plugin_configuration,
                max_num_requests=max_async_infer_requests,
            )
            if max_async_infer_requests == 0:
                # Compile model to query optimal infer requests for the device
                compiled_model = core.compile_model(model_adapter.get_model(), device)
                optimal_requests = compiled_model.get_property("OPTIMAL_NUMBER_OF_INFER_REQUESTS")
                model_adapter.max_num_requests = optimal_requests
                compiled_model = None
                logging.info(f"Model `{self.name}` -- Optimal number of infer requests: {optimal_requests}")
        else:
            logging.warning(
                "Model inference through OpenVINO model server is DEPRECATED and will be removed in the future. "
                "Please use local inference instead for more stable results."
            )
            # Connect to an OpenVINO model server instance
            model_name = generate_ovms_model_name(project=project, model=self)
            model_address = generate_ovms_model_address(ovms_address=device, model_name=model_name)

            ovms_connected = False
            ovms_error: BaseException | None = None
            t_start = time.time()
            while not ovms_connected and time.time() - t_start < OVMS_TIMEOUT:
                # If OVMS has just started, model needs some time to initialize
                try:
                    model_adapter = OVMSAdapter(model_address)
                    ovms_connected = True
                except RuntimeError as error:
                    time.sleep(0.5)
                    ovms_error = error
            if not ovms_connected:
                if ovms_error is not None:
                    raise RuntimeError("Unable to connect to OVMS") from ovms_error
                raise RuntimeError("Unknown error encountered while connecting to OVMS")

        # Load model configuration
        config_path = os.path.join(self.model_data_path, "config.json")
        if not os.path.isfile(config_path):
            raise ValueError(
                f"Missing configuration file `config.json` for deployed model `{self}`, unable to load inference model."
            )
        with open(config_path) as config_file:
            configuration_json = json.load(config_file)

        # Update model parameters
        parameters = self.get_model_config()
        if configuration is not None:
            configuration.update(parameters)
        else:
            configuration = parameters

        model_type = configuration.get("model_type")
        # Get label metadata from the project
        self._labels = LabelList.from_project(project=project, task_index=task_index)

        model_api_configuration = self._get_clean_model_config(configuration)

        model = model_api_Model.create_model(
            model=model_adapter,
            model_type=model_type,
            preload=False,
            core=core,
            configuration=model_api_configuration,
        )

        self._inference_model = model

        # Load a Results-to-Prediction converter
        self._domain = Domain.from_task_type(project.get_trainable_tasks()[task_index].type)
        self._converter = ConverterFactory.create_converter(
            self.labels, configuration=configuration, domain=self._domain, model=model
        )

        # Extract tiling parameters, if applicable
        # OTX < 2.0: extract from config.json
        legacy_tiling_parameters = configuration_json.get("tiling_parameters", {})
        tiling_configuration = {}

        enable_tiling = legacy_tiling_parameters.get("enable_tiling", False)
        if enable_tiling:
            try:
                tile_overlap = legacy_tiling_parameters["tile_overlap"]["value"]
                tile_max_number = legacy_tiling_parameters["tile_max_number"]["value"]
                tile_size = legacy_tiling_parameters["tile_size"]["value"]
                tile_ir_scale_factor = legacy_tiling_parameters["tile_ir_scale_factor"]["value"]
                tiling_configuration = {
                    "tile_size": int(tile_size * tile_ir_scale_factor),
                    "tiles_overlap": tile_overlap / tile_ir_scale_factor,
                    "max_pred_number": tile_max_number,
                }
            except KeyError as exc:
                logging.warning(
                    f"Unable to load legacy tiling parameter `{exc.args[0]}` from config.json. "
                    f"Using default tiling parameters."
                )
        else:  # OTX >= 2.0: extract from "rt_info.model_info"
            model_info = model.inference_adapter.get_rt_info("model_info").astype(dict)
            enable_tiling = "tile_size" in model_info
            if enable_tiling:
                tiling_configuration = {
                    "tile_size": model_info["tile_size"].astype(int),
                    "tiles_overlap": model_info["tiles_overlap"].astype(float),
                    "max_pred_number": model_info["max_pred_number"].astype(int),
                }

        if enable_tiling:
            logging.info("Tiling is enabled for this model, initializing Tiler")
            tiler_type = TILER_MAPPING.get(self._domain, None)
            if tiler_type is None:
                raise ValueError(f"Tiling is not supported for domain {self._domain}")
            # InstanceSegmentationTiler supports a `tile_classifier` model, which is
            # used to filter tiles based on their objectness score. The tile
            # classifier model will be exported to the same directory as the instance
            # segmentation model. If it's there, we will load it and add it to the Tiler
            classifier_name = "tile_classifier"
            tiler_arguments = {
                "model": model,
                "execution_mode": "sync",
                "configuration": tiling_configuration,
            }
            if classifier_name in os.listdir(self.model_data_path):
                classifier_path = os.path.join(self.model_data_path, classifier_name)
                tile_classifier_model = model_api_Model.create_model(
                    model=classifier_path + ".xml",
                    weights_path=classifier_path + ".bin",
                    core=core,
                    preload=True,
                )
                tiler_arguments.update({"tile_classifier_model": tile_classifier_model})
            self._tiler = tiler_type(**tiler_arguments)
            self._tiling_enabled = True

        # TODO: This is a workaround to fix the issue that causes the output blob name
        #  to be unset. Remove this once it has been fixed on ModelAPI side
        output_names = list(self._inference_model.outputs.keys())
        if hasattr(self._inference_model, "output_blob_name") and not self._inference_model.output_blob_name:
            self._inference_model.output_blob_name = {name: name for name in output_names}

        # Force reload model to account for any postprocessing changes that may have
        # been applied while creating the ModelAPI wrapper
        logging.info(
            f"Inference model wrapper initialized, force reloading model on device "
            f"`{device}` to finalize inference model initialization process."
        )
        self._inference_model.load(force=True)

    @classmethod
    def from_model_and_hypers(cls, model: OptimizedModel, hyper_parameters: dict | None = None) -> "DeployedModel":
        """
        Create a DeployedModel instance out of an OptimizedModel and it's
        corresponding set of hyper parameters.

        :param model: OptimizedModel to convert to a DeployedModel
        :param hyper_parameters: TaskConfiguration instance containing the hyper
            parameters for the model
        :return: DeployedModel instance
        """
        model_dict = model.to_dict()
        model_dict.update({"hyper_parameters": hyper_parameters})
        model_group_id = model_dict.pop("model_group_id", None)
        deployed_model = cls(**model_dict)
        deployed_model.model_group_id = model_group_id
        try:
            base_url = model.base_url
            deployed_model.base_url = base_url
        except ValueError:
            pass
        return deployed_model

    @classmethod
    def from_folder(cls, path_to_folder: str | os.PathLike) -> "DeployedModel":
        """
        Create a DeployedModel instance from a folder containing the model data.

        :param path_to_folder: Path to the folder that holds the model data
        :return: DeployedModel instance
        """
        config_filepath = os.path.join(path_to_folder, "hyper_parameters.json")
        if os.path.isfile(config_filepath):
            with open(config_filepath) as config_file:
                hparams = json.load(config_file)
        else:
            hparams = None
        model_detail_path = os.path.join(path_to_folder, "model.json")
        with open(model_detail_path) as model_detail_file:
            model_detail_dict = json.load(model_detail_file)
        model = ModelRESTConverter.optimized_model_from_dict(model_detail_dict)
        deployed_model = cls.from_model_and_hypers(model=model, hyper_parameters=hparams)
        deployed_model.get_data(source=path_to_folder)
        return deployed_model

    def save(self, path_to_folder: str | os.PathLike) -> bool:
        """
        Save the DeployedModel instance to the designated folder.

        :param path_to_folder: Path to the folder to save the model to
        :return: True if the model was saved successfully, False otherwise
        """
        if self._model_data_path is None:
            raise ValueError("No model definition and model weights data was found for {self}, unable to save model.")
        os.makedirs(path_to_folder, exist_ok=True, mode=0o770)

        new_model_data_path = os.path.join(path_to_folder, MODEL_DIR_NAME)
        new_model_python_path = os.path.join(path_to_folder, PYTHON_DIR_NAME)

        if new_model_python_path == self._model_python_path or new_model_data_path == self._model_data_path:
            logging.warning(
                f"Model '{self.name}' already exist in target path {path_to_folder}, "
                f"please save to a different location."
            )
            return False

        shutil.copytree(
            src=self._model_data_path,
            dst=new_model_data_path,
            dirs_exist_ok=True,
        )
        if self._model_python_path is not None:
            shutil.copytree(
                src=self._model_python_path,
                dst=new_model_python_path,
                dirs_exist_ok=True,
            )
            self._model_python_path = new_model_python_path

        self._model_data_path = new_model_data_path

        config_filepath = os.path.join(path_to_folder, "hyper_parameters.json")
        with open(config_filepath, "w") as config_file:
            json.dump(self.hyper_parameters, config_file, indent=4)

        model_detail_dict = self.to_dict()
        model_detail_dict.pop("hyper_parameters")
        model_detail_path = os.path.join(path_to_folder, "model.json")
        with open(model_detail_path, "w") as model_detail_file:
            json.dump(model_detail_dict, model_detail_file, indent=4)
        return True

    def _preprocess(self, image: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, tuple[int, int, int]]]:
        """
        Preprocess an image for inference.

        :param image: Numpy array containing pixel data. The image is expected to have
            dimensions [height x width x channels], with the channels in RGB order
        :return: Dictionary holding the preprocessing result, the original shape of
            the image and the shape after preprocessing
        """
        return self._inference_model.preprocess(image)

    def _postprocess(
        self,
        inference_results: dict[str, np.ndarray],
        metadata: dict[str, Any] | None = None,
    ) -> np.ndarray | list[tuple[int, float]] | Any:
        """
        Postprocess the model outputs.

        :param inference_results: Dictionary holding the results of inference
        :param metadata: Dictionary holding metadata
        :return: Postprocessed inference results. The exact format depends on the
            type of model that is loaded:

            * For segmentation models, it will be a numpy array holding the output mask
            * For classification models, it will be a list of tuples holding the
                output class index and class probability
            * For detection models, it will be an instance of
                `openvino.model_zoo.model_api.models.utils.Detection`, holding the
                bounding box output
        """
        return self._inference_model.postprocess(inference_results, metadata)

    def infer(self, image: np.ndarray, explain: bool = False) -> Prediction:
        """
        Run inference on an already preprocessed image.

        :param image: numpy array representing an image
        :param explain: True to include saliency maps and feature maps in the returned
            Prediction. Note that these are only available if supported by the model.
        :return: Dictionary containing the model outputs
        """
        if not self._tiling_enabled:
            preprocessed_image, metadata = self._preprocess(image)
            # metadata is a dict with keys 'original_shape' and 'resized_shape'
            inference_results: dict[str, np.ndarray] = self._inference_model.infer_sync(preprocessed_image)
        else:
            inference_results = self._tiler(image)
            metadata = {"original_shape": image.shape}
        return self._apply_postprocessing_steps(inference_results=inference_results, metadata=metadata, explain=explain)

    def infer_async(
        self,
        image: np.ndarray,
        explain: bool = False,
        runtime_data: Any | None = None,
    ) -> None:
        """
        Perform asynchronous inference on the `image`.

        **NOTE**: Inference results are not returned directly! Instead, a
        post-inference callback should be defined to handle results, using the
        `.set_asynchronous_callback` method.

        :param image: numpy array representing an image
        :param explain: True to include saliency maps and feature maps in the returned
            Prediction. Note that these are only available if supported by the model.
        :param runtime_data: An optional object containing any additional data.
            that should be passed to the asynchronous callback for each infer request.
            This can for example be a timestamp or filename for the image to infer.
            You can for instance pass a dictionary, or a tuple/list of objects.
        """
        if not self._async_callback_defined:
            logging.warning(
                "No callback function defined to handle asynchronous inference, "
                "please make sure to define a callback using "
                "`.set_asynchronous_callback`, otherwise your inference results may be "
                "lost."
            )
        preprocessed_image, metadata = self._preprocess(image)
        self._inference_model.infer_async_raw(
            preprocessed_image,
            {"explain": explain, "metadata": metadata, "runtime_data": runtime_data},
        )

    def set_asynchronous_callback(self, callback_function: Callable[[Prediction, Any | None], None]) -> None:
        """
        Set the callback function to handle asynchronous inference results. This
        function is called whenever a result for an asynchronous inference request
        comes available.

        :param callback_function: Function that should be called to handle
            asynchronous inference results. The function should take the following
            input parameters:

             1. The inference results (the Prediction). This is the primary input
             2. Any additional data that will be passed with the infer
                request at runtime. For example, this could be a timestamp for the
                frame, or a title/filepath, etc. This can be in the form of any object:
                You can for instance pass a dictionary, or a tuple/list of multiple
                objects

        """
        if self._tiling_enabled:
            raise ValueError(
                "Asynchronous inference mode is not supported with models that use "
                "Tiling. Please use the synchronous inference mode instead."
            )

        def full_callback(infer_request, async_metadata: dict[str, Any]):
            # Basic postprocessing, convert to `Prediction` object
            metadata = async_metadata["metadata"]
            explain = async_metadata["explain"]
            runtime_data = async_metadata["runtime_data"]

            raw_result = self._inference_model.inference_adapter.get_raw_result(infer_request)
            prediction = self._apply_postprocessing_steps(raw_result, metadata, explain)
            # User defined callback to further process the prediction results
            callback_function(prediction, runtime_data)

        self._inference_model.inference_adapter.set_callback(full_callback)
        self._async_callback_defined = True

    @property
    def labels(self) -> LabelList:
        """
        Return the Labels for the model.

        This requires the inference model to be loaded, getting this property while
        inference models are not loaded will raise a ValueError

        :return: LabelList containing the SDK labels for the model
        """
        if self._labels is None:
            raise ValueError(
                "Inference model is not loaded, unable to retrieve labels. Please load inference model first."
            )
        return self._labels

    def _apply_postprocessing_steps(
        self, inference_results: Any, metadata: dict[str, Any], explain: bool
    ) -> Prediction:
        """
        Apply the required postprocessing steps to convert the model output to a
        Prediction object, with saliency maps and feature vector included if needed.

        :param inference_results: The results of the model
        :param metadata: Dictionary containing metadata about the original image
        :param explain: True to enable XAI outputs (saliency map and feature vector)
        :return: Prediction object containing the model predictions
        """
        if not self._tiling_enabled:
            postprocessing_results = self._postprocess(inference_results, metadata=metadata)
        else:
            postprocessing_results = inference_results

        prediction = self._converter.convert_to_prediction(
            postprocessing_results, image_shape=metadata["original_shape"]
        )

        # Add optional explainability outputs
        if explain:
            if self.has_xai_head:
                if hasattr(postprocessing_results, "feature_vector"):
                    prediction.feature_vector = postprocessing_results.feature_vector
                result_medium = ResultMedium(name="saliency map", type="saliency map")
                result_medium.data = self._converter.convert_saliency_map(
                    postprocessing_results, image_shape=metadata["original_shape"]
                )
                prediction.maps.append(result_medium)
            else:
                raise ValueError(
                    "Explainability outputs are not available for this model. "
                    "Please ensure the model has an explainability head."
                )
        return prediction

    def infer_queue_full(self) -> bool:
        """
        Return True if the queue for asynchronous infer requests is full, False
        otherwise

        :return: True if the infer queue is full, False otherwise
        """
        return not self._inference_model.inference_adapter.is_ready()

    def await_all(self) -> None:
        """
        Block execution untill all asynchronous infer requests have finished
        processing.

        This means that program execution will resume once the infer queue is empty

        This is a flow control function, it is only applicable when using
        asynchronous inference.
        """
        self._inference_model.inference_adapter.await_all()

    def await_any(self) -> None:
        """
        Block execution untill any of the asynchronous infer requests currently in
        the infer queue completes processing

        This means that program execution will resume once a single spot becomes
        available in the infer queue

        This is a flow control function, it is only applicable when using
        asynchronous inference.
        """
        self._inference_model.inference_adapter.await_any()

    @property
    def asynchronous_mode(self):
        """
        Return True if the DeployedModel is in asynchronous inference mode, False
        otherwise
        """
        return self._async_callback_defined

    @asynchronous_mode.setter
    def asynchronous_mode(self, mode: bool):
        """
        Set the DeployedModel to synchronous or asynchronous inference mode
        """
        if mode:
            if not self._async_callback_defined:
                raise ValueError(
                    "Please use the method `.set_asynchronous_callback()` to define a "
                    "callback and set the DeployedModel to asynchronous inference "
                    "mode."
                )
            logging.debug("DeployedModel is already in asynchronous mode")
        else:

            def do_nothing(request, userdata):
                pass

            self._async_callback_defined = False
            self._inference_model.inference_adapter.set_callback(do_nothing)

    def get_model_config(self) -> dict[str, Any]:
        """
        Return the model configuration as specified in the model.xml metadata file of
        the OpenVINO model

        :return: Dictionary containing the OpenVINO model configuration
        """
        model_xml = os.path.join(self._model_data_path, "model.xml")
        tree = ET.parse(model_xml)
        root = tree.getroot()

        rt_info_node = root.find("rt_info")
        model_info_node = rt_info_node.find("model_info")

        config = {}
        for child in model_info_node:
            value = child.attrib["value"]
            if " " in value and "{" not in value:
                value = value.split(" ")
                value_list = []
                for item in value:
                    if isinstance(item, str) and item.lower() == "none":
                        value_list.append(None)
                        continue
                    if child.tag == "labels":
                        # Do not try to parse labels as floats
                        value_list.append(item)
                        continue
                    try:
                        value_list.append(float(item))
                    except ValueError:
                        value_list.append(item)
                config[child.tag] = value_list
            elif "{" in value:
                # Dictionaries are kept in string representation
                config[child.tag] = value
            else:
                try:
                    value = int(value)
                    config[child.tag] = value
                    continue
                except ValueError:
                    pass
                try:
                    value = float(value)
                    config[child.tag] = value
                    continue
                except ValueError:
                    pass
                if isinstance(value, str) and value.lower() in ["false", "no"]:
                    value = False
                elif isinstance(value, str) and value.lower() in ["true", "yes"]:
                    value = True
                config[child.tag] = value
        return config

    @staticmethod
    def _get_clean_model_config(configuration: dict[str, Any]) -> dict[str, Any]:
        """
        Return a copy of the model configurations with unused values removed.

        :param configuration: Dictionary containing the model configuration to clean
        :return: Copy of the configuration dictionary with unused values removed
        """
        _config = deepcopy(configuration)
        unused_keys = [
            "label_ids",
            "label_info",
            "model_type",
            "optimization_config",
            "task_type",
            "labels",
            "image_shape",
            "domain",
        ]
        for key in unused_keys:
            _config.pop(key, None)
        return _config
