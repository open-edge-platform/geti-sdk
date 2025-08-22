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

import copy
import json
import logging
from pprint import pformat
from typing import Any, ClassVar

import attr

from geti_sdk.data_models.enums import ModelStatus, OptimizationType
from geti_sdk.data_models.utils import (
    attr_value_serializer,
    deidentify,
    remove_null_fields,
    str_to_datetime,
    str_to_enum_converter,
)
from geti_sdk.utils.serialization_helpers import deserialize_dictionary

from .label import Label
from .performance import Performance


@attr.define
class OptimizationCapabilities:
    """
    Representation of the various model optimization capabilities in Intel Geti.
    """

    is_nncf_supported: bool
    is_filter_pruning_supported: bool | None = None
    is_filter_pruning_enabled: bool | None = None


@attr.define
class ModelPurgeInfo:
    """
    Representation of the model soft deletion status. If `is_purged==True`,
    the model binaries (i.e. the trained weights)
    have been deleted from the server storage.
    """

    is_purged: bool
    purge_time: str | None = None
    user_uid: str | None = None


@attr.define
class TrainingFramework:
    """
    Representation of the training framework used to train the model.
    """

    type: str
    version: str


@attr.define
class OptimizationConfigurationParameter:
    """
    Representation of a parameter for model optimization in Intel Geti.
    """

    name: str
    value: Any


@attr.define(slots=False)
class BaseModel:
    """
    Representation of the basic information for a Model or OptimizedModel in Intel Geti
    """

    _identifier_fields: ClassVar[str] = [
        "id",
        "previous_revision_id",
        "previous_trained_revision_id",
    ]

    name: str
    precision: list[str]
    creation_date: str = attr.field(converter=str_to_datetime)
    latency: str | None = None  # Deprecated in Geti 2.6
    fps_throughput: float | None = None  # Deprecated in Geti 2.6
    purge_info: ModelPurgeInfo | None = None
    size: int | None = None
    target_device: str | None = None  # Deprecated in Geti 2.6
    target_device_type: str | None = None  # Deprecated in Geti 2.6
    previous_revision_id: str | None = None
    previous_trained_revision_id: str | None = None
    performance: Performance | None = None
    id: str | None = attr.field(default=None)
    label_schema_in_sync: bool | None = attr.field(default=None)  # Added in Intel Geti 1.1
    total_disk_size: int | None = None  # Added in Intel Geti 2.3
    training_framework: TrainingFramework | None = None  # Added in Intel Geti 2.5
    learning_approach: str | None = None  # Added in Intel Geti v2.6

    def __attrs_post_init__(self):
        """
        Initialize private attributes.
        """
        self._model_group_id: str | None = None
        self._base_url: str | None = None

    @property
    def model_group_id(self) -> str | None:
        """
        Return the unique database ID of the model group to which the model belongs,
        if available.

        :return: ID of the model group for the model
        """
        return self._model_group_id

    @model_group_id.setter
    def model_group_id(self, id_: str):
        """
        Set the model group id for this model.

        :param id_: ID to set
        """
        self._model_group_id = id_

    @property
    def base_url(self) -> str | None:
        """
        Return the base url that can be used to get the model details, download the
        model, etc., if available.

        :return: base url at which the model can be addressed. The url is defined
            relative to the ip address or hostname of the Intel® Geti™ server
        """
        if self._base_url is not None:
            return self._base_url
        raise ValueError(
            f"Insufficient data to determine base url for model {self}. Please "
            f"make sure that property `base_url` is set first."
        )

    @base_url.setter
    def base_url(self, base_url: str):
        """
        Set the base url that can be used to get the model details, download the
        model, etc.

        :param base_url: base url at which the model can be addressed
        :return:
        """
        if self.model_group_id is not None:
            if self.model_group_id in base_url:
                if base_url.endswith(f"models/{self.id}"):
                    base_url = base_url
                else:
                    base_url += f"/models/{self.id}"
            else:
                base_url += f"/{self.model_group_id}/models/{self.id}"
        else:
            base_url = base_url
        if hasattr(self, "optimized_models"):
            for model in self.optimized_models:
                model._base_url = base_url + f"/optimized_models/{model.id}"
        self._base_url = base_url

    def to_dict(self) -> dict[str, Any]:
        """
        Return the dictionary representation of the model.

        :return:
        """
        base_dict = attr.asdict(self, recurse=True, value_serializer=attr_value_serializer)
        base_dict["model_group_id"] = self.model_group_id
        return base_dict

    @property
    def overview(self) -> str:
        """
        Return a string that represents an overview of the model.

        :return:
        """
        deidentified = copy.deepcopy(self)
        deidentified.deidentify()
        overview_dict = deidentified.to_dict()
        remove_null_fields(overview_dict)
        return pformat(overview_dict)

    def deidentify(self) -> None:
        """
        Remove unique database IDs from the BaseModel.
        """
        deidentify(self)


@attr.define(slots=False)
class OptimizedModel(BaseModel):
    """
    Representation of an OptimizedModel in Intel® Geti™. An optimized model is a trained model
    that has been converted OpenVINO representation. This conversion may involve weight
    quantization, filter pruning, or other optimization techniques supported by
    OpenVINO.
    """

    model_status: str = attr.field(kw_only=True, converter=str_to_enum_converter(ModelStatus))
    optimization_methods: list[str] = attr.field(kw_only=True)
    optimization_objectives: dict[str, Any] = attr.field(kw_only=True)
    optimization_type: str = attr.field(kw_only=True, converter=str_to_enum_converter(OptimizationType))
    version: int | None = attr.field(kw_only=True, default=None)
    configurations: list[OptimizationConfigurationParameter] | None = attr.field(
        kw_only=True, default=None
    )  # Added in Intel Geti v1.4
    model_format: str | None = None  # Added in Intel Geti v1.5
    has_xai_head: bool = False  # Added in Intel Geti v1.5


@attr.define(slots=False)
class Model(BaseModel):
    """
    Representation of a trained Model in Intel® Geti™.
    """

    architecture: str = attr.field(kw_only=True)
    score_up_to_date: bool | None = attr.field(default=None, kw_only=True)  # Deprecated in Geti 2.6
    optimized_models: list[OptimizedModel] = attr.field(kw_only=True)
    # Removed in Geti 2.2
    optimization_capabilities: OptimizationCapabilities | None = attr.field(default=None, kw_only=True)
    labels: list[Label] | None = None
    version: int | None = attr.field(default=None, kw_only=True)
    # 'version' is deprecated in v1.1 -- IS IT?
    training_dataset_info: dict[str, str] | None = None

    @property
    def model_group_id(self) -> str | None:
        """
        Return the unique database ID of the model group to which the model belongs,
        if available.

        :return: ID of the model group for the model
        """
        return self._model_group_id

    @model_group_id.setter
    def model_group_id(self, id_: str):
        """
        Set the model group id for this model.

        :param id: ID to set
        """
        self._model_group_id = id_
        for model in self.optimized_models:
            model.model_group_id = id_

    @classmethod
    def from_dict(cls, model_dict: dict[str, Any]) -> "Model":
        """
        Create a Model instance from a dictionary holding the model data.

        :param model_dict: Dictionary representing a model
        :return: Model instance reflecting the data contained in `model_dict`
        """
        return deserialize_dictionary(model_dict, cls)

    @classmethod
    def from_file(cls, filepath: str) -> "Model":
        """
        Create a Model instance from a .json file holding the model data.

        :param filepath: Path to a json file holding the model data
        :return:
        """
        with open(filepath) as file:
            model_dict = json.load(file)
        return cls.from_dict(model_dict=model_dict)

    def get_optimized_model(
        self,
        optimization_type: str | None = None,
        precision: str | None = None,
        require_xai: bool = False,
    ) -> OptimizedModel | None:
        """
        Return the OptimizedModel of the specified `optimization_type` or `precision`.
        The following optimization types are supported: 'nncf', 'pot', 'onnx', 'mo',
        `openvino`
        (case insensitive). The supported precision levels are `FP32`, `FP16`, `INT8`.
        Precision, optimization type or both can be specified.

        :param optimization_type: Optimization type for which to return the model
        :param precision: Precision level for which to return the model. Can be `FP32`,
            `FP16` or `INT8`
        :param require_xai: If True, only include models that have an XAI head for
            saliency map generation. Defaults to False
        :return: OptimizedModel object representing the optimized model
        """
        if optimization_type is None and precision is None:
            raise ValueError("Please specify optimization_type or precision, or both")
        optimized_models: list[OptimizedModel] = []
        if optimization_type is not None:
            capitalized_ot = optimization_type.upper()
            if capitalized_ot == "OPENVINO":
                optimized_models = [model for model in self.optimized_models if "ONNX" not in model.name]
            else:
                allowed_types = [item.name for item in OptimizationType]
                if capitalized_ot not in allowed_types:
                    raise ValueError(
                        f"Invalid optimization type passed, supported values are {allowed_types} or `openvino`"
                    )
                optimization_type = OptimizationType(capitalized_ot)
                optimized_models = [
                    model for model in self.optimized_models if model.optimization_type == optimization_type
                ]

        if precision is not None:
            models_to_search = self.optimized_models if len(optimized_models) == 0 else optimized_models
            optimized_models = [model for model in models_to_search if precision in model.name]

        if len(optimized_models) == 0:
            logging.info("No optimized model meeting the optimization criteria was found.")
            return None
        if len(optimized_models) == 1:
            if not require_xai:
                return optimized_models[0]
            if optimized_models[0].has_xai_head:
                return optimized_models[0]
            logging.info(
                f"An optimized model of type {optimization_type} was found, but it does "
                f"not include an XAI head. Method `get_optimized_model` returned "
                f"None."
            )
            return None
        models_to_check = optimized_models if not require_xai else [m for m in optimized_models if m.has_xai_head]
        if len(models_to_check) == 0:
            logging.info(
                f"An optimized model of type {optimization_type} was found, but it does "
                f"not include an XAI head. Method `get_optimized_model` returned "
                f"None."
            )
            return None
        creation_dates = [om.creation_date for om in models_to_check]
        max_index = creation_dates.index(max(creation_dates))
        return models_to_check[max_index]
