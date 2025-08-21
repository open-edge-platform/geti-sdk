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

from typing import Any

from geti_sdk.data_models.model import Model, OptimizedModel
from geti_sdk.data_models.model_group import ModelGroup
from geti_sdk.utils import deserialize_dictionary


class ModelRESTConverter:
    """
    Class that handles conversion of Intel® Geti™ REST output for media entities to
    objects and vice versa.
    """

    @staticmethod
    def model_group_from_dict(input_dict: dict[str, Any]) -> ModelGroup:
        """
        Convert a dictionary representing a model group to a ModelGroup object.

        :param input_dict: Dictionary representing a model group, as returned by the
            Intel® Geti™ /model_groups REST endpoint
        :return: ModelGroup object corresponding to the data in `input_dict`
        """
        return deserialize_dictionary(input_dict, output_type=ModelGroup)

    @staticmethod
    def model_from_dict(input_dict: dict[str, Any]) -> Model:
        """
        Convert a dictionary representing a model to a Model object.

        :param input_dict: Dictionary representing a model, as returned by the
            Intel® Geti™ /model_groups/models REST endpoint
        :return: Model object corresponding to the data in `input_dict`
        """
        model_group_id = input_dict.pop("model_group_id", None)
        model_object = deserialize_dictionary(input_dict, output_type=Model)
        model_object.model_group_id = model_group_id
        return model_object

    @staticmethod
    def optimized_model_from_dict(input_dict: dict[str, Any]) -> OptimizedModel:
        """
        Convert a dictionary representing an optimized model to a OptimizedModel object.

        :param input_dict: Dictionary representing an optimized model, as returned by
            the Intel® Geti™ /model_groups/models REST endpoint
        :return: OptimizedModel object corresponding to the data in `input_dict`
        """
        model_group_id = input_dict.pop("model_group_id", None)
        model_object = deserialize_dictionary(input_dict, output_type=OptimizedModel)
        model_object.model_group_id = model_group_id
        return model_object
