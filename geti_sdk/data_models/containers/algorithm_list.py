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
from collections import UserList
from collections.abc import Sequence
from typing import Any

from geti_sdk.data_models.algorithms import Algorithm, LegacyAlgorithm
from geti_sdk.data_models.enums import TaskType
from geti_sdk.platform_versions import GetiVersion

DEFAULT_ALGORITHMS = {
    "classification": "Custom_Image_Classification_EfficinetNet-B0",
    "detection": "Custom_Object_Detection_Gen3_ATSS",
    "segmentation": "Custom_Semantic_Segmentation_Lite-HRNet-18-mod2_OCR",
    "anomaly_classification": "ote_anomaly_classification_padim",
    "anomaly_detection": "ote_anomaly_classification_padim",
    "anomaly_segmentation": "ote_anomaly_segmentation_padim",
    "anomaly": "ote_anomaly_classification_padim",
    "rotated_detection": "Custom_Rotated_Detection_via_Instance_Segmentation_MaskRCNN_ResNet50",
    "instance_segmentation": "Custom_Counting_Instance_Segmentation_MaskRCNN_ResNet50",
}


class AlgorithmList(UserList):
    """
    A list containing the algorithms supported in Intel® Geti™.
    """

    def __init__(self, data: Sequence[Algorithm] | None = None):
        self.data: list[Algorithm] = []
        if data is not None:
            super().__init__(list(data))

    @staticmethod
    def from_rest(rest_input: dict[str, Any], geti_version: GetiVersion) -> "AlgorithmList":
        """
        Create an AlgorithmList from the response of the /supported_algorithms REST
        endpoint in Intel® Geti™.

        :param rest_input: Dictionary retrieved from the /supported_algorithms REST endpoint
        :param geti_version: Version of Intel® Geti™ platform
        :return: AlgorithmList holding the information related to the supported algorithms in Intel® Geti™
        """
        algorithm_list = AlgorithmList([])
        if "items" in rest_input:
            algo_rest = rest_input["items"]
        elif "supported_algorithms" in rest_input:
            algo_rest = rest_input["supported_algorithms"]
        else:
            raise KeyError("The input dictionary does not contain the supported algorithms.")
        algo_rest_list = copy.deepcopy(algo_rest)
        for algorithm_dict in algo_rest_list:
            algorithm: Algorithm
            if geti_version.is_configuration_revamped:
                algorithm = Algorithm.model_validate(algorithm_dict)
            else:
                algorithm = Algorithm.from_legacy_algorithm(LegacyAlgorithm(**algorithm_dict))
            algorithm_list.append(algorithm)
        algorithm_list.sort(key=lambda x: x.stats.gigaflops)
        return algorithm_list

    def get_by_model_manifest_id(self, model_manifest_id: str) -> Algorithm:
        """
        Retrieve an algorithm from the list by its model_manifest_id.

        :param model_manifest_id: Name of the model manifest to get the Algorithm
            information for
        :return: Algorithm holding the algorithm details
        """
        for algo in self.data:
            if algo.model_manifest_id == model_manifest_id:
                return algo
        raise ValueError(
            f"Algorithm for model manifest {model_manifest_id} was not found in the list of supported algorithms."
        )

    def get_by_task_type(self, task_type: TaskType) -> "AlgorithmList":
        """
        Return a list of supported algorithms for a particular task type.

        :param task_type: TaskType to get the supported algorithms for
        :return: List of supported algorithms for the task type
        """
        return AlgorithmList([algo for algo in self.data if algo.task == task_type])

    @property
    def summary(self) -> str:
        """
        Return a string that gives a very brief summary of the algorithm list.

        :return: String holding a brief summary of the list of algorithms
        """
        summary_str = "Algorithms:\n"
        for algorithm in self.data:
            summary_str += (
                f"  Name: {algorithm.name}\n"
                f"    Task type: {algorithm.task}\n"
                f"    Model size: {algorithm.stats.trainable_parameters} parameters\n"
                f"    Gigaflops: {algorithm.stats.gigaflops}\n"
                f"    Recommended for: {algorithm.performance_category}\n\n"
            )
        return summary_str

    def get_by_name(self, name: str) -> Algorithm:
        """
        Retrieve an algorithm from the list by its algorithm_name.

        :param name: Name of the Algorithm to get
        :return: Algorithm holding the algorithm details
        """
        for algo in self.data:
            if algo.name == name:
                return algo
        raise ValueError(f"Algorithm named {name} was not found in the list of supported algorithms.")

    def get_default_for_task_type(self, task_type: TaskType) -> Algorithm:
        """
        Return the default algorithm for a given task type. If there is no algorithm
        for the task type in the AlgorithmList, this method will raise a ValueError.

        :param task_type: TaskType of the task to get the default algorithm for
        :raises: ValueError if there are no available algorithms for the specified
            task_type in the AlgorithmList
        :return: Default algorithm for the task
        """
        task_algos = self.get_by_task_type(task_type=task_type)
        default = [algo for algo in task_algos if algo.is_default_model]
        if len(default) == 1:
            return default[0]
        # The old method used in Geti v1.8 and lower. Keep for backwards compatibility
        return self.get_by_model_manifest_id(DEFAULT_ALGORITHMS[str(task_type)])
