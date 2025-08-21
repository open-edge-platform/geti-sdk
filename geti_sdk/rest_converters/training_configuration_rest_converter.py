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
from collections import defaultdict
from copy import deepcopy
from typing import Any

from geti_sdk.data_models.configuration_models import (
    DatasetPreparationParameters as HyperparametersDatasetPreparationParameters,
)
from geti_sdk.data_models.configuration_models import (
    EvaluationParameters,
    GlobalDatasetPreparationParameters,
    GlobalParameters,
    Hyperparameters,
    TrainingConfiguration,
    TrainingHyperParameters,
)
from geti_sdk.rest_converters.configurable_parameters_rest_converter import (
    ConfigurableParametersRESTConverter,
)

DATASET_PREPARATION = "dataset_preparation"
TRAINING = "training"
EVALUATION = "evaluation"


class TrainingConfigurationRESTConverter(ConfigurableParametersRESTConverter):
    """
    Converter class for transforming TrainingConfiguration objects to/from REST API format.

    This class handles the conversion between TrainingConfiguration domain objects and their
    REST API representations, managing the separation and combination of global parameters
    and hyperparameters across different configuration sections (dataset preparation,
    training, and evaluation).
    """

    @classmethod
    def _dataset_preparation_to_rest(
        cls,
        global_parameters: GlobalParameters | None,
        hyperparameters: Hyperparameters | None,
    ) -> dict[str, Any]:
        """
        Convert dataset preparation parameters from both global and hyperparameters to REST format.

        Merges dataset preparation parameters from global parameters and hyperparameters
        into a single dictionary for REST API consumption. Global parameters take precedence
        over hyperparameters when both contain the same keys.

        :param global_parameters: Global parameters object containing dataset preparation settings,
                                 None if not available
        :param hyperparameters: Hyperparameters object containing dataset preparation settings,
                               None if not available
        :return: Dictionary containing merged dataset preparation parameters for REST API
        :raises ValueError: If the parameter conversion doesn't result in dictionaries
        """
        # Return a combined view of global and hyperparameters for dataset preparation
        global_parameters_rest = (
            cls.configurable_parameters_to_rest(
                configurable_parameters=global_parameters.dataset_preparation,
            )
            if global_parameters and global_parameters.dataset_preparation
            else {}
        )
        hyperparameters_rest = (
            cls.configurable_parameters_to_rest(
                configurable_parameters=hyperparameters.dataset_preparation,
            )
            if hyperparameters and hyperparameters.dataset_preparation
            else {}
        )
        if not isinstance(global_parameters_rest, dict) or not isinstance(hyperparameters_rest, dict):
            raise ValueError("Expected dictionary for global and hyperparameters REST views")
        return global_parameters_rest | hyperparameters_rest

    @classmethod
    def training_configuration_to_rest(cls, training_configuration: TrainingConfiguration) -> dict[str, Any]:
        """
        Convert a TrainingConfiguration object to its REST API representation.

        Transforms the training configuration into a dictionary structure expected by the
        REST API, including task_id, model_manifest_id, and parameter sections for
        dataset_preparation, training, and evaluation.

        :param training_configuration: TrainingConfiguration or PartialTrainingConfiguration
                                     object to convert to REST format
        :return: Dictionary containing the complete REST representation with task_id,
                model_manifest_id, and parameter sections (dataset_preparation, training, evaluation)
        """
        training_params_rest = (
            cls.configurable_parameters_to_rest(
                configurable_parameters=training_configuration.hyperparameters.training,
            )
            if training_configuration.hyperparameters and training_configuration.hyperparameters.training
            else []
        )

        return {
            "task_id": training_configuration.task_id,
            "model_manifest_id": training_configuration.model_manifest_id,
            DATASET_PREPARATION: cls._dataset_preparation_to_rest(
                global_parameters=training_configuration.global_parameters,
                hyperparameters=training_configuration.hyperparameters,
            ),
            TRAINING: training_params_rest,
            EVALUATION: [],  # Evaluation parameters are not yet available
        }

    @classmethod
    def configuration_dict_from_rest(cls, rest_input: dict[str, Any]) -> dict[str, Any]:
        """
        Parse REST input and reorganize it into a structure suitable for Pydantic model creation.

        Separates the flat REST parameter structure into global_parameters and hyperparameters
        sections based on the model field definitions. This method handles the reverse operation
        of training_configuration_to_rest by organizing parameters by their proper domains.

        :param rest_input: REST API dictionary containing parameters organized by sections
                          (dataset_preparation, training, evaluation) plus task_id and model_manifest_id
        :return: Dictionary with 'global_parameters' and 'hyperparameters' keys properly organized
                for TrainingConfiguration model validation, plus any additional fields from rest_input
        """
        rest_input = deepcopy(rest_input)
        dataset_preparation = cls.configurable_parameters_from_rest(rest_input.pop(DATASET_PREPARATION, {}))
        training = cls.configurable_parameters_from_rest(rest_input.pop(TRAINING, {}))
        evaluation = cls.configurable_parameters_from_rest(rest_input.pop(EVALUATION, {}))

        global_parameters: dict = defaultdict(dict)
        hyperparameters: dict = defaultdict(dict)

        for field, _ in GlobalDatasetPreparationParameters.model_fields.items():
            global_parameters[DATASET_PREPARATION][field] = dataset_preparation.pop(field, None)

        for (
            field,
            _,
        ) in HyperparametersDatasetPreparationParameters.model_fields.items():
            hyperparameters[DATASET_PREPARATION][field] = dataset_preparation.pop(field, None)

        for field, _ in TrainingHyperParameters.model_fields.items():
            hyperparameters[TRAINING][field] = training.pop(field, None)

        for field, _ in EvaluationParameters.model_fields.items():
            hyperparameters[EVALUATION][field] = evaluation.pop(field, None)

        # add remaining parameters for validation (extra parameters should not be present)
        global_parameters[DATASET_PREPARATION].update(dataset_preparation)
        hyperparameters[TRAINING].update(training)
        hyperparameters[EVALUATION].update(evaluation)

        # Convert defaultdict to regular dicts for the model validation
        global_parameters = dict(global_parameters)
        hyperparameters = dict(hyperparameters)
        global_parameters.pop("default_factory", None)
        hyperparameters.pop("default_factory", None)

        return {
            "global_parameters": global_parameters,
            "hyperparameters": hyperparameters,
        } | rest_input

    @classmethod
    def training_configuration_from_rest(cls, rest_input: dict[str, Any]) -> TrainingConfiguration:
        """
        Create a TrainingConfiguration object from REST API input.

        Combines the parsing and validation steps to convert a REST API dictionary
        directly into a validated TrainingConfiguration domain object.

        :param rest_input: REST API dictionary containing all configuration data including
                          task_id, model_manifest_id, and parameter sections
        :return: Validated TrainingConfiguration object created from the REST input
        """
        dict_model = cls.configuration_dict_from_rest(rest_input)
        return TrainingConfiguration.model_validate(dict_model)

    @classmethod
    def hyperparameters_from_rest(cls, rest_input: dict[str, Any]) -> Hyperparameters:
        """
        Extract and create a Hyperparameters object from REST API input.

        Parses the REST input to extract only the hyperparameters portion, ignoring
        global parameters and other configuration data. Useful when only hyperparameters
        are needed from a full configuration.

        :param rest_input: REST API dictionary containing configuration parameters
        :return: Validated Hyperparameters object containing training, dataset preparation,
                and evaluation hyperparameters extracted from the REST input
        """
        dict_model = cls.configuration_dict_from_rest(rest_input)["hyperparameters"]
        return Hyperparameters.model_validate(dict_model)
