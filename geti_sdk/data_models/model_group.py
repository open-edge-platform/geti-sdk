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

from datetime import datetime
from typing import ClassVar

import attr

from geti_sdk.data_models.algorithms import Algorithm
from geti_sdk.data_models.model import Model, ModelPurgeInfo
from geti_sdk.data_models.performance import Performance
from geti_sdk.data_models.utils import str_to_datetime


@attr.define
class ModelSummary:
    """
    Representation of a Model on the Intel® Geti™ platform, containing only the
    minimal information about the model.

    :var name: Name of the model
    :var creation_date: Creation date of the model
    :var version: Model version
    :var score: Score that was achieved upon evaluation of the model on the test set
    :var active_model: True if this model was the active model for the project it was
        created in, False if it was not the active model
    :var id: Unique database ID of the model
    :var model_storage_id: Unique database ID of the model storage (also referred to
        as model group) that this model belongs to
    :var label_schema_in_sync: Boolean indicating whether the labels of the model are
        matching with the latest project labels
    """

    _identifier_fields: ClassVar[list[str]] = ["id", "model_storage_id"]

    name: str
    creation_date: str = attr.field(converter=str_to_datetime)
    score_up_to_date: bool | None = None  # Deprecated in Geti 2.6
    purge_info: ModelPurgeInfo | None = None
    size: int | None = None
    version: int | None = None  # 'version' is removed in v1.1
    score: float | None = attr.field(default=None)  # 'score' is removed in v1.1
    performance: Performance | None = None
    active_model: bool = attr.field(default=False)
    id: str | None = attr.field(default=None, repr=False)
    model_storage_id: str | None = attr.field(default=None, repr=False)
    label_schema_in_sync: bool | None = attr.field(default=None)  # Added in Geti 1.1
    lifecycle_stage: str | None = attr.field(default=None)  # Added in Geti v2.8


@attr.define(slots=False)
class ModelGroup:
    """
    Representation of a ModelGroup on the Intel® Geti™ server. A model group is a
    collection of models that all share the same neural network architecture, but may
    have been trained with different training datasets or hyper parameters.
    """

    _identifier_fields: ClassVar[list[str]] = ["id", "task_id"]

    name: str
    model_template_id: str
    models: list[ModelSummary] = attr.field(repr=False)
    task_id: str | None = attr.field(default=None)
    id: str | None = attr.field(default=None)
    learning_approach: str | None = attr.field(default=None)  # Added in Geti v2.5
    lifecycle_stage: str | None = attr.field(default=None)  # Added in Geti v2.6

    def __attrs_post_init__(self) -> None:
        """
        Initialize private attributes.
        """
        self._algorithm: Algorithm | None = None

    @property
    def has_trained_models(self) -> bool:
        """
        Return True if the ModelGroup contains at least one trained model.

        :return: True if the model group holds at least one trained model, False
            otherwise
        """
        trained_models = [model for model in self.models if (model.performance is not None or model.score is not None)]
        return len(trained_models) > 0

    def get_latest_model(self) -> ModelSummary | None:
        """
        Return the latest model in the model group.

        :return: summary information of the most recently model in the model group
        """
        if not self.has_trained_models:
            return None
        creation_dates = [model.creation_date for model in self.models]
        return self.get_model_by_creation_date(max(creation_dates))

    def get_model_by_version(self, version: int) -> ModelSummary:
        """
        Return the model with version `version` in the model group. If no model with
        the version is found, this method raises a ValueError.

        :param version: Number specifying the desired model version
        :return: ModelSummary instance with the specified version, if any
        """
        if not self.has_trained_models:
            return None
        try:
            model = next(model for model in self.models if model.version == version)
        except StopIteration:
            raise ValueError(f"Model with version {version} does not exist in model group {self}")
        return model

    def get_model_by_creation_date(self, creation_date: datetime) -> ModelSummary:
        """
        Return the model created on `creation_date` in the model group. If no model
        by that date is found, this method raises a ValueError

        :param creation_date: Datetime object representing the desired creation_date
        :return: ModelSummary instance with the specified creation_date, if any
        """
        if not self.has_trained_models:
            return None
        try:
            model = next(model for model in self.models if model.creation_date == creation_date)
        except StopIteration:
            raise ValueError(f"Model with creation date {creation_date} does not exist in model group {self}")
        return model

    @property
    def algorithm(self) -> Algorithm | None:
        """
        Return the details for the algorithm corresponding to the ModelGroup
        This property will return None unless the `get_algorithm_details` method is
        called to retrieve the algorithm information from the Intel® Geti™ server

        :return: Algorithm details, if available
        """
        return self._algorithm

    @algorithm.setter
    def algorithm(self, algorithm: Algorithm) -> None:
        """
        Set the algorithm details for this ModelGroup instance.

        :param algorithm: Algorithm information to set
        """
        self._algorithm = algorithm

    def contains_model(self, model: ModelSummary | Model) -> bool:
        """
        Return True if the model group contains the `model`.

        :param model: Model or ModelSummary object
        :return: True if the group contains the model, False otherwise
        """
        return model.id in [model.id for model in self.models]
