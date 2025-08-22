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

from enum import Enum
from pprint import pformat
from typing import Any

import attr
from pydantic import BaseModel, Field

from geti_sdk.data_models.enums import TaskType
from geti_sdk.data_models.utils import str_to_optional_enum_converter

from .utils import attr_value_serializer, remove_null_fields


@attr.define
class LegacyAlgorithm:
    """
    Representation of a supported algorithm on the Intel® Geti™ platform.
    """

    model_size: str
    model_template_id: str
    gigaflops: float
    name: str | None = None
    summary: str | None = None
    task_type: str | None = attr.field(default=None, converter=str_to_optional_enum_converter(TaskType))
    supports_auto_hpo: bool | None = None  # Deprecated in Geti v2.7
    default_algorithm: bool | None = None  # Added in Geti v1.16
    performance_category: str | None = None  # Added in Geti v1.9
    lifecycle_stage: str | None = None  # Added in Geti v1.9

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the Algorithm to a dictionary representation.

        :return: Dictionary holding the algorithm data
        """
        return attr.asdict(self, value_serializer=attr_value_serializer)

    @property
    def overview(self) -> str:
        """
        Return a string that shows an overview of the Algorithm properties.

        :return: String holding an overview of the algorithm
        """
        overview_dict = self.to_dict()
        remove_null_fields(overview_dict)
        return pformat(overview_dict)


class GPUMaker(str, Enum):
    """GPU maker names."""

    NVIDIA = "nvidia"
    INTEL = "intel"

    def __str__(self) -> str:
        """Returns the name of the GPU maker."""
        return str(self.name)


class AlgorithmDeprecationStatus(str, Enum):
    """Status of a model architecture with respect to the deprecation process."""

    ACTIVE = "active"  # Model architecture is fully supported, models can be trained
    DEPRECATED = "deprecated"  # Model architecture is deprecated, can still view and train but it's discouraged
    OBSOLETE = "obsolete"  # Model architecture is no longer supported, models can be still viewed but not trained

    def __str__(self) -> str:
        """Returns the name of the model status."""
        return str(self.name)


class PerformanceRatings(BaseModel):
    """Ratings for different performance aspects of a model."""

    accuracy: int = Field(
        ge=1,
        le=3,
        title="Accuracy rating",
        description="Rating of the model accuracy. "
        "The value should be interpreted relatively to the other available models, "
        "and it ranges from 1 (below average) to 3 (above average).",
    )
    training_time: int = Field(
        ge=1,
        le=3,
        title="Training time rating",
        description="Rating of the model training time. "
        "The value should be interpreted relatively to the other available models, "
        "and it ranges from 1 (below average/slower) to 3 (above average/faster).",
    )
    inference_speed: int = Field(
        ge=1,
        le=3,
        title="Inference speed rating",
        description="Rating of the model inference speed. "
        "The value should be interpreted relatively to the other available models, "
        "and it ranges from 1 (below average/slower) to 3 (above average/faster).",
    )


class ModelStats(BaseModel):
    """Information about a machine learning model."""

    gigaflops: float = Field(
        ge=0,
        title="Gigaflops",
        description="Billions of floating-point operations per second required by the model",
    )
    trainable_parameters: float = Field(
        ge=0.0,
        title="Trainable parameters (millions)",
        description="Number of trainable parameters in the model, expressed in millions",
    )
    performance_ratings: PerformanceRatings = Field(
        title="Performance ratings",
        description="Standardized ratings for model performance metrics",
    )


class Capabilities(BaseModel):
    """Model capabilities configuration."""

    xai: bool = Field(
        title="Explainable AI Support",
        description="Whether the model supports explainable AI features",
    )
    tiling: bool = Field(
        title="Tiling Support",
        description="Whether the model supports image tiling for processing large images",
    )


class Algorithm(BaseModel):
    model_manifest_id: str
    task: TaskType
    name: str
    description: str
    stats: ModelStats
    support_status: AlgorithmDeprecationStatus
    supported_gpus: dict[GPUMaker, bool]
    capabilities: Capabilities
    is_default_model: bool
    performance_category: str

    @staticmethod
    def from_legacy_algorithm(legacy_algorithm: LegacyAlgorithm) -> "Algorithm":
        """
        Convert a LegacyAlgorithm to an Algorithm.

        :param legacy_algorithm: LegacyAlgorithm to convert
        :return: Algorithm representation of the legacy algorithm
        """
        return Algorithm(
            model_manifest_id=legacy_algorithm.model_template_id,
            task=legacy_algorithm.task,
            name=legacy_algorithm.name or "Unnamed Algorithm",
            description=legacy_algorithm.summary or "",
            stats=ModelStats(
                gigaflops=legacy_algorithm.gigaflops,
                trainable_parameters=0.0,  # Trainable parameters not available in legacy algorithm
                performance_ratings=PerformanceRatings(
                    accuracy=1,  # Default value, as legacy does not provide ratings
                    training_time=1,
                    inference_speed=1,
                ),
            ),
            support_status=AlgorithmDeprecationStatus[legacy_algorithm.lifecycle_stage],
            supported_gpus={
                GPUMaker.NVIDIA: True,  # Default assumption, as legacy does not specify
                GPUMaker.INTEL: False,  # Default assumption, as legacy does not specify
            },
            capabilities=Capabilities(xai=False, tiling=False),  # Default capabilities
            is_default_model=legacy_algorithm.default_algorithm,
            performance_category=legacy_algorithm.performance_category,
        )
