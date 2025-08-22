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


from pydantic import BaseModel, ConfigDict, Field

from .hyperparameters.hyperparameters import Hyperparameters


class SubsetSplit(BaseModel):
    """
    Parameters for splitting a dataset into training, validation, and test subsets.
    The sum of training, validation, and test percentages must equal 100.
    """

    training: int = Field(
        ge=1,
        le=100,
        title="Training percentage",
        description="Percentage of data to use for training",
    )
    validation: int = Field(
        ge=1,
        le=100,
        title="Validation percentage",
        description="Percentage of data to use for validation",
    )
    test: int = Field(
        ge=1,
        le=100,
        title="Test percentage",
        description="Percentage of data to use for testing",
    )
    auto_selection: bool = Field(
        title="Auto selection",
        description="Whether to automatically select data for each subset",
    )
    remixing: bool = Field(
        default=False,
        title="Remixing",
        description="Whether to remix data between subsets",
    )


class MinAnnotationPixels(BaseModel):
    """Parameters for minimum annotation pixels."""

    enable: bool = Field(
        default=False,
        title="Enable minimum annotation pixels filtering",
        description="Whether to apply minimum annotation pixels filtering",
    )
    min_annotation_pixels: int = Field(
        gt=0,
        le=200000000,  # reasonable upper limit for pixel count to 200MP
        default=1,
        title="Minimum annotation pixels",
        description="Minimum number of pixels in an annotation",
    )


class MaxAnnotationPixels(BaseModel):
    """Parameters for maximum annotation pixels."""

    enable: bool = Field(
        default=False,
        title="Enable maximum annotation pixels filtering",
        description="Whether to apply maximum annotation pixels filtering",
    )
    max_annotation_pixels: int = Field(
        gt=0,
        default=10000,
        title="Maximum annotation pixels",
        description="Maximum number of pixels in an annotation",
    )


class MinAnnotationObjects(BaseModel):
    """Parameters for maximum annotation objects."""

    enable: bool = Field(
        default=False,
        title="Enable minimum annotation objects filtering",
        description="Whether to apply minimum annotation objects filtering",
    )
    min_annotation_objects: int = Field(
        gt=0,
        default=1,
        title="Minimum annotation objects",
        description="Minimum number of objects in an annotation",
    )


class MaxAnnotationObjects(BaseModel):
    """Parameters for maximum annotation objects."""

    enable: bool = Field(
        default=False,
        title="Enable maximum annotation objects filtering",
        description="Whether to apply maximum annotation objects filtering",
    )
    max_annotation_objects: int = Field(
        gt=0,
        default=10000,
        title="Maximum annotation objects",
        description="Maximum number of objects in an annotation",
    )


class Filtering(BaseModel):
    """Parameters for filtering annotations in the dataset."""

    min_annotation_pixels: MinAnnotationPixels = Field(
        title="Minimum annotation pixels",
        description="Minimum number of pixels in an annotation",
    )
    max_annotation_pixels: MaxAnnotationPixels = Field(
        title="Maximum annotation pixels",
        description="Maximum number of pixels in an annotation",
    )
    min_annotation_objects: MinAnnotationObjects = Field(
        title="Minimum annotation objects",
        description="Minimum number of objects in an annotation",
    )
    max_annotation_objects: MaxAnnotationObjects = Field(
        title="Maximum annotation objects",
        description="Maximum number of objects in an annotation",
    )


class GlobalDatasetPreparationParameters(BaseModel):
    """
    Parameters for preparing a dataset for training within the global configuration.
    Controls data splitting and filtering before being passed for the training.
    """

    subset_split: SubsetSplit = Field(
        title="Subset split",
        description="Configuration for splitting data into subsets",
    )
    filtering: Filtering = Field(title="Filtering", description="Configuration for filtering annotations")


class GlobalParameters(BaseModel):
    """
    Global parameters that are used within the application but are not directly passed to the training backend.
    These parameters still impact the final training outcome by controlling dataset preparation.
    """

    dataset_preparation: GlobalDatasetPreparationParameters = Field(
        title="Dataset preparation", description="Parameters for preparing the dataset"
    )


class TrainingConfiguration(BaseModel):
    """Configuration for model training"""

    task_id: str = Field(title="Task ID", description="Unique identifier for the task")
    model_config = ConfigDict(protected_namespaces=())  # avoid conflict with "model_" namespace
    model_manifest_id: str | None = Field(
        default=None,
        title="Model manifest ID",
        description="ID for the model manifest that defines the supported parameters and capabilities for training",
    )
    global_parameters: GlobalParameters = Field(
        title="Global parameters",
        description="Global configuration parameters for training",
    )
    hyperparameters: Hyperparameters = Field(title="Hyperparameters", description="Hyperparameters for training")
