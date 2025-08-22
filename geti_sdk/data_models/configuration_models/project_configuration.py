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

from pydantic import BaseModel, Field, model_validator


class TrainConstraints(BaseModel):
    """Constraints applied for model training."""

    min_images_per_label: int = Field(
        ge=0,
        title="Minimum number of images per label",
        description="Minimum number of images that must be present for each label to train",
    )


class AutoTrainingParameters(BaseModel):
    """Configuration for auto-training feature."""

    enable: bool = Field(
        title="Enable auto training",
        description="Whether automatic training is enabled for this task",
    )
    enable_dynamic_required_annotations: bool = Field(
        title="Enable dynamic required annotations",
        description="Whether to dynamically adjust the number of required annotations",
    )
    min_images_per_label: int = Field(
        ge=0,
        title="Minimum images per label",
        description="Minimum number of images needed for each label to trigger auto-training",
    )


class TrainingParameters(BaseModel):
    """Parameters that control the training process."""

    constraints: TrainConstraints = Field(
        title="Training constraints",
        description="Constraints that must be satisfied for training to proceed",
    )


class TaskConfig(BaseModel):
    """Configuration for a specific task within a project."""

    task_id: str = Field(title="Task ID", description="Unique identifier for the task")
    training: TrainingParameters = Field(
        title="Training parameters",
        description="Parameters controlling the training process",
    )
    auto_training: AutoTrainingParameters = Field(
        title="Auto-training parameters",
        description="Parameters controlling auto-training",
    )

    @model_validator(mode="after")
    def task_id_not_empty(self) -> "TaskConfig":
        if not self.task_id:
            raise ValueError("Task ID must be provided as part of the task configuration and cannot be empty.")
        return self


class ProjectConfiguration(BaseModel):
    """
    Configurable parameters for a project.

    Each project has exactly one configuration entity. The ID of this entity
    matches the project ID, as there is a one-to-one relationship between
    projects and their configurations.
    """

    task_configs: list[TaskConfig] = Field(
        title="Task configurations",
        description="List of configurations for all tasks in this project",
    )

    def get_task_config(self, task_id: str) -> TaskConfig:
        """
        Retrieves the configuration for a specific task by its ID.

        :param task_id: The ID of the task to retrieve the configuration for.
        :return: The TaskConfig for the specified task
        :raises ValueError: If the task ID is not found in the project configuration.
        """
        for task_conf in self.task_configs:
            if task_conf.task_id == task_id:
                return task_conf
        raise ValueError(f"Task configuration with ID {task_id} not found.")
