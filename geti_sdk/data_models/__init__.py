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

"""
Introduction
------------

The `data_models` package contains the SDK representation for all entities in GETi, such
as :py:class:`~geti_sdk.data_models.annotation_scene.AnnotationScene`,
:py:class:`~geti_sdk.data_models.media.Image`,
:py:class:`~geti_sdk.data_models.project.Project` and
:py:class:`~geti_sdk.data_models.model.Model` and many more.

When interacting with the GETi cluster through the
:py:class:`geti_sdk.geti.Geti` or the
:py:mod:`~geti_sdk.rest_clients`, all entities retrieved from the cluster will be
deserialized into the data models defined in this package.

Module contents
---------------

Algorithm-related entities
++++++++++++++++++++++++++

.. automodule:: geti_sdk.data_models.algorithms
   :members:
   :undoc-members:

Project-related entities
++++++++++++++++++++++++

.. automodule:: geti_sdk.data_models.task
   :members:
   :undoc-members:

.. automodule:: geti_sdk.data_models.project
   :members:
   :undoc-members:

Credit System-related entities
++++++++++++++++++++++++

.. automodule:: geti_sdk.data_models.credit_system
   :members:
   :undoc-members:

Annotation-related entities
+++++++++++++++++++++++++++
.. automodule:: geti_sdk.data_models.label
   :members:
   :undoc-members:

.. automodule:: geti_sdk.data_models.shapes
   :members:
   :undoc-members:

.. automodule:: geti_sdk.data_models.annotations
   :members:
   :undoc-members:

.. automodule:: geti_sdk.data_models.annotation_scene
   :members:
   :undoc-members:

Configuration-related entities
++++++++++++++++++++++++++++++

.. automodule:: geti_sdk.data_models.configurable_parameter
   :members:
   :undoc-members:

.. automodule:: geti_sdk.data_models.configurable_parameter_group
   :members:
   :undoc-members:

.. automodule:: geti_sdk.data_models.configuration_identifiers
   :members:
   :undoc-members:

.. automodule:: geti_sdk.data_models.configuration
   :members:
   :undoc-members:

Model-related entities
++++++++++++++++++++++

.. automodule:: geti_sdk.data_models.model
   :members:
   :undoc-members:

.. automodule:: geti_sdk.data_models.model_group
   :members:
   :undoc-members:

Media-related entities
++++++++++++++++++++++

.. automodule:: geti_sdk.data_models.media
   :members:
   :undoc-members:

.. automodule:: geti_sdk.data_models.media_identifiers
   :members:
   :undoc-members:

.. automodule:: geti_sdk.data_models.task_annotation_state
   :members:
   :undoc-members:

Prediction-related entities
+++++++++++++++++++++++++++

.. automodule:: geti_sdk.data_models.predictions
   :members:
   :undoc-members:

Status- and job-related entities
++++++++++++++++++++++++++++++++

.. automodule:: geti_sdk.data_models.status
   :members:
   :undoc-members:

.. automodule:: geti_sdk.data_models.job
   :members:
   :undoc-members:

Deployment-related entities
+++++++++++++++++++++++++++

.. automodule:: geti_sdk.data_model.code_deployment_info
   :members:
   :undoc-members:

Utility functions
+++++++++++++++++

.. automodule:: geti_sdk.data_models.utils
   :members:
   :undoc-members:

Custom container classes
++++++++++++++++++++++++

.. automodule:: geti_sdk.data_models.containers
   :members:
   :undoc-members:

Enumerations
++++++++++++

.. automodule:: geti_sdk.data_models.enums
   :members:
   :undoc-members:

"""

from .algorithms import Algorithm
from .annotation_scene import AnnotationScene
from .annotations import Annotation
from .code_deployment_info import CodeDeploymentInformation
from .configuration import (
    ConfigurableParameters,
    FullConfiguration,
    GlobalConfiguration,
    TaskConfiguration,
)
from .credit_system import CreditAccount, CreditBalance, Subscription
from .dataset import Dataset, Subset, TrainingDatasetStatistics
from .enums import AnnotationKind, MediaType, TaskType
from .job import Job
from .label import Label, ScoredLabel
from .media import Image, MediaItem, Video, VideoFrame
from .model import Model, OptimizedModel
from .model_group import ModelGroup, ModelSummary
from .performance import Performance
from .predictions import Prediction
from .project import Pipeline, Project
from .shapes import Ellipse, Keypoint, Point, Polygon, Rectangle, RotatedRectangle
from .status import ProjectStatus
from .task import Task
from .test_result import Score, TestResult
from .user import User

__all__ = [
    "TaskType",
    "AnnotationKind",
    "Project",
    "Label",
    "Task",
    "Pipeline",
    "Dataset",
    "Image",
    "Video",
    "MediaItem",
    "MediaType",
    "Annotation",
    "AnnotationScene",
    "Algorithm",
    "ScoredLabel",
    "VideoFrame",
    "Prediction",
    "Performance",
    "TaskConfiguration",
    "GlobalConfiguration",
    "ConfigurableParameters",
    "FullConfiguration",
    "Model",
    "ModelGroup",
    "OptimizedModel",
    "ModelSummary",
    "ProjectStatus",
    "Job",
    "CodeDeploymentInformation",
    "TestResult",
    "Score",
    "User",
    "CreditAccount",
    "CreditBalance",
    "Subscription",
    "Subset",
    "TrainingDatasetStatistics",
    "Rectangle",
    "RotatedRectangle",
    "Ellipse",
    "Polygon",
    "Point",
    "Keypoint",
]
