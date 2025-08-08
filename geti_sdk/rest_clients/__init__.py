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

The `rest_clients` package contains clients for interacting with the various entities (
such as :py:class:`~geti_sdk.data_models.project.Project`,
:py:class:`~geti_sdk.data_models.media.Image` and
:py:class:`~geti_sdk.data_models.model.Model`)
on the Intel® Geti™ server.

The rest clients are initialized with a
:py:class:`~geti_sdk.http_session.geti_session.GetiSession` and a workspace id. The
:py:class:`~geti_sdk.rest_clients.project_client.project_client.ProjectClient`
can be initialized with just that, while all other clients are initialized
*per project* and thus take an additional `project` argument.

For example, to initialize the
:py:class:`~geti_sdk.rest_clients.media_client.image_client.ImageClient` for
a specific project and get a list of all images in the project, the following code
snippet can be used:

.. code-block:: python

   from geti_sdk import Geti
   from geti_sdk.rest_clients import ProjectClient, ImageClient

   geti = Geti(
     host="https://0.0.0.0", username="dummy_user", password="dummy_password"
   )

   project_client = ProjectClient(
       session=geti.session, workspace_id=geti.workspace_id
   )
   project = project_client.get_project_by_name(project_name='dummy_project')

   image_client = ImageClient(
       session=geti.session, workspace_id=geti.workspace_id, project=project
   )
   image_client.get_all_images()

Module contents
---------------

.. autoclass:: geti_sdk.rest_clients.project_client.project_client.ProjectClient
   :members:
   :undoc-members:

.. autoclass:: geti_sdk.rest_clients.dataset_client.DatasetClient
   :members:

.. autoclass:: geti_sdk.rest_clients.media_client.image_client.ImageClient
   :members:

.. autoclass:: geti_sdk.rest_clients.media_client.video_client.VideoClient
   :members:

.. autoclass:: geti_sdk.rest_clients.annotation_clients.annotation_client.AnnotationClient
   :members:

.. autoclass:: geti_sdk.rest_clients.configuration_client.ConfigurationClient
   :members:

.. autoclass:: geti_sdk.rest_clients.prediction_client.PredictionClient
   :members:

.. autoclass:: geti_sdk.rest_clients.model_client.ModelClient
   :members:

.. autoclass:: geti_sdk.rest_clients.training_client.TrainingClient
   :members:

.. autoclass:: geti_sdk.rest_clients.deployment_client.DeploymentClient
   :members:

.. autoclass:: geti_sdk.rest_clients.credit_system_client.CreditSystemClient
   :members:

"""

from .active_learning_client import ActiveLearningClient
from .annotation_clients import AnnotationClient
from .configuration_client import ConfigurationClient
from .configuration_clients import (
    ProjectConfigurationClient,
    TrainingConfigurationClient,
)
from .credit_system_client import CreditSystemClient
from .dataset_client import DatasetClient
from .deployment_client import DeploymentClient
from .media_client import ImageClient, VideoClient
from .model_client import ModelClient
from .prediction_client import PredictionClient
from .project_client import ProjectClient
from .testing_client import TestingClient
from .training_client import TrainingClient

__all__ = [
    "AnnotationClient",
    "ConfigurationClient",
    "DatasetClient",
    "ProjectClient",
    "VideoClient",
    "ImageClient",
    "PredictionClient",
    "ModelClient",
    "TrainingClient",
    "DeploymentClient",
    "ActiveLearningClient",
    "TestingClient",
    "CreditSystemClient",
    "ProjectConfigurationClient",
    "TrainingConfigurationClient",
]
