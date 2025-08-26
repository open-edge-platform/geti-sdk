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

These pages contain the documentation for the main SDK class,
:py:class:`~geti_sdk.geti.Geti`.

The :py:class:`~geti_sdk.geti.Geti` class implements convenience
methods for common operations that can be performed on the Geti™ cluster, such as
creating a project from a pre-existing dataset, downloading or uploading a project,
uploading an image and getting a prediction for it and creating a deployment for a
project.

For example, to download a project, simply do:

.. code-block:: python

   from geti_sdk import Geti

   geti = Geti(
     host="https://0.0.0.0", username="dummy_user", password="dummy_password"
   )

   geti.download_project_data(project_name="dummy_project")

The :py:class:`~geti_sdk.geti.Geti` class provides a high-level interface for
import-export operations in Geti™ platform. Here is a list of these operations:
* Project download
   :py:meth:`~geti_sdk.geti.Geti.download_project_data` method fetches the project data
   and creates a local Python object that supports a range of operations with the project.
* Project upload
   :py:meth:`~geti_sdk.geti.Geti.upload_project_data` method uploads the project data
   from a local Python object to the Geti™ platform.
* Batched project download and upload
   :py:meth:`~geti_sdk.geti.Geti.download_all_projects` and
   :py:meth:`~geti_sdk.geti.Geti.upload_all_projects` methods download and upload
   multiple projects at once.
* Project export
   :py:meth:`~geti_sdk.geti.Geti.export_project` method exports the project snapshot
   to a zip archive. The archive can be used to import the project to another or the same Geti™
   instance.
* Project import
   :py:meth:`~geti_sdk.geti.Geti.import_project` method imports the project from a zip archive.
* Dataset export
   :py:meth:`~geti_sdk.geti.Geti.export_dataset` method exports the dataset to a zip archive.
* Dataset import
   :py:meth:`~geti_sdk.geti.Geti.import_dataset` method imports the dataset from a zip archive
   as a new project.

For custom operations or more fine-grained control over the behavior, the
:py:mod:`~geti_sdk.rest_clients` subpackage should be used.

Module contents
---------------

.. autoclass:: geti_sdk.geti::Geti
   :no-members:

   .. rubric:: Project download and upload

   .. automethod:: download_project_data

   .. automethod:: upload_project_data

   .. automethod:: download_all_projects

   .. automethod:: upload_all_projects

   .. automethod:: import_project

   .. automethod:: export_project

   .. rubric:: Dataset export

   .. automethod:: export_dataset

   .. rubric:: Project creation from dataset

   .. automethod:: create_single_task_project_from_dataset

   .. automethod:: create_task_chain_project_from_dataset

   .. automethod:: import_dataset

   .. rubric:: Project deployment

   .. automethod:: deploy_project

   .. rubric:: Media upload and prediction

   .. automethod:: upload_and_predict_image

   .. automethod:: upload_and_predict_video

   .. automethod:: upload_and_predict_media_folder

"""

from ._version import __version__  # noqa: F401
from .geti import Geti
from .prediction_visualization.visualizer import Visualizer

__all__ = ["Geti", "Visualizer"]
