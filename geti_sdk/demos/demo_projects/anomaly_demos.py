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

import logging

from geti_sdk import Geti
from geti_sdk.annotation_readers import DirectoryTreeAnnotationReader
from geti_sdk.data_models import Project
from geti_sdk.demos.data_helpers import get_mvtec_dataset
from geti_sdk.rest_clients import AnnotationClient, ImageClient, ProjectClient

from .utils import ensure_project_is_trained


def create_anomaly_classification_demo_project(
    geti: Geti,
    n_images: int,
    n_annotations: int = -1,
    project_name: str = "Anomaly demo",
    dataset_path: str | None = None,
) -> Project:
    """
    Create a demo project of type 'anomaly', based off the MVTec
    anomaly detection dataset.

    This method creates a project with a single 'Anomaly' task.

    :param geti: Geti instance, representing the GETi server on which
        the project should be created.
    :param n_images: Number of images that should be uploaded. Pass -1 to upload all
        available images in the dataset for the given labels
    :param n_annotations: Number of images that should be annotated. Pass -1 to
        upload annotations for all images.
    :param project_name: Name of the project to create.
        Defaults to 'Anomaly demo'
    :param dataset_path: Path to the dataset to use as data source. Defaults to
        the 'data' directory in the top level folder of the geti_sdk package. If
        the dataset is not found in the target folder, this method will attempt to
        download it from the internet.
    :return: Project object, holding detailed information about the project that was
        created on the Intel® Geti™ server.
    """
    project_client = ProjectClient(session=geti.session, workspace_id=geti.workspace_id)
    data_path = get_mvtec_dataset(dataset_path)
    logging.info(" ------- Creating anomaly project --------------- ")

    # Create annotation reader
    annotation_reader = DirectoryTreeAnnotationReader(base_data_folder=data_path, subset_folder_names=["train", "test"])
    annotation_reader.group_labels(
        labels_to_group=["damaged_case", "cut_lead", "misplaced", "bent_lead"],
        group_name="Anomalous",
    )
    annotation_reader.group_labels(labels_to_group=["good"], group_name="Normal")

    # Create the project
    project = project_client.create_project(
        project_name=project_name,
        project_type="anomaly_classification",
        labels=[[]],
    )

    # Upload images to the project
    data_filepaths = annotation_reader.get_data_filenames()
    image_client = ImageClient(session=geti.session, workspace_id=geti.workspace_id, project=project)
    images = image_client.upload_from_list(
        path_to_folder=data_path,
        image_names=data_filepaths,
        extension_included=False,
        image_names_as_full_paths=True,
        n_images=n_images,
    )
    annotation_client = AnnotationClient(
        session=geti.session,
        workspace_id=geti.workspace_id,
        project=project,
        annotation_reader=annotation_reader,
    )

    # The annotation_reader is used to upload annotations for all images
    for image, full_path in zip(images, data_filepaths):
        image.name = full_path

    if n_annotations == -1:
        n_annotations = len(images) - 1
    annotation_client.upload_annotations_for_images(images=images[:n_annotations])

    return project


def ensure_trained_anomaly_project(geti: Geti, project_name: str = "Transistor anomaly detection"):
    """
    Check whether the project named `project_name` exists on the server, and create it
    if it not.

    If the project does not exist, this method will create an anomaly detection
    project based on the MVTec AD `transistor` dataset.

    :param geti: Geti instance pointing to the Intel® Geti™ server
    :param project_name: Name of the project to look for or create
    :return: Project object representing the project on the server
    """
    project_client = ProjectClient(session=geti.session, workspace_id=geti.workspace_id)
    project = project_client.get_project_by_name(project_name)
    if project is None:
        logging.info(f"Project '{project_name}' does not exist yet, creating it...")

        project = create_anomaly_classification_demo_project(geti=geti, n_images=-1, project_name=project_name)
        logging.info(f"Project `{project_name}` of type `anomaly` was created on host `{geti.session.config.host}`.")

    ensure_project_is_trained(geti, project)
    return project
