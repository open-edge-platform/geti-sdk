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
import shutil

import pytest

from geti_sdk import Geti
from geti_sdk.data_models import Project
from geti_sdk.demos import (
    create_anomaly_classification_demo_project,
    create_classification_demo_project,
    create_detection_demo_project,
    create_detection_to_classification_demo_project,
    create_detection_to_segmentation_demo_project,
    create_segmentation_demo_project,
    get_coco_dataset,
    get_mvtec_dataset,
)
from geti_sdk.rest_clients import ProjectClient
from tests.helpers import force_delete_project
from tests.helpers.constants import PROJECT_PREFIX


@pytest.fixture(scope="session")
def fxt_demo_images_and_annotations() -> tuple[int, int]:
    """
    Return the number of images and annotations used in the demo project fixtures

    :return: Tuple containing (n_images, n_annotations)
    """
    yield 12, 12


@pytest.fixture(scope="class")
def fxt_anomaly_classification_demo_project(
    fxt_geti_no_vcr: Geti,
    fxt_project_client_no_vcr: ProjectClient,
    fxt_demo_images_and_annotations: tuple[int, int],
) -> Project:
    """
    Create an annotated anomaly detection project on the Geti instance, and
    return the Project object representing it.
    """
    project_name = f"{PROJECT_PREFIX}_anomaly_demo"
    project = create_anomaly_classification_demo_project(
        geti=fxt_geti_no_vcr,
        n_images=fxt_demo_images_and_annotations[0],
        n_annotations=fxt_demo_images_and_annotations[1],
        project_name=project_name,
    )
    yield project
    force_delete_project(
        project,
        project_client=fxt_project_client_no_vcr,
    )


@pytest.fixture(scope="class")
def fxt_segmentation_demo_project(
    fxt_geti_no_vcr: Geti,
    fxt_project_client_no_vcr: ProjectClient,
    fxt_demo_images_and_annotations: tuple[int, int],
) -> Project:
    """
    Create an annotated segmentation project on the Geti instance, and
    return the Project object representing it.
    """
    project_name = f"{PROJECT_PREFIX}_segmentation_demo"
    project = create_segmentation_demo_project(
        geti=fxt_geti_no_vcr,
        n_images=fxt_demo_images_and_annotations[0],
        n_annotations=fxt_demo_images_and_annotations[1],
        project_name=project_name,
    )
    yield project
    force_delete_project(
        project,
        project_client=fxt_project_client_no_vcr,
    )


@pytest.fixture(scope="class")
def fxt_detection_to_classification_demo_project(
    fxt_geti_no_vcr: Geti,
    fxt_project_client_no_vcr: ProjectClient,
    fxt_demo_images_and_annotations: tuple[int, int],
) -> Project:
    """
    Create an annotated detection_to_classification project on the Geti instance, and
    return the Project object representing it.
    """
    project_name = f"{PROJECT_PREFIX}_detection_to_classification_demo"
    project = create_detection_to_classification_demo_project(
        geti=fxt_geti_no_vcr,
        n_images=fxt_demo_images_and_annotations[0],
        n_annotations=fxt_demo_images_and_annotations[1],
        project_name=project_name,
    )
    yield project
    force_delete_project(
        project,
        project_client=fxt_project_client_no_vcr,
    )


@pytest.fixture(scope="class")
def fxt_detection_to_segmentation_demo_project(
    fxt_geti_no_vcr: Geti,
    fxt_project_client_no_vcr: ProjectClient,
    fxt_demo_images_and_annotations: tuple[int, int],
) -> Project:
    """
    Create an annotated detection_to_segmentation project on the Geti instance, and
    return the Project object representing it.
    """
    project_name = f"{PROJECT_PREFIX}_detection_to_segmentation_demo"
    project = create_detection_to_segmentation_demo_project(
        geti=fxt_geti_no_vcr,
        n_images=fxt_demo_images_and_annotations[0],
        n_annotations=fxt_demo_images_and_annotations[1],
        project_name=project_name,
    )
    yield project
    force_delete_project(
        project,
        project_client=fxt_project_client_no_vcr,
    )


@pytest.fixture(scope="class")
def fxt_classification_demo_project(
    fxt_geti_no_vcr: Geti,
    fxt_project_client_no_vcr: ProjectClient,
    fxt_demo_images_and_annotations: tuple[int, int],
) -> Project:
    """
    Create an annotated classification project on the Geti instance, and
    return the Project object representing it.
    """
    project_name = f"{PROJECT_PREFIX}_classification_demo"
    project = create_classification_demo_project(
        geti=fxt_geti_no_vcr,
        n_images=fxt_demo_images_and_annotations[0],
        n_annotations=fxt_demo_images_and_annotations[1],
        project_name=project_name,
    )
    yield project
    force_delete_project(
        project,
        project_client=fxt_project_client_no_vcr,
    )


@pytest.fixture(scope="class")
def fxt_detection_demo_project(
    fxt_geti_no_vcr: Geti,
    fxt_project_client_no_vcr: ProjectClient,
    fxt_demo_images_and_annotations: tuple[int, int],
) -> Project:
    """
    Create an annotated detection project on the Geti instance, and
    return the Project object representing it.
    """
    project_name = f"{PROJECT_PREFIX}_detection_demo"
    project = create_detection_demo_project(
        geti=fxt_geti_no_vcr,
        n_images=fxt_demo_images_and_annotations[0],
        n_annotations=fxt_demo_images_and_annotations[1],
        project_name=project_name,
    )
    yield project
    force_delete_project(
        project,
        project_client=fxt_project_client_no_vcr,
    )


@pytest.fixture(scope="session")
def fxt_coco_dataset(fxt_github_actions_environment: bool):
    """
    Return the path to the coco dataset (subset val2017)
    """
    coco_path = get_coco_dataset()
    yield coco_path
    # If running in CI, clean up the dataset after the test session
    if fxt_github_actions_environment:
        shutil.rmtree(coco_path)


@pytest.fixture(scope="session")
def fxt_anomaly_dataset(fxt_github_actions_environment: bool):
    """
    Return the path to the MVTec AD 'transistor' dataset
    """
    anomaly_path = get_mvtec_dataset()
    yield anomaly_path
    # If running in CI, clean up the dataset after the test session
    if fxt_github_actions_environment:
        shutil.rmtree(anomaly_path)
