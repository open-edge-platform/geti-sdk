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
import time

import pytest
from _pytest.fixtures import FixtureRequest

from geti_sdk import Geti
from geti_sdk.data_models import Project
from geti_sdk.demos import ensure_trained_example_project, get_coco_dataset
from geti_sdk.demos.data_helpers.anomaly_helpers import get_mvtec_dataset, is_ad_dataset
from geti_sdk.demos.data_helpers.coco_helpers import (
    COCOSubset,
    directory_has_coco_subset,
)
from geti_sdk.demos.demo_projects import ensure_project_is_trained
from geti_sdk.demos.demo_projects.coco_demos import DEMO_PROJECT_NAME
from geti_sdk.rest_clients import (
    AnnotationClient,
    ImageClient,
    PredictionClient,
    ProjectClient,
)
from tests.helpers import force_delete_project
from tests.helpers.constants import PROJECT_PREFIX


class TestDemoProjects:
    def test_get_coco_dataset(self, fxt_coco_dataset: str):
        """
        Test that the `get_coco_dataset` method returns the path to a directory
        containing the val2017 subset of the coco dataset.
        """
        assert directory_has_coco_subset(fxt_coco_dataset, COCOSubset.VAL2017)
        # Test that getting the existing dataset results in the correct path
        assert get_coco_dataset() == fxt_coco_dataset

    def test_get_mvtec_dataset(self, fxt_anomaly_dataset: str):
        """
        Test that the `get_mvtec_dataset` method returns the path to a directory
        containing the MVTec AD 'transistor' dataset
        """
        assert is_ad_dataset(fxt_anomaly_dataset)
        # Test that getting the existing dataset results in the correct path
        assert get_mvtec_dataset() == fxt_anomaly_dataset

    @pytest.mark.parametrize(
        "demo_project_fixture_name",
        [
            "fxt_detection_demo_project",
            "fxt_classification_demo_project",
            "fxt_anomaly_classification_demo_project",
            "fxt_segmentation_demo_project",
            "fxt_detection_to_classification_demo_project",
            "fxt_detection_to_segmentation_demo_project",
        ],
        ids=[
            "Detection",
            "Classification",
            "Anomaly",
            "Segmentation",
            "Detection to classification",
            "Detection to segmentation",
        ],
    )
    def test_create_demo_projects(
        self,
        request,
        demo_project_fixture_name: str,
        fxt_geti_no_vcr: Geti,
        fxt_project_client_no_vcr: ProjectClient,
        fxt_demo_images_and_annotations: tuple[int, int],
    ):
        project = request.getfixturevalue(demo_project_fixture_name)
        project_on_server = fxt_project_client_no_vcr.get_project_by_name(project.name)
        image_client = ImageClient(
            session=fxt_geti_no_vcr.session,
            workspace_id=fxt_geti_no_vcr.workspace_id,
            project=project_on_server,
        )
        annotation_client = AnnotationClient(
            session=fxt_geti_no_vcr.session,
            workspace_id=fxt_geti_no_vcr.workspace_id,
            project=project_on_server,
        )
        images = image_client.get_all_images()
        annotations = [annotation_client.get_annotation(image) for image in images]

        assert len(images) == fxt_demo_images_and_annotations[0]
        assert len(annotations) == fxt_demo_images_and_annotations[1]
        for attribute_name in ["name", "pipeline", "creation_time", "id", "creator_id"]:
            assert getattr(project_on_server, attribute_name) == getattr(project, attribute_name)

    def test_ensure_project_is_trained(
        self,
        fxt_geti_no_vcr: Geti,
        fxt_project_client_no_vcr: ProjectClient,
        fxt_detection_demo_project: Project,
    ):
        """
        Test that the `ensure_project_is_trained` function results in a trained project
        """
        prediction_client = PredictionClient(
            session=fxt_geti_no_vcr.session,
            workspace_id=fxt_geti_no_vcr.workspace_id,
            project=fxt_detection_demo_project,
        )
        assert not prediction_client.ready_to_predict

        ensure_project_is_trained(geti=fxt_geti_no_vcr, project=fxt_detection_demo_project)

        assert prediction_client.ready_to_predict

    def test_ensure_trained_example_project(
        self,
        request: FixtureRequest,
        fxt_geti_no_vcr: Geti,
        fxt_project_client_no_vcr: ProjectClient,
        fxt_detection_demo_project: Project,
    ):
        """
        Test the `ensure_trained_example_project` method. Three cases are tested:

        1. Project already exists and is already trained -> project should be returned
            quickly
        2. Project does not exist yet, and does not match default example project name
            -> ValueError should be raised
        3. Project does not exist yet and matches default example project name
            `DEMO_PROJECT_NAME` -> Project should be created and training should start.
        """

        # Case 1: Project exists and is already trained
        t_start = time.time()
        ensure_trained_example_project(geti=fxt_geti_no_vcr, project_name=fxt_detection_demo_project.name)
        assert time.time() - t_start < 10

        # Case 2: Project does not exist and name does not not match the default
        # example project name. ValueError should be raised
        non_existing_project_name = f"{PROJECT_PREFIX}_this_project_does_not_exist"
        any_project = fxt_project_client_no_vcr.get_project_by_name(non_existing_project_name)
        if any_project is not None:
            force_delete_project(
                project=any_project,
                project_client=fxt_project_client_no_vcr,
            )
        assert non_existing_project_name not in [
            project.name for project in fxt_project_client_no_vcr.get_all_projects()
        ]
        with pytest.raises(ValueError):
            ensure_trained_example_project(geti=fxt_geti_no_vcr, project_name=non_existing_project_name)

        # Case 3: Project does not exist and name matches default example project name.
        # Project will be created.
        project = ensure_trained_example_project(geti=fxt_geti_no_vcr, project_name=DEMO_PROJECT_NAME)

        request.addfinalizer(
            lambda proj=project: fxt_project_client_no_vcr.delete_project(project=proj, requires_confirmation=False)
        )
        prediction_client = PredictionClient(
            session=fxt_geti_no_vcr.session,
            workspace_id=fxt_geti_no_vcr.workspace_id,
            project=project,
        )
        assert prediction_client.ready_to_predict
