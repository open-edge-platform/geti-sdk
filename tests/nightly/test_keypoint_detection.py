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
from test_nightly_project import TestNightlyProject

from geti_sdk import Geti
from tests.helpers import DatumAnnotationReader, ProjectService


class TestKeypointDetection(TestNightlyProject):
    """
    Class to test project creation, annotation upload, training, prediction and
    deployment for a keypoint detection project
    """

    PROJECT_TYPE = "keypoint_detection"
    __test__ = True

    def test_project_setup(
        self,
        fxt_project_service_no_vcr: ProjectService,
        fxt_annotation_reader_keypoint: DatumAnnotationReader,
        fxt_annotation_reader_grouped_keypoint: DatumAnnotationReader,
        fxt_learning_parameter_settings: str,
    ):
        """
        This test sets up an annotated project on the server, that persists for the
        duration of this test class.
        """
        super().test_project_setup(
            fxt_project_service_no_vcr,
            fxt_annotation_reader_keypoint,
            fxt_annotation_reader_grouped_keypoint,
            fxt_learning_parameter_settings,
        )

    def test_monitor_jobs(self, fxt_project_service_no_vcr: ProjectService):
        """
        For anomaly projects, the training is run in the project_setup
        phase. No need to monitor jobs.
        """
        super().test_monitor_jobs(fxt_project_service_no_vcr)

    def test_upload_and_predict_image(
        self,
        fxt_project_service_no_vcr: ProjectService,
        fxt_image_path: str,
        fxt_geti_no_vcr: Geti,
    ):
        pass

    def test_deployment(
        self,
        fxt_project_service_no_vcr: ProjectService,
        fxt_geti_no_vcr: Geti,
        fxt_temp_directory: str,
        fxt_image_path: str,
        fxt_image_path_complex: str,
        fxt_artifact_directory: str,
    ):
        pass
