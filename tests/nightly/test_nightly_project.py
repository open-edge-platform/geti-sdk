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
import os
import time
from typing import ClassVar, List, Optional

import cv2
import numpy as np

from geti_sdk import Geti
from geti_sdk.data_models import Job, Prediction
from geti_sdk.data_models.enums import JobState
from geti_sdk.http_session import GetiRequestException
from tests.helpers import (
    DatumAnnotationReader,
    ProjectService,
    get_or_create_annotated_project_for_test_class,
    plot_predictions_side_by_side,
)
from tests.helpers.constants import PROJECT_PREFIX


class TestNightlyProject:
    PROJECT_TYPE: ClassVar[str] = "none"

    # Setting __test__ to False indicates to pytest that this class is not part of
    # the tests. This allows it to be imported in other test files.
    __test__: ClassVar[bool] = False

    def test_project_setup(
        self,
        fxt_project_service_no_vcr: ProjectService,
        fxt_annotation_reader: DatumAnnotationReader,
        fxt_annotation_reader_grouped: DatumAnnotationReader,
        fxt_learning_parameter_settings: str,
    ):
        """
        This test sets up an annotated project on the server, that persists for the
        duration of this test class.
        """
        if self.PROJECT_TYPE == "classification":
            fxt_annotation_reader.filter_dataset(
                labels=["cube", "cylinder"], criterion="XOR"
            )

        annotation_readers = [fxt_annotation_reader]
        if "_to_" in self.PROJECT_TYPE:
            annotation_readers = [fxt_annotation_reader_grouped, fxt_annotation_reader]

        get_or_create_annotated_project_for_test_class(
            project_service=fxt_project_service_no_vcr,
            annotation_readers=annotation_readers,
            project_type=self.PROJECT_TYPE,
            project_name=f"{PROJECT_PREFIX}_nightly_{self.PROJECT_TYPE}",
            enable_auto_train=True,
            learning_parameter_settings=fxt_learning_parameter_settings,
            annotation_requirements_first_training=6,
        )

    def test_monitor_jobs(self, fxt_project_service_no_vcr: ProjectService):
        """
        This test monitors training jobs for the project, and completes when the jobs
        are finished
        """
        training_client = fxt_project_service_no_vcr.training_client
        max_attempts = 10
        jobs: List[Job] = []
        n = 0
        # Wait for a while, giving the server time to initialize the jobs
        time.sleep(30)
        while len(jobs) == 0 and n < max_attempts:
            jobs = training_client.get_jobs(project_only=True)
            n += 1
            # If no jobs are found yet, wait for a while and retry
            time.sleep(10)

        if len(jobs) == 0 and n == max_attempts:
            raise RuntimeError(
                f"No auto-train job was started on the platform for project "
                f"'{fxt_project_service_no_vcr.project.name}'. Test failed."
            )

        jobs = training_client.monitor_jobs(jobs=jobs, timeout=10000)
        for job in jobs:
            # We allow scheduled jobs to pass, sometimes an auto-training job for a
            # task chain project gets scheduled twice. In that case one of them will
            # never execute. This will cause the test to fail, even though it's not an
            # SDK issue. By allowing 'scheduled' state, this case passes
            assert job.state in [JobState.FINISHED, JobState.SCHEDULED]

    def test_upload_and_predict_image(
        self,
        fxt_project_service_no_vcr: ProjectService,
        fxt_image_path: str,
        fxt_geti_no_vcr: Geti,
    ):
        """
        Tests uploading and predicting an image to the project. Waits for the
        inference servers to be initialized.
        """
        # First make sure that all jobs for the project are finished
        training_client = fxt_project_service_no_vcr.training_client
        timeout = 900
        t_start = time.time()
        training = training_client.is_training()
        while training and time.time() - t_start < timeout:
            training = training_client.is_training()
            time.sleep(10)

        n_attempts = 3
        project = fxt_project_service_no_vcr.project

        prediction: Optional[Prediction] = None
        request_exception: Optional[Exception] = None
        for j in range(n_attempts):
            try:
                image, prediction = fxt_geti_no_vcr.upload_and_predict_image(
                    project=project,
                    image=fxt_image_path,
                    visualise_output=False,
                    delete_after_prediction=False,
                )
            except GetiRequestException as error:
                prediction = None
                time.sleep(20)
                logging.debug(error)
                request_exception = error
            if prediction is not None:
                assert isinstance(prediction, Prediction)
                break
        if prediction is None:
            raise request_exception

    def test_deployment(
        self,
        fxt_project_service_no_vcr: ProjectService,
        fxt_geti_no_vcr: Geti,
        fxt_temp_directory: str,
        fxt_image_path: str,
        fxt_image_path_complex: str,
        fxt_artifact_directory: str,
    ):
        """
        Tests local deployment for the project. Compares the local prediction to the
        platform prediction for a sample image. Test passes if they are equal
        """
        project = fxt_project_service_no_vcr.project

        deployment_folder = os.path.join(fxt_temp_directory, project.name)
        deployment = fxt_geti_no_vcr.deploy_project(
            project=project,
            output_folder=deployment_folder,
            enable_explainable_ai=True,
        )

        assert os.path.isdir(os.path.join(deployment_folder, "deployment"))
        deployment.load_inference_models(device="CPU")

        images = {"simple": fxt_image_path, "complex": fxt_image_path_complex}

        for image_name, image_path in images.items():
            image_bgr = cv2.imread(image_path)
            image_np = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

            local_prediction = deployment.infer(image_np)
            assert isinstance(local_prediction, Prediction)
            image, online_prediction = fxt_geti_no_vcr.upload_and_predict_image(
                project,
                image=image_bgr,
                delete_after_prediction=True,
                visualise_output=False,
            )

            explain_prediction = deployment.explain(image_np)
            if "anomaly" not in self.PROJECT_TYPE:
                if all([model.has_xai_head for model in deployment.models]):
                    assert explain_prediction.feature_vector is not None
            assert len(explain_prediction.maps) > 0

            online_mask = online_prediction.as_mask(image.media_information)
            local_mask = local_prediction.as_mask(image.media_information)

            assert online_mask.shape == local_mask.shape
            equal_masks = np.all(local_mask == online_mask)
            if not equal_masks:
                logging.warning("Local and online prediction masks are not equal!")
                logging.info(
                    f"Number of shapes: {len(local_prediction.annotations)} - local   "
                    f"----    {len(online_prediction.annotations)} - online."
                )

            logging.info("\n\n-------- Local prediction --------")
            logging.info(local_prediction.overview)
            logging.info("\n\n-------- Online prediction --------")
            logging.info(online_prediction.overview)

            # Save the predictions as test artifacts
            predictions_dir = os.path.join(fxt_artifact_directory, "predictions")
            if not os.path.isdir(predictions_dir):
                os.makedirs(predictions_dir)

            image_path = os.path.join(
                predictions_dir, project.name + "_" + image_name + ".jpg"
            )
            plot_predictions_side_by_side(
                image,
                prediction_1=local_prediction,
                prediction_2=online_prediction,
                filepath=image_path,
            )
