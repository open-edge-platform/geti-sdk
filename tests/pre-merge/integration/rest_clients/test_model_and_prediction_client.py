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
import copy
import os
import time

import numpy as np
import pytest

from geti_sdk.data_models import Image, Prediction, Project
from geti_sdk.data_models.enums import JobState, PredictionMode, SubsetPurpose
from geti_sdk.demos import EXAMPLE_IMAGE_PATH
from geti_sdk.http_session import GetiRequestException
from geti_sdk.platform_versions import GETI_15_VERSION, GETI_22_VERSION
from tests.helpers import (
    DatumAnnotationReader,
    ProjectService,
    SdkTestMode,
    attempt_to_train_task,
    get_or_create_annotated_project_for_test_class,
)
from tests.helpers.constants import PROJECT_PREFIX


class TestModelAndPredictionClient:
    @staticmethod
    def ensure_annotated_project(project_service: ProjectService, annotation_reader: DatumAnnotationReader) -> Project:
        return get_or_create_annotated_project_for_test_class(
            project_service=project_service,
            annotation_readers=[annotation_reader],
            project_type="classification",
            project_name=f"{PROJECT_PREFIX}_model_and_prediction_client",
        )

    @pytest.mark.vcr()
    def test_project_setup_and_get_model_by_job(
        self,
        fxt_project_service: ProjectService,
        fxt_annotation_reader: DatumAnnotationReader,
        fxt_default_labels: list[str],
        fxt_test_mode: SdkTestMode,
    ) -> None:
        """
        Sets up the project and starts training a model as preparation for further tests

        Implements two test cases:
            1. Requesting image prediction while no models trained raises ValueError
            2. Fetching the model resulting from a training job results in valid model
        """
        annotation_reader = fxt_annotation_reader
        annotation_reader.filter_dataset(labels=fxt_default_labels, criterion="XOR")
        project = self.ensure_annotated_project(
            project_service=fxt_project_service, annotation_reader=annotation_reader
        )

        job = attempt_to_train_task(
            training_client=fxt_project_service.training_client,
            task=project.get_trainable_tasks()[0],
            test_mode=fxt_test_mode,
        )

        # Test that requesting a prediction while the project is not ready raises
        # ValueError
        image = fxt_project_service.image_client.get_all_images()[0]
        with pytest.raises(ValueError):
            fxt_project_service.prediction_client.get_image_prediction(image)

        # Monitor train job to make sure the project is train-ready
        timeout = 600 if fxt_test_mode != SdkTestMode.OFFLINE else 1
        interval = 5 if fxt_test_mode != SdkTestMode.OFFLINE else 1
        fxt_project_service.training_client.monitor_jobs([job], timeout=timeout, interval=interval)

        # Test that getting model for the train job works
        if fxt_test_mode == SdkTestMode.OFFLINE:
            job.state = JobState.FINISHED
        model = fxt_project_service.model_client.get_model_for_job(job=job, check_status=False)
        assert model is not None

    @pytest.mark.vcr()
    def test_get_model_group_by_algo_name(
        self,
        fxt_project_service: ProjectService,
    ) -> None:
        """
        Test that the model group can be retrieved by algorithm type
        """
        model_client = fxt_project_service.model_client
        project = fxt_project_service.project
        task = project.get_trainable_tasks()[0]
        default_algo = model_client.supported_algos.get_default_for_task_type(task_type=task.type)

        model_group = fxt_project_service.model_client.get_model_group_by_algo_name(algorithm_name=default_algo.name)

        assert model_group is not None

    @pytest.mark.vcr()
    def test_set_active_model(
        self,
        fxt_project_service: ProjectService,
        fxt_test_mode: SdkTestMode,
    ) -> None:
        """
        Test that the specific model or algorithm can be set as active.
        """
        model_client = fxt_project_service.model_client
        project = fxt_project_service.project
        task = project.get_trainable_tasks()[0]
        default_algorithm = model_client.supported_algos.get_default_for_task_type(task.type)
        default_model = model_client.get_active_model_for_task(task=task)

        unsupported_algorithm_name = "unsupported_algorithm"

        untrained_algos = copy.deepcopy(model_client.supported_algos.get_by_task_type(task.type))
        untrained_algos.remove(default_algorithm)
        untrained_algo = sorted(untrained_algos, key=lambda x: x.stats.gigaflops)[0]

        # Act
        model_client.set_active_model(algorithm=default_algorithm)
        model_client.set_active_model(algorithm=default_algorithm.name)
        model_client.set_active_model(model=default_model)

        with pytest.raises(ValueError):
            model_client.set_active_model()
        with pytest.raises(ValueError):
            model_client.set_active_model(algorithm=untrained_algo)
        with pytest.raises(ValueError):
            model_client.set_active_model(algorithm=unsupported_algorithm_name)

        # Train a model for the new algorithm
        job = attempt_to_train_task(
            training_client=fxt_project_service.training_client,
            task=project.get_trainable_tasks()[0],
            test_mode=fxt_test_mode,
            algorithm=untrained_algo,
        )
        # Monitor train job to make sure the project is train-ready
        timeout = 600 if fxt_test_mode != SdkTestMode.OFFLINE else 1
        interval = 5 if fxt_test_mode != SdkTestMode.OFFLINE else 1
        fxt_project_service.training_client.monitor_jobs([job], timeout=timeout, interval=interval)

        # Set the new algorithm active once a model is trained
        model_client.set_active_model(algorithm=untrained_algo)
        assert model_client.get_active_model_for_task(task=task).architecture == untrained_algo.name
        # Activate the old one again
        model_client.set_active_model(algorithm=default_algorithm)
        assert model_client.get_active_model_for_task(task=task).architecture == default_algorithm.name

    @pytest.mark.vcr()
    def test_get_model_algorithm_task_and_version(
        self,
        fxt_project_service: ProjectService,
    ) -> None:
        """
        Test that the specific model can be retrieved by algorithm, task and model
        version
        """
        model_client = fxt_project_service.model_client
        project = fxt_project_service.project
        task = project.get_trainable_tasks()[0]
        algorithm = model_client.supported_algos.get_default_for_task_type(task.type)

        untrained_algos = copy.deepcopy(model_client.supported_algos.get_by_task_type(task.type))
        default_algo = untrained_algos.get_default_for_task_type(task.type)
        untrained_algos.remove(default_algo)
        untrained_algo = sorted(untrained_algos, key=lambda x: x.stats.gigaflops)[1]

        model_1 = model_client.get_model_by_algorithm_task_and_version(algorithm=algorithm, task=task, version=1)

        model_no_task = model_client.get_model_by_algorithm_task_and_version(algorithm=algorithm, version=1)

        latest_model = model_client.get_model_by_algorithm_task_and_version(
            algorithm=algorithm,
        )

        model_not_trained = model_client.get_model_by_algorithm_task_and_version(algorithm=untrained_algo, task=task)

        model_invalid_version = model_client.get_model_by_algorithm_task_and_version(
            algorithm=algorithm, task=task, version=10
        )

        assert model_1 == model_no_task
        assert latest_model == model_no_task
        assert model_not_trained is None
        assert model_invalid_version is None

    @pytest.mark.vcr()
    def test_download_active_model_for_task(self, fxt_project_service: ProjectService, fxt_temp_directory: str) -> None:
        """
        Test that downloading the active model for a task works.
        """
        model_client = fxt_project_service.model_client
        project = fxt_project_service.project
        task = project.get_trainable_tasks()[0]
        algorithm = model_client.supported_algos.get_default_for_task_type(task.type)

        models_folder_name = "models"
        models_filepath = os.path.join(fxt_temp_directory, models_folder_name)

        model_client.download_active_model_for_task(path_to_folder=fxt_temp_directory, task=task)

        assert os.path.isdir(models_filepath)
        models_content = os.listdir(models_filepath)

        assert len(models_content) >= 3
        assert f"{algorithm.name}_base.zip" in models_content
        assert f"{task.type}_model_details.json" in models_content

        if fxt_project_service.session.version < GETI_15_VERSION:
            mo_model_name = f"{algorithm.name} OpenVINO_MO_optimized.zip"
        else:
            mo_model_name = f"{algorithm.name} OpenVINO FP16_MO_optimized.zip"
        assert mo_model_name in models_content

    def test_prediction_client_set_mode(self, fxt_project_service: ProjectService) -> None:
        """
        Test that changing the prediction mode for the PredictionClient works
        """
        prediction_client = fxt_project_service.prediction_client

        prediction_client.mode = "online"

        assert prediction_client.mode == PredictionMode.ONLINE

    @pytest.mark.vcr()
    def test_predict_image(
        self,
        fxt_project_service: ProjectService,
        fxt_numpy_image: np.ndarray,
        fxt_geti_image: Image,
        fxt_test_mode: SdkTestMode,
    ) -> None:
        """
        Test the 'predict_image' method of the prediction client, for various input
        types
        """
        prediction_client = fxt_project_service.prediction_client
        sleep_time = 10 if fxt_test_mode != SdkTestMode.OFFLINE else 1
        attempts = 10

        prediction_file: Prediction | None = None
        n = 0
        while prediction_file is None and n < attempts:
            try:
                prediction_file = prediction_client.predict_image(image=EXAMPLE_IMAGE_PATH)
                break
            except GetiRequestException as error:
                if error.status_code != 503:
                    raise error
            except TimeoutError:
                pass
            time.sleep(sleep_time)
        prediction_numpy = prediction_client.predict_image(image=fxt_numpy_image)
        prediction_geti_image = prediction_client.predict_image(image=fxt_geti_image)

        assert len(prediction_file.annotations) == len(prediction_numpy.annotations)
        assert len(prediction_numpy.annotations) == len(prediction_geti_image.annotations)

    @pytest.mark.vcr()
    def test_purge_model(
        self,
        fxt_project_service: ProjectService,
        fxt_test_mode: SdkTestMode,
    ) -> None:
        """
        Test that an inactive model may be purged.
        """
        if fxt_project_service.session.version < GETI_22_VERSION:
            pytest.skip("Model purging is not supported in this version")
        # Arrange
        model_client = fxt_project_service.model_client
        project = fxt_project_service.project
        task = project.get_trainable_tasks()[0]

        active_model = model_client.get_active_model_for_task(task=task)
        # Train another model for the active algo
        algo = next(
            algorithm for algorithm in model_client.supported_algos if algorithm.name == active_model.architecture
        )
        job = attempt_to_train_task(
            training_client=fxt_project_service.training_client,
            task=project.get_trainable_tasks()[0],
            test_mode=fxt_test_mode,
            algorithm=algo,
        )
        # Monitor train job to make sure the project is train-ready
        timeout = 600 if fxt_test_mode != SdkTestMode.OFFLINE else 1
        interval = 5 if fxt_test_mode != SdkTestMode.OFFLINE else 1
        fxt_project_service.training_client.monitor_jobs([job], timeout=timeout, interval=interval)
        models_in_group = model_client.get_model_group_by_algo_name(algo.name).models
        models_in_group.sort(key=lambda x: x.version)

        # Act
        assert len(models_in_group) > 1
        active_model = model_client.update_model_detail(models_in_group[-1])
        with pytest.raises(GetiRequestException):
            # Can not archive the active model
            model_client.purge_model(model=models_in_group[-1])
        previous_model = model_client.update_model_detail(models_in_group[-2])
        assert not previous_model.purge_info.is_purged
        model_client.purge_model(model=previous_model)

        # Assert
        purged_model = model_client.update_model_detail(previous_model)
        assert purged_model.purge_info.is_purged

    @pytest.mark.vcr()
    def test_get_training_dataset(
        self,
        fxt_project_service: ProjectService,
        fxt_test_mode: SdkTestMode,
    ) -> None:
        """
        Test that the media in the training dataset for a model con be recovered.
        """
        model = fxt_project_service.model_client.get_all_active_models()[0]
        dataset_client = fxt_project_service.dataset_client

        training_ds_summary = dataset_client.get_training_dataset_summary(model)

        train_set = dataset_client.get_media_in_training_dataset(model, "training")
        val_set = dataset_client.get_media_in_training_dataset(model, "validation")
        test_set = dataset_client.get_media_in_training_dataset(model, "testing")

        n_train_items = training_ds_summary.training_size
        n_val_items = training_ds_summary.validation_size
        n_test_items = training_ds_summary.testing_size

        assert train_set.size == n_train_items
        assert val_set.size == n_val_items
        assert test_set.size == n_test_items

        assert train_set.purpose == SubsetPurpose.TRAINING
        assert val_set.purpose == SubsetPurpose.VALIDATION
        assert test_set.purpose == SubsetPurpose.TESTING
