# Copyright (C) 2024 Intel Corporation
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

import csv
from pathlib import Path

import numpy as np
import pytest
from pytest_mock import MockerFixture

from geti_sdk import Geti
from geti_sdk.benchmarking import Benchmarker
from geti_sdk.data_models import Project


class TestBenchmarker:
    def test_initialize(
        self,
        fxt_mocked_geti: Geti,
        fxt_classification_project: Project,
        mocker: MockerFixture,
    ):
        # Arrange
        mock_get_project = mocker.patch(
            "geti_sdk.geti.Geti.get_project",
            return_value=fxt_classification_project,
        )
        mocked_model_client = mocker.patch("geti_sdk.benchmarking.benchmarker.ModelClient")
        mocked_training_client = mocker.patch("geti_sdk.benchmarking.benchmarker.TrainingClient")
        project_mock = mocker.MagicMock()
        algorithms_to_benchmark = ("ALGO_1", "ALGO_2")
        precision_levels = ("PRECISION_1", "PRECISION_2")
        images = ("path_1", "path_2")
        videos = ("path_3", "path_4")

        # Act
        # Single task project, benchmarking on images
        benchmarker = Benchmarker(
            geti=fxt_mocked_geti,
            project=project_mock,
            algorithms=algorithms_to_benchmark,
            precision_levels=precision_levels,
            benchmark_images=images,
        )

        # Assert
        mock_get_project.assert_called_once_with(project_id=project_mock.id)
        mocked_model_client.assert_called_once()
        mocked_training_client.assert_called_once()
        assert benchmarker._is_single_task
        assert not benchmarker._are_models_specified
        assert benchmarker.precision_levels == precision_levels
        assert benchmarker._algorithms == algorithms_to_benchmark

        # Act 2
        # Single task project, videos and images
        with pytest.raises(ValueError):
            benchmarker = Benchmarker(
                geti=fxt_mocked_geti,
                project=project_mock,
                algorithms=algorithms_to_benchmark,
                precision_levels=precision_levels,
                benchmark_images=images,
                benchmark_video=videos,
            )

    def test_initialize_task_chain(
        self,
        fxt_mocked_geti: Geti,
        fxt_detection_to_classification_project: Project,
        mocker: MockerFixture,
    ):
        # Arrange
        mocker.patch(
            "geti_sdk.geti.Geti.get_project",
            return_value=fxt_detection_to_classification_project,
        )
        fetched_images = (mocker.MagicMock(),)
        mock_image_client_get_all = mocker.patch(
            "geti_sdk.geti.ImageClient._get_all",
            return_value=fetched_images,
        )
        model_client_object_mock = mocker.MagicMock()
        mocked_model_client = mocker.patch(
            "geti_sdk.benchmarking.benchmarker.ModelClient",
            return_value=model_client_object_mock,
        )
        active_models = (mocker.MagicMock(), mocker.MagicMock())
        model_client_object_mock.get_all_active_models.return_value = active_models

        mocked_training_client = mocker.patch("geti_sdk.benchmarking.benchmarker.TrainingClient")
        project_mock = mocker.MagicMock()
        precision_levels = ["PRECISION_1", "PRECISION_2"]

        # Act
        # Multi task project, no media provided
        benchmarker = Benchmarker(
            geti=fxt_mocked_geti,
            project=project_mock,
            precision_levels=precision_levels,
        )

        # Assert
        mock_image_client_get_all.assert_called_once()
        mocked_model_client.assert_called_once()
        model_client_object_mock.get_all_active_models.assert_called_once()
        model_client_object_mock.get_task_for_model.assert_called_with(model=active_models[1])
        mocked_training_client.assert_called_once()
        assert not benchmarker._is_single_task
        assert benchmarker._task_chain_algorithms == [
            [active_models[0].architecture],
            [active_models[1].architecture],
        ]
        assert benchmarker._task_chain_models == [
            [active_models[0]],
            [active_models[1]],
        ]
        assert tuple(benchmarker.images) == fetched_images
        assert benchmarker._are_models_specified

    def test_set_task_chain_models(self, fxt_benchmarker_task_chain: Benchmarker, mocker: MockerFixture):
        # Arrange
        models_task_1 = (mocker.MagicMock(), mocker.MagicMock())
        models_task_2 = (mocker.MagicMock(), mocker.MagicMock())

        # Act
        fxt_benchmarker_task_chain.set_task_chain_models(models_task_1, models_task_2)

        # Assert
        assert fxt_benchmarker_task_chain._are_models_specified
        assert fxt_benchmarker_task_chain._task_chain_models == [
            [models_task_1[0], models_task_2[0]],
            [models_task_1[0], models_task_2[1]],
            [models_task_1[1], models_task_2[0]],
            [models_task_1[1], models_task_2[1]],
        ]
        assert fxt_benchmarker_task_chain._task_chain_algorithms == [
            [models_task_1[0].architecture, models_task_2[0].architecture],
            [models_task_1[0].architecture, models_task_2[1].architecture],
            [models_task_1[1].architecture, models_task_2[0].architecture],
            [models_task_1[1].architecture, models_task_2[1].architecture],
        ]

    def test_set_task_chain_algorithms(self, fxt_benchmarker_task_chain: Benchmarker, mocker: MockerFixture):
        # Arrange
        algorithms_task_1 = ("ALGO_1", "ALGO_2")
        algorithms_task_2 = ("ALGO_3", "ALGO_4")

        # Act
        fxt_benchmarker_task_chain.set_task_chain_algorithms(algorithms_task_1, algorithms_task_2)

        # Assert
        assert not fxt_benchmarker_task_chain._are_models_specified
        assert fxt_benchmarker_task_chain._task_chain_models is None
        assert fxt_benchmarker_task_chain._task_chain_algorithms == [
            ["ALGO_1", "ALGO_3"],
            ["ALGO_1", "ALGO_4"],
            ["ALGO_2", "ALGO_3"],
            ["ALGO_2", "ALGO_4"],
        ]

    def test_properties(self, fxt_benchmarker_task_chain: Benchmarker):
        # Assert
        assert fxt_benchmarker_task_chain.models == fxt_benchmarker_task_chain._task_chain_models
        assert fxt_benchmarker_task_chain.algorithms == fxt_benchmarker_task_chain._task_chain_algorithms
        with pytest.raises(ValueError):
            # benchmarker has not been initialized
            fxt_benchmarker_task_chain.optimized_models

    def test__train_model_for_algorithm(self, fxt_benchmarker_task_chain: Benchmarker):
        # Arrange
        task_index = 0
        algorithm_name = "ALGO_1"

        # Act
        model = fxt_benchmarker_task_chain._train_model_for_algorithm(task_index, algorithm_name)

        # Assert
        fxt_benchmarker_task_chain.training_client.train_task.assert_called_once()
        assert model is fxt_benchmarker_task_chain.model_client.get_model_for_job()

    def test__optimize_model_for_algorithm(self, fxt_benchmarker_task_chain: Benchmarker, mocker: MockerFixture):
        # Arrange
        model = mocker.MagicMock()
        precision = "INT8"

        # Act
        optimized_model = fxt_benchmarker_task_chain._optimize_model_for_algorithm(model, precision)

        # Assert
        fxt_benchmarker_task_chain.model_client.optimize_model.assert_called_once_with(model=model)
        assert optimized_model is fxt_benchmarker_task_chain.model_client.update_model_detail().get_optimized_model()

    def test_prepare_benchmark(
        self,
        fxt_benchmarker: Benchmarker,
        mocker: MockerFixture,
        fxt_temp_directory: str,
    ):
        # Arrange
        mock_deploy_project = mocker.patch.object(fxt_benchmarker.geti, "deploy_project")

        # Act
        fxt_benchmarker.prepare_benchmark(fxt_temp_directory)

        # Assert
        assert mock_deploy_project.call_count == len(fxt_benchmarker.models) * len(fxt_benchmarker.precision_levels)
        assert fxt_benchmarker._optimized_models is not None
        assert fxt_benchmarker._deployment_folders == [
            str(Path(fxt_temp_directory) / f"deployment_{i}") for i in range(len(fxt_benchmarker._optimized_models))
        ]

    def test_initialize_from_folder(
        self,
        fxt_benchmarker: Benchmarker,
        mocker: MockerFixture,
        fxt_temp_directory: str,
    ):
        # Arrange
        deployment_path = Path(fxt_temp_directory) / "deployment_test"
        deployment_path.mkdir()
        deployment = mocker.MagicMock()
        deployment.project.name = fxt_benchmarker.project.name
        mocked_deployment_from_folder = mocker.patch(
            "geti_sdk.benchmarking.benchmarker.Deployment.from_folder",
            return_value=deployment,
        )

        # Act
        fxt_benchmarker.initialize_from_folder(fxt_temp_directory)

        # Assert
        mocked_deployment_from_folder.assert_called_once_with(path_to_folder=str(deployment_path))
        assert len(fxt_benchmarker._deployment_folders) == 1
        assert fxt_benchmarker._deployment_folders[0] == str(deployment_path)

    def test_throughput_benchmark(
        self,
        fxt_benchmarker: Benchmarker,
        mocker: MockerFixture,
        fxt_temp_directory: str,
    ):
        # Arrange
        _ = mocker.patch.object(fxt_benchmarker.geti, "deploy_project")
        loaded_images = [mocker.MagicMock(), mocker.MagicMock()]
        mock_load_benchmark_media = mocker.patch(
            "geti_sdk.benchmarking.benchmarker.load_benchmark_media",
            return_value=loaded_images,
        )
        fxt_benchmarker.prepare_benchmark(fxt_temp_directory)
        deployment = mocker.MagicMock()
        deployment.project.name = fxt_benchmarker.project.name
        deployment.models = [
            mocker.MagicMock(),
        ]
        _ = mocker.patch(
            "geti_sdk.benchmarking.benchmarker.Deployment.from_folder",
            return_value=deployment,
        )

        number_of_runs = len(fxt_benchmarker.models) * len(fxt_benchmarker.precision_levels)
        frames, repeats = len(loaded_images), 2
        results_file = Path(fxt_temp_directory) / "results.csv"

        # Act
        fxt_benchmarker.run_throughput_benchmark(
            working_directory=fxt_temp_directory,
            results_filename=results_file.stem,
            frames=frames,
            repeats=repeats,
        )

        # Assert
        mock_load_benchmark_media.assert_called_once()
        assert results_file.is_file()
        assert len(list(csv.DictReader(results_file.open()))) == number_of_runs
        assert deployment.load_inference_models.call_count == number_of_runs
        # For each model infer is called: 1 Warm-up call, 1 time estimation call and `frames * repeats` for benchmark
        assert deployment.infer.call_count == number_of_runs * (2 + frames * repeats)

    def test_compare_predictions(
        self,
        fxt_benchmarker: Benchmarker,
        mocker: MockerFixture,
        fxt_temp_directory: str,
    ):
        # Arrange
        mock_image = np.array((10, 10, 3), dtype=np.uint8)
        mocked_prediction_client = mocker.patch(
            "geti_sdk.benchmarking.benchmarker.PredictionClient",
        )
        _ = mocker.patch.object(fxt_benchmarker.geti, "deploy_project")
        _ = mocker.patch(
            "geti_sdk.benchmarking.benchmarker.cv2.getTextSize",
            return_value=((10, 10), 10),
        )
        _ = mocker.patch(
            "geti_sdk.benchmarking.benchmarker.cv2.copyMakeBorder",
            return_value=mock_image,
        )
        _ = mocker.patch(
            "geti_sdk.benchmarking.benchmarker.cv2.putText",
        )
        mock_visualizer_draw = mocker.patch(
            "geti_sdk.benchmarking.benchmarker.Visualizer.draw",
            return_value=mock_image,
        )
        mock_pad_image_and_put_caption = mocker.patch(
            "geti_sdk.benchmarking.benchmarker.pad_image_and_put_caption",
        )
        mock_concat_prediction_results = mocker.patch(
            "geti_sdk.benchmarking.benchmarker.concat_prediction_results",
        )
        fxt_benchmarker.prepare_benchmark(fxt_temp_directory)
        deployment = mocker.MagicMock()
        deployment.project.name = fxt_benchmarker.project.name
        deployment.models = [
            mocker.MagicMock(),
        ]
        _ = mocker.patch(
            "geti_sdk.benchmarking.benchmarker.Deployment.from_folder",
            return_value=deployment,
        )

        results_file = Path(fxt_temp_directory) / "comparison.jpg"

        # Act
        fxt_benchmarker.compare_predictions(
            working_directory=fxt_temp_directory,
            image=mock_image,
            saved_image_name=results_file.stem,
        )

        # Assert
        assert results_file.is_file()
        mocked_prediction_client.return_value.predict_image.assert_called_once_with(mock_image)
        assert (
            mock_visualizer_draw.call_count
            == mock_pad_image_and_put_caption.call_count
            == (
                # Calls for deployments + online prediction call
                len(fxt_benchmarker.models) * len(fxt_benchmarker.precision_levels) + 1
            )
        )
        mock_concat_prediction_results.assert_called_once()
