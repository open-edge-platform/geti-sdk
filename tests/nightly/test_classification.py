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

import os

import pandas as pd
from test_nightly_project import TestNightlyProject

from geti_sdk.benchmarking.benchmarker import Benchmarker
from geti_sdk.geti import Geti
from tests.helpers import project_service


class TestClassification(TestNightlyProject):
    """
    Class to test project creation, annotation upload, training, prediction, benchmarking and
    deployment for a classification project
    """

    PROJECT_TYPE = "classification"
    __test__ = True

    def test_benchmarking(
        self,
        fxt_project_service_no_vcr: project_service,
        fxt_geti_no_vcr: Geti,
        fxt_temp_directory: str,
        fxt_image_path: str,
        fxt_image_path_complex: str,
    ):
        """
        Tests benchmarking for the project.
        """
        project = fxt_project_service_no_vcr.project
        algorithms_to_benchmark = [
            algo.name for algo in fxt_project_service_no_vcr._training_client.get_algorithms_for_task(0)
        ][:2]
        images = [fxt_image_path, fxt_image_path_complex]
        precision_levels = ["FP16", "INT8"]

        benchmarker = Benchmarker(
            geti=fxt_geti_no_vcr,
            project=project,
            algorithms=algorithms_to_benchmark,
            precision_levels=precision_levels,
            benchmark_images=images,
        )
        benchmarker.prepare_benchmark(working_directory=fxt_temp_directory)
        results = benchmarker.run_throughput_benchmark(
            working_directory=fxt_temp_directory,
            results_filename="results",
            target_device="CPU",
            frames=2,
            repeats=2,
        )
        pd.DataFrame(results)
        benchmarker.compare_predictions(working_directory=fxt_temp_directory, throughput_benchmark_results=results)

    def test_export_import_project(
        self,
        fxt_project_service_no_vcr: project_service,
        fxt_geti_no_vcr: Geti,
        fxt_temp_directory: str,
    ):
        """
        Tests export import loop for the project.
        """
        project = fxt_project_service_no_vcr.project
        target_folder = os.path.join(fxt_temp_directory, project.name + "_snapshot")
        archive_path = target_folder + "/project_archive.zip"
        imported_project_name = "IMPORTED_PROJECT"

        # Project is exported
        assert not os.path.exists(archive_path)
        fxt_geti_no_vcr.export_project(
            filepath=archive_path,
            project=project,
        )
        assert os.path.exists(archive_path)

        # Project is imported
        existing_projects_pre_import = fxt_geti_no_vcr.project_client.get_all_projects(get_project_details=False)
        # Import
        imported_project = fxt_geti_no_vcr.import_project(filepath=archive_path, project_name=imported_project_name)
        assert imported_project.name == imported_project_name
        existing_projects = fxt_geti_no_vcr.project_client.get_all_projects(get_project_details=False)
        # Assert the imported project is NOT in the project list before the import
        assert (
            next(
                (p for p in existing_projects_pre_import if p.id == imported_project.id),
                None,
            )
            is None
        )
        # Assert the imported project is in the project list after the import
        assert next((p for p in existing_projects if p.id == imported_project.id), None) is not None
        # Project is deleted
        fxt_geti_no_vcr.project_client.delete_project(imported_project, requires_confirmation=False)

    def test_export_import_dataset(
        self,
        fxt_project_service_no_vcr: project_service,
        fxt_geti_no_vcr: Geti,
        fxt_temp_directory: str,
    ):
        """
        Tests export import loop for a dataset.
        """
        project = fxt_project_service_no_vcr.project
        assert project.datasets
        dataset = project.datasets[0]
        target_folder = os.path.join(fxt_temp_directory, project.name + "_snapshot")
        archive_path = target_folder + "/dataset_archive.zip"
        imported_project_name = "IMPORTED_PROJECT_FROM_DATASET"

        # Dataset is exported
        assert not os.path.exists(archive_path)
        fxt_geti_no_vcr.export_dataset(project=project, dataset=dataset, filepath=archive_path)
        assert os.path.exists(archive_path)
        # Dataset is imported as a project
        existing_projects_pre_import = fxt_geti_no_vcr.project_client.get_all_projects(get_project_details=False)
        # Import
        imported_project = fxt_geti_no_vcr.import_dataset(
            filepath=archive_path,
            project_name=imported_project_name,
            project_type=project.project_type,
        )
        assert imported_project.name == imported_project_name
        existing_projects = fxt_geti_no_vcr.project_client.get_all_projects(get_project_details=False)
        # Assert the imported project is NOT in the project list before the import
        assert (
            next(
                (p for p in existing_projects_pre_import if p.id == imported_project.id),
                None,
            )
            is None
        )
        # Assert the imported project is in the project list after the import
        assert next((p for p in existing_projects if p.id == imported_project.id), None) is not None
        # Project is deleted
        fxt_geti_no_vcr.project_client.delete_project(imported_project, requires_confirmation=False)
