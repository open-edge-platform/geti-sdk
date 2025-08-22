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

import pandas as pd
from test_nightly_project import TestNightlyProject

from geti_sdk.benchmarking.benchmarker import Benchmarker
from geti_sdk.geti import Geti
from tests.helpers import project_service


class TestDetectionToClassification(TestNightlyProject):
    """
    Class to test project creation, annotation upload, training, prediction and
    deployment for a detection_to_classification project
    """

    PROJECT_TYPE = "detection_to_classification"
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
        images = [fxt_image_path, fxt_image_path_complex]
        precision_levels = ["FP16", "INT8"]

        benchmarker = Benchmarker(
            geti=fxt_geti_no_vcr,
            project=project,
            precision_levels=precision_levels,
            benchmark_images=images,
        )
        benchmarker.set_task_chain_algorithms(
            [algo.name for algo in fxt_project_service_no_vcr._training_client.get_algorithms_for_task(0)][:2],
            [algo.name for algo in fxt_project_service_no_vcr._training_client.get_algorithms_for_task(1)][:2],
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
