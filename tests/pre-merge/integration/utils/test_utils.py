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
import os

import pytest

from geti_sdk.data_models import Algorithm, TaskType
from geti_sdk.http_session import GetiSession
from geti_sdk.utils import get_server_details_from_env, get_supported_algorithms
from tests.helpers import ProjectService
from tests.helpers.constants import (
    DUMMY_HOST,
    DUMMY_PASSWORD,
    DUMMY_TOKEN,
    DUMMY_USER,
    PROJECT_PREFIX,
)


class TestUtils:
    @pytest.mark.vcr()
    def test_get_supported_algorithms(
        self,
        fxt_geti_session: GetiSession,
        fxt_project_service: ProjectService,
    ):
        """
        Verifies that getting the list of supported algorithms from the server works
        as expected

        Test steps:
        1. Retrieve a list of supported algorithms from the server
        2. Assert that the returned list is not emtpy
        3. Assert that each entry in the list is a properly deserialized Algorithm
            instance
        4. Filter the AlgorithmList to select only the classification algorithms from
            it
        5. Assert that the list of classification algorithms is not empty and that
            each algorithm in it has the proper task type
        """
        project = fxt_project_service.create_project(
            project_name=f"{PROJECT_PREFIX}_test_supported_algos",
            project_type="detection_to_classification",
            labels=[["a"], ["b", "c"]],
        )
        algorithms = get_supported_algorithms(
            rest_session=fxt_geti_session,
            project=project,
            workspace_id=fxt_project_service.workspace_id,
        )

        assert len(algorithms) > 0
        for algorithm in algorithms:
            assert isinstance(algorithm, Algorithm)

        classification_algos = algorithms.get_by_task_type(task_type=TaskType.CLASSIFICATION)
        assert len(classification_algos) > 0
        for algorithm in classification_algos:
            assert algorithm.task == TaskType.CLASSIFICATION

    def test_get_server_details_from_env(self, fxt_env_filepath: str):
        """
        Verifies that fetching server details from a .env file works.

        This also tests that getting the server details from the global environment
        works as expected.
        """
        server_config = get_server_details_from_env(fxt_env_filepath)

        assert server_config.host == f"https://{DUMMY_HOST}"
        assert server_config.token == "this_is_a_fake_token"
        assert "https" in server_config.proxies
        assert not hasattr(server_config, "username")
        assert not hasattr(server_config, "password")

        environ_keys = ["GETI_HOST", "GETI_USERNAME", "GETI_PASSWORD", "GETI_TOKEN"]
        expected_results = {}
        dummy_results = {
            "GETI_HOST": DUMMY_HOST,
            "GETI_USERNAME": DUMMY_USER,
            "GETI_PASSWORD": DUMMY_PASSWORD,
            "GETI_TOKEN": DUMMY_TOKEN,
        }
        for ekey in environ_keys:
            evalue = os.environ.get(ekey, None)
            if evalue is not None:
                expected_results.update({ekey: evalue})
            else:
                variable_dictionary = {ekey: dummy_results[ekey]}
                os.environ.update(variable_dictionary)
                expected_results.update(variable_dictionary)

        server_config = get_server_details_from_env(use_global_variables=True)
        assert server_config.host.replace("https://", "") == expected_results["GETI_HOST"].replace("https://", "")
        assert server_config.token == expected_results["GETI_TOKEN"]
        assert server_config.proxies is None
