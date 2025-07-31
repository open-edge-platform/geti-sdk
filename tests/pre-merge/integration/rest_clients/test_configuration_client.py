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

from typing import List

import pytest

from tests.helpers.constants import PROJECT_PREFIX
from tests.helpers.project_service import ProjectService


class TestConfigurationClient:
    @pytest.mark.vcr()
    @pytest.mark.skip(
        reason="ConfigurationClient is deprecated. "
        "This test is disabled because VCR cassettes conflicts with the new configuration clients."
    )
    def test_get_and_set_configuration(
        self, fxt_project_service: ProjectService, fxt_default_labels: List[str]
    ):
        """
        Verifies that getting and setting the configuration for a single task project
        works as expected

        Steps:
        1. Create detection project
        2. Get task configuration
        3. Assert that 'learning_rate' is part of the task configuration
        4. Update the configuration so that the new batch size is half the old learning_rate
        5. POST the new configuration to the server
        6. GET the task configuration again, and assert that the learning_rate has changed
        """
        project = fxt_project_service.create_project(
            project_name=f"{PROJECT_PREFIX}_configuration_client",
            project_type="detection",
            labels=[fxt_default_labels],
        )
        task = project.get_trainable_tasks()[0]

        configuration_client = fxt_project_service.configuration_client
        task_configuration = configuration_client.get_task_configuration(task.id)
        assert "learning_rate" in task_configuration.get_all_parameter_names()
        old_learning_rate = task_configuration.learning_rate.value
        new_learning_rate = old_learning_rate / 2
        task_configuration.set_parameter_value("learning_rate", new_learning_rate)
        configuration_client.set_configuration(task_configuration)

        new_task_configuration = configuration_client.get_task_configuration(task.id)
        assert new_task_configuration.learning_rate.value == new_learning_rate
