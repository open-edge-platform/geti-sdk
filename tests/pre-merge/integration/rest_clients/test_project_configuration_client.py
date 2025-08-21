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


import pytest

from tests.helpers.constants import PROJECT_PREFIX
from tests.helpers.project_service import ProjectService


class TestConfigurationClient:
    @pytest.mark.vcr()
    def test_get_and_set_project_configuration(
        self, fxt_project_service: ProjectService, fxt_default_labels: list[str]
    ):
        """
        Verifies that getting and setting the project configuration for a single task project
        works as expected

        Steps:
        1. Create detection project
        2. Get project configuration
        3. Switch auto training
        4. PATCH the new configuration to the server
        5. GET the project configuration again, and assert that auto-training has changed
        """
        fxt_project_service.create_project(
            project_name=f"{PROJECT_PREFIX}_project_configuration_client",
            project_type="detection",
            labels=[fxt_default_labels],
        )
        project_configuration_client = fxt_project_service.project_configuration_client
        config = project_configuration_client.get_configuration()

        old_auto_training = config.task_configs[0].auto_training.enable
        new_auto_training = not old_auto_training
        config.task_configs[0].auto_training.enable = new_auto_training

        project_configuration_client.set_configuration(config)

        config = project_configuration_client.get_configuration()

        assert config.task_configs[0].auto_training.enable == new_auto_training
