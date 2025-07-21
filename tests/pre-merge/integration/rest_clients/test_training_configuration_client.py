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

from typing import List

import pytest

from tests.helpers.constants import PROJECT_PREFIX
from tests.helpers.project_service import ProjectService


class TestConfigurationClient:
    @pytest.mark.vcr()
    def test_get_and_set_configuration(
        self, fxt_project_service: ProjectService, fxt_default_labels: List[str]
    ):
        """
        Verifies that getting and setting the training configuration for a single task project
        works as expected

        Steps:
        1. Create detection project
        2. Get training configuration
        3. Update the configuration
        4. PATCH the new configuration to the server
        5. GET the training configuration again, and assert that parameters have changed
        """
        project = fxt_project_service.create_project(
            project_name=f"{PROJECT_PREFIX}_training_configuration_client",
            project_type="detection",
            labels=[fxt_default_labels],
        )
        task = project.get_trainable_tasks()[0]
        model_client = fxt_project_service.model_client
        model_manifest = model_client.supported_algos.get_default_for_task_type(task_type=task.type)

        training_configuration_client = fxt_project_service.training_configuration_client
        config = training_configuration_client.get_configuration(
            model_manifest_id=model_manifest.model_manifest_id
        )
        assert config.hyperparameters.training.learning_rate > 0.0

        new_lr = config.hyperparameters.training.learning_rate + 0.001
        new_max_epochs = config.hyperparameters.training.max_epochs + 1
        config.hyperparameters.training.learning_rate = new_lr
        config.hyperparameters.training.max_epochs = new_max_epochs
        training_configuration_client.set_configuration(config)

        config = training_configuration_client.get_configuration(
            model_manifest_id=model_manifest.model_manifest_id
        )

        assert config.hyperparameters.training.learning_rate == new_lr
        assert config.hyperparameters.training.max_epochs == new_max_epochs
