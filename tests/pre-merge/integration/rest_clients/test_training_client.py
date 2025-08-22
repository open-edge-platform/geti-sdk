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

import pytest

from geti_sdk.data_models import Job, Project, ProjectStatus
from geti_sdk.data_models.enums import JobState
from tests.helpers import (
    DatumAnnotationReader,
    ProjectService,
    SdkTestMode,
    attempt_to_train_task,
    await_training_start,
    get_or_create_annotated_project_for_test_class,
)
from tests.helpers.constants import PROJECT_PREFIX


class TestTrainingClient:
    @staticmethod
    def ensure_annotated_project(project_service: ProjectService, annotation_reader: DatumAnnotationReader) -> Project:
        return get_or_create_annotated_project_for_test_class(
            project_service=project_service,
            annotation_readers=[annotation_reader],
            project_type="detection",
            project_name=f"{PROJECT_PREFIX}_training_client",
        )

    @pytest.mark.vcr()
    def test_train_task_and_get_jobs(
        self,
        fxt_project_service: ProjectService,
        fxt_annotation_reader: DatumAnnotationReader,
        fxt_test_mode: SdkTestMode,
    ) -> None:
        """
        Verifies that submitting a training job for a task in a project with
        sufficient annotations works

        """
        project = self.ensure_annotated_project(
            project_service=fxt_project_service, annotation_reader=fxt_annotation_reader
        )

        job = attempt_to_train_task(
            training_client=fxt_project_service.training_client,
            task=project.get_trainable_tasks()[0],
            test_mode=fxt_test_mode,
        )
        assert isinstance(job, Job)

        await_training_start(fxt_test_mode, fxt_project_service.training_client)

        jobs = fxt_project_service.training_client.get_jobs(project_only=True)
        assert job.id in [project_job.id for project_job in jobs]

        # Update job status
        job.update(fxt_project_service.session)
        if job.state in JobState.active_states():
            # Cancel the job
            logging.info(f"Job '{job.name}' is still active, cancelling...")
            job.cancel(fxt_project_service.session)
        else:
            logging.info(f"Job '{job.name}' has already excited with status {job.state}.")
        logging.info(job.state)

    @pytest.mark.vcr()
    def test_get_status(
        self,
        fxt_project_service: ProjectService,
        fxt_annotation_reader: DatumAnnotationReader,
    ) -> None:
        """
        Test that fetching project status works

        """
        self.ensure_annotated_project(project_service=fxt_project_service, annotation_reader=fxt_annotation_reader)
        status = fxt_project_service.training_client.get_status()
        assert isinstance(status, ProjectStatus)
