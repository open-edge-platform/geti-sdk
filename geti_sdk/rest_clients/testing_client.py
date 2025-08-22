# Copyright (C) 2023 Intel Corporation
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
from collections.abc import Sequence

from geti_sdk.data_models import Dataset, Job, Model, Project, TestResult
from geti_sdk.http_session import GetiSession
from geti_sdk.rest_converters import TestResultRESTConverter
from geti_sdk.utils.job_helpers import get_job_with_timeout, monitor_job, monitor_jobs

SUPPORTED_METRICS = ["global", "local"]


class TestingClient:
    """
    Class to manage testing jobs for a certain Intel® Geti™ project.
    """

    def __init__(self, workspace_id: str, project: Project, session: GetiSession):
        self.session = session
        self.project = project
        self.workspace_id = workspace_id
        self.base_url = f"workspaces/{workspace_id}/projects/{project.id}/tests"

    def test_model(
        self,
        model: Model,
        datasets: Sequence[Dataset],
        name: str | None = None,
        metric: str | None = None,
    ) -> Job:
        """
        Start a model testing job for a specific `model` and `dataset`

        :param model: Model to evaluate
        :param datasets: Testing dataset(s) to evaluate the model on
        :param name: Optional name to assign to the testing job
        :param metric: Optional metric to calculate. This is only valid for either
            anomaly segmentation or anomaly detection models. Possible values are
            `global` or `local`
        :return: Job object representing the testing job
        """
        if name is None:
            name = f"Test_{model.name}_{model.version}"
        dataset_ids = [ds.id for ds in datasets]

        test_data = {
            "name": name,
            "model_group_id": model.model_group_id,
            "model_id": model.id,
            "dataset_ids": dataset_ids,
        }
        if metric is not None:
            if metric not in SUPPORTED_METRICS:
                raise ValueError(f"Invalid metric received! Only `{SUPPORTED_METRICS}` are supported currently.")
            test_data.update({"metric": metric})

        response = self.session.get_rest_response(url=self.base_url, method="POST", data=test_data)
        job = get_job_with_timeout(
            job_id=response["job_id"],
            session=self.session,
            workspace_id=self.workspace_id,
            job_type="testing",
        )
        logging.info(f"Testing job with id {job.id} submitted successfully.")
        return job

    def get_test_result(self, job: Job) -> TestResult:
        """
        Retrieve the result of the model testing job from the Intel® Geti™
        server

        :param job: Job instance representing the model testing job
        :return: TestResult instance containing the test results
        """
        response = self.session.get_rest_response(url=self.base_url, method="GET")
        test_results = [TestResultRESTConverter.from_dict(result) for result in response["test_results"]]
        result_for_job = next((result for result in test_results if result.job_info.id == job.id), None)
        if result_for_job is None:
            raise ValueError(
                f"Unable to find test result for job `{job.name}`, please make sure "
                f"that a valid testing job was passed."
            )
        return result_for_job

    def monitor_jobs(self, jobs: list[Job], timeout: int = 10000, interval: int = 15) -> list[Job]:
        """
        Monitor and print the progress of all jobs in the list `jobs`. Execution is
        halted until all jobs have either finished, failed or were cancelled.

        Progress will be reported in 15s intervals

        :param jobs: List of jobs to monitor
        :param timeout: Timeout (in seconds) after which to stop the monitoring
        :param interval: Time interval (in seconds) at which the TrainingClient polls
            the server to update the status of the jobs. Defaults to 15 seconds
        :return: List of finished (or failed) jobs with their status updated
        """
        return monitor_jobs(session=self.session, jobs=jobs, timeout=timeout, interval=interval)

    def monitor_job(self, job: Job, timeout: int = 10000, interval: int = 15) -> Job:
        """
        Monitor and print the progress of the `job`. Execution is
        halted until the job has either finished, failed or was cancelled.

        Progress will be reported in 15s intervals

        :param job: Job to monitor
        :param timeout: Timeout (in seconds) after which to stop the monitoring
        :param interval: Time interval (in seconds) at which the TrainingClient polls
            the server to update the status of the jobs. Defaults to 15 seconds
        :return: Job with its status updated
        """
        return monitor_job(session=self.session, job=job, timeout=timeout, interval=interval)
