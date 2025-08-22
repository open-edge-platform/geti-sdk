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
from typing import Any

from geti_sdk.data_models import (
    Algorithm,
    Dataset,
    Job,
    Project,
    ProjectStatus,
    Task,
    TaskConfiguration,
)
from geti_sdk.data_models.containers import AlgorithmList
from geti_sdk.data_models.enums import JobState, JobType
from geti_sdk.http_session import GetiSession
from geti_sdk.rest_converters import (
    ConfigurationRESTConverter,
    JobRESTConverter,
    StatusRESTConverter,
)
from geti_sdk.utils import get_supported_algorithms
from geti_sdk.utils.job_helpers import get_job_with_timeout, monitor_job, monitor_jobs


class TrainingClient:
    """
    Class to manage training jobs for a certain Intel® Geti™ project.
    """

    def __init__(self, workspace_id: str, project: Project, session: GetiSession):
        self.session = session
        self.project = project
        self.workspace_id = workspace_id
        self.base_url = f"workspaces/{workspace_id}/projects/{project.id}"
        self.supported_algos = get_supported_algorithms(
            rest_session=session, project=project, workspace_id=workspace_id
        )

    def get_status(self) -> ProjectStatus:
        """
        Get the current status of the project from the Intel® Geti™ server.

        :return: ProjectStatus object reflecting the current project status
        """
        response = self.session.get_rest_response(url=f"{self.base_url}/status", method="GET")
        return StatusRESTConverter.from_dict(response)

    def is_training(self) -> bool:
        """
        Request the project status and return True if the project is training
        """
        return self.get_status().is_training

    def get_jobs(self, project_only: bool = True, running_only: bool = False) -> list[Job]:
        """
        Return a list of all jobs on the Intel® Geti™ server.

        If `project_only = True` (the default), only those jobs related to the project
        managed by this TrainingClient will be returned. If set to False, all jobs in
        the workspace are returned.

        :param project_only: True to return only those jobs pertaining to the project
            for which the TrainingClient is active. False to return all jobs in the
            Intel® Geti™ workspace.
        :param running_only: If set to True, only return those jobs that are still
            running. Completed or Scheduled jobs will not be included in that case
        :return: List of Jobs
        """
        query = "?limit=100"
        if project_only:
            query += f"&project_id={self.project.id}"
        if running_only:
            query += "&state=running"
        response_list_key = "jobs"

        job_rest_list: list[dict] = []
        while response := self.session.get_rest_response(
            url=f"workspaces/{self.workspace_id}/jobs{query}&skip={len(job_rest_list)}",
            method="GET",
        ):
            job_rest_list.extend(response[response_list_key])
            job_count_dict = response["jobs_count"]
            total_job_count = (
                job_count_dict.get("n_running_jobs", 0)
                + job_count_dict.get("n_finished_jobs", 0)
                + job_count_dict.get("n_failed_jobs", 0)
                + job_count_dict.get("n_cancelled_jobs", 0)
            )
            if len(job_rest_list) >= total_job_count:
                break

        job_list: list[Job] = []
        for job_dict in job_rest_list:
            job = JobRESTConverter.from_dict(job_dict)
            if running_only and not job.is_running:
                continue
            job.workspace_id = self.workspace_id
            job.geti_version = self.session.version
            job_list.append(job)

        return job_list

    def get_algorithms_for_task(self, task: Task | int) -> AlgorithmList:
        """
        Return a list of supported algorithms for a specific task.

        The `task` parameter accepts both a Task object and an integer. If an int is
        passed, this will be considered the index of the task in the list of trainable
        tasks for the project which is managed by the TrainingClient.

        :param task: Task to get the supported algorithms for. If an integer is passed,
            this is considered the index of the task in the trainable task list of the
            project. So passing `task=0` will return the algorithms for the first
            trainable task, etc.
        :return: List of supported algorithms for the task
        """
        if isinstance(task, int):
            task = self.project.get_trainable_tasks()[task]
        return self.supported_algos.get_by_task_type(task.type)

    def train_task(
        self,
        task: Task | int,
        dataset: Dataset | None = None,
        algorithm: Algorithm | None = None,
        train_from_scratch: bool = False,
        hyper_parameters: TaskConfiguration | None = None,
        hpo_parameters: dict[str, Any] | None = None,
        await_running_jobs: bool = True,
        timeout: int = 3600,
    ) -> Job:
        """
        Start training of a specific task in the project.

        The `task` parameter accepts both a Task object and an integer. If an int is
        passed, this will be considered the index of the task in the list of trainable
        tasks for the project which is managed by the TrainingClient.

        :param task: Task or index of Task to train
        :param dataset: Optional Dataset to train on
        :param algorithm: Optional Algorithm to use in training. If left as None (the
            default), the default algorithm for the task will be used.
        :param train_from_scratch: True to train the model from scratch, False to
            continue training from an existing checkpoint (if any)
        :param hyper_parameters: Optional hyper parameters to use for training
        :param hpo_parameters: [Deprecated] Optional set of parameters to use for automatic hyper
            parameter optimization.
        :param await_running_jobs: True to wait for currently running jobs to
            complete. This will guarantee that the training request can be submitted
            successfully. Setting this to False will cause an error to be raised when
            a training request is submitted for a task for which a training job is
            already in progress.
        :param timeout: Timeout (in seconds) to wait for the Job to be created. If a
            training request is submitted successfully, a training job should be
            instantiated on the Geti server. If the Job does not appear on the server
            job list within the `timeout`, an error will be raised. This parameter only
            takes effect when `await_running_jobs` is set to True.
        :return: The training job that has been created
        """
        if isinstance(task, int):
            task = self.project.get_trainable_tasks()[task]
        if dataset is not None:
            logging.warning(
                "Training on a Dataset other than the default training dataset is not "
                "supported in the version of the Geti platform running on your server. "
                "The `dataset` parameter will be disregarded."
            )
        if dataset is None:
            dataset = self.project.training_dataset
        if algorithm is None:
            algorithm = self.supported_algos.get_default_for_task_type(task.type)
        request_data: dict[str, Any] = {
            "dataset_id": dataset.id,
            "task_id": task.id,
            "train_from_scratch": train_from_scratch,
            "model_template_id": algorithm.model_manifest_id,
        }
        if hyper_parameters is not None:
            hypers = hyper_parameters.model_configurations
            hypers_rest = ConfigurationRESTConverter.configurable_parameter_list_to_rest(hypers)
            request_data.update({"hyper_parameters": hypers_rest})
        if hpo_parameters is not None:
            logging.warning("`hpo_parameters` have been deprecated and will be ignored.")

        request_data.pop("dataset_id")
        data = request_data

        task_jobs = self.get_jobs_for_task(task=task)
        task_training_jobs = [job for job in task_jobs if job.type == JobType.TRAIN]
        log_warning_msg = ""
        if len(task_training_jobs) >= 1:
            if len(task_training_jobs) == 1:
                msg_start = f"A training job for task '{task.title}' is"
            else:
                msg_start = f"Multiple training jobs for task '{task.title}' are"
            log_warning_msg = msg_start + " already in progress on the server."

        if len(task_training_jobs) >= 1:
            if not await_running_jobs:
                raise RuntimeError(
                    log_warning_msg + " Unable to submit training request. Please "
                    "wait for the current training job to finish or "
                    "set the `await_running_jobs` parameter in this "
                    "method to `True`."
                )
            logging.info(
                log_warning_msg + f" Awaiting completion of currently running jobs "
                f"before a new train request can be submitted. "
                f"Maximum waiting time set to {timeout} seconds."
            )
            self.monitor_jobs(jobs=task_training_jobs, timeout=timeout)

        response = self.session.get_rest_response(url=f"{self.base_url}:train", method="POST", data=data)

        job_id = response["job_id"]

        try:
            job = get_job_with_timeout(
                job_id=job_id,
                session=self.session,
                workspace_id=self.workspace_id,
                job_type="training",
            )
        except RuntimeError:
            raise RuntimeError(
                "Training job was submitted, but the TrainingClient was unable to "
                "find the resulting job on the platform."
            )
        logging.info(f"Training job with id {job.id} submitted successfully.")
        return job

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
        Monitor and print the progress of a `job`. Program execution is
        halted until the job has either finished, failed or was cancelled.

        Progress will be reported in 15s intervals

        :param job: job to monitor
        :param timeout: Timeout (in seconds) after which to stop the monitoring
        :param interval: Time interval (in seconds) at which the TrainingClient polls
            the server to update the status of the jobs. Defaults to 15 seconds
        :return: job with it's status updated
        """
        return monitor_job(session=self.session, job=job, timeout=timeout, interval=interval)

    def get_jobs_for_task(self, task: Task, running_only: bool = True) -> list[Job]:
        """
        Return a list of current jobs for the task, if any

        :param task: Task to retrieve the jobs for
        :param running_only: True to return only jobs that are currently running,
            False to return all jobs (including cancelled, finished or errored jobs)
        :return: List of Jobs running on the server for this particular task
        """
        project_jobs = self.get_jobs(project_only=True)
        task_jobs: list[Job] = []
        for job in project_jobs:
            if job.metadata is not None and job.metadata.task is not None and job.metadata.task.task_id == task.id:
                if running_only:
                    if job.state == JobState.RUNNING:
                        task_jobs.append(job)
                else:
                    task_jobs.append(job)
        return task_jobs
