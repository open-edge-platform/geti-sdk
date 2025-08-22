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
import time
import warnings

from tqdm import TqdmWarning
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from geti_sdk.data_models.enums.job_state import JobState
from geti_sdk.data_models.job import Job
from geti_sdk.http_session import GetiRequestException, GetiSession
from geti_sdk.rest_converters.job_rest_converter import JobRESTConverter


def restrict(value: float, min: float = 0, max: float = 1) -> float:  # noqa: A002
    """
    Restrict the input `value` to a certain range such that `min` < `value` < `max`

    :param value: Variable to restrict
    :param min: Minimum allowed value. Defaults to zero
    :param max: Maximally allowed value. Defaults to one
    :return: The value constrained to the specified range
    """
    if value < min:
        return min
    if value > max:
        return max
    return value


def get_job_by_id(job_id: str, session: GetiSession, workspace_id: str) -> Job | None:
    """
    Retrieve Job information from the Intel® Geti™ server.

    :param job_id: Unique ID of the job to retrieve
    :param session: GetiSession instance addressing the Intel® Geti™ platform
    :param workspace_id: ID of the workspace in which the job was created
    :return: Job instance holding the details of the job
    """
    try:
        response = session.get_rest_response(url=f"workspaces/{workspace_id}/jobs/{job_id}", method="GET")
    except GetiRequestException as error:
        if error.status_code == 404:
            return None
        raise error
    job = JobRESTConverter.from_dict(response)
    job.workspace_id = workspace_id
    job.geti_version = session.version
    return job


def get_job_with_timeout(
    job_id: str,
    session: GetiSession,
    workspace_id: str,
    job_type: str = "training",
    timeout: int = 15,
) -> Job:
    """
    Retrieve a Job from the Intel® Geti™ server, by it's unique ID. If the job is not
    found within the specified `timeout`, a RuntimeError is raised.

    :param job_id: Unique ID of the job to retrieve
    :param session: GetiSession instance addressing the Intel® Geti™ platform
    :param workspace_id: ID of the workspace in which the job was created
    :param job_type: String representing the type of job, for instance "training" or
        "testing"
    :param timeout: Time (in seconds) after which the job retrieval will timeout
    :raises: RuntimeError if the job is not found within the specified timeout
    :return: Job instance holding the details of the job
    """
    try:
        job = get_job_by_id(job_id=job_id, session=session, workspace_id=workspace_id)
    except GetiRequestException as job_error:
        if job_error.status_code == 403:
            job = None
        else:
            raise job_error
    if job is not None:
        logging.debug(f"{job_type.capitalize()} job with ID {job_id} retrieved from the platform.")
    else:
        t_start = time.time()
        while job is None and (time.time() - t_start < timeout):
            logging.debug(
                f"{job_type.capitalize()} job status could not be retrieved from the "
                f"platform yet. Re-attempting to fetch job status. Looking for job "
                f"with ID {job_id}"
            )
            time.sleep(2)
            try:
                job = get_job_by_id(session=session, job_id=job_id, workspace_id=workspace_id)
            except GetiRequestException as job_error:
                if job_error.status_code == 403:
                    job = None
                else:
                    raise job_error
        if job is None:
            raise RuntimeError(f"Unable to find the resulting {job_type} job on the Intel® Geti™ server.")
    return job


def monitor_jobs(session: GetiSession, jobs: list[Job], timeout: int = 10000, interval: int = 15) -> list[Job]:
    """
    Monitor and print the progress of all jobs in the list `jobs`. Execution is
    halted until all jobs have either finished, failed or were cancelled.

    Progress will be reported in 15s intervals

    :param session: GetiSession instance addressing the Intel® Geti™ platform
    :param jobs: List of jobs to monitor
    :param timeout: Timeout (in seconds) after which to stop the monitoring
    :param interval: Time interval (in seconds) at which the TrainingClient polls
        the server to update the status of the jobs. Defaults to 15 seconds
    :return: List of finished (or failed) jobs with their status updated
    """
    monitoring = True
    completed_states = [
        JobState.FINISHED,
        JobState.CANCELLED,
        JobState.FAILED,
        JobState.ERROR,
    ]
    jobs_to_monitor = [job for job in jobs if job.state not in completed_states]
    logging.info(f"Monitoring progress for {len(jobs_to_monitor)} jobs...")
    outer_bars = []
    inner_bars = []
    descriptions = []
    progress_values = []
    job_steps = []
    total_job_steps = []
    finished_jobs: list[Job] = []
    jobs_with_error: list[Job] = []
    with warnings.catch_warnings(), logging_redirect_tqdm(tqdm_class=tqdm):
        warnings.filterwarnings("ignore", category=TqdmWarning)
        for index, job in enumerate(jobs_to_monitor):
            inner_description = job.current_step_message
            outer_description = f"Project `{job.metadata.project.name}` - {job.name}"
            outer_bars.append(
                tqdm(
                    total=job.total_steps,
                    desc=outer_description,
                    position=2 * index,
                    unit="step",
                    initial=job.current_step,
                    leave=True,
                    bar_format="{desc}: Step {n_fmt}/{total_fmt} |{bar}| [Total time elapsed: {elapsed}]",
                    miniters=0,
                )
            )
            inner_bar_format_string = (
                "{desc:>"
                + str(len(outer_description) + 2)
                + "}"
                + "{percentage:7.0f}% |{bar}| [{elapsed}<{remaining}, {rate_fmt}]"
            )
            inner_bar = tqdm(
                total=100,
                unit="%",
                bar_format=inner_bar_format_string,
                position=2 * index + 1,
                leave=True,
            )
            inner_bar.set_description(inner_description)
            inner_bars.append(inner_bar)
            descriptions.append(inner_description)
            progress_values.append(0)
            job_steps.append(job.current_step)
            total_job_steps.append(job.total_steps)
        try:
            t_start = time.time()
            t_elapsed = 0
            complete_count = 0
            while monitoring and t_elapsed < timeout:
                for index, job in enumerate(jobs_to_monitor):
                    if job in finished_jobs or job in jobs_with_error:
                        # Job has completed some time ago, skip further updates
                        continue

                    try:
                        job.update(session)
                    except GetiRequestException as error:
                        if error.status_code == 404:
                            logging.warning(
                                f"Job with name `{job.name}` and id `{job.id}` was not "
                                f"found on the Intel Geti instance. Monitoring is skipped "
                                f"for this job."
                            )
                        jobs_with_error.append(job)
                        complete_count += 1
                    if job.state in completed_states:
                        # Job has just completed, update progress bars to final state
                        complete_count += 1
                        finished_jobs.append(job)
                        inner_bars[index].set_description(
                            "Job completed.",
                            refresh=True,
                        )
                        inner_bars[index].update(110)
                        outer_bars[index].update(total_job_steps[index] - job_steps[index])
                        continue

                    no_step_message = job.current_step_message
                    if no_step_message != descriptions[index]:
                        # Next phase of the job, reset progress bar
                        inner_bars[index].set_description(no_step_message, refresh=True)
                        inner_bars[index].reset(total=100)
                        descriptions[index] = no_step_message
                        progress_values[index] = 0
                        outer_bars[index].total = job.total_steps
                        outer_bars[index].update(job.current_step - job_steps[index])
                        job_steps[index] = job.current_step

                    incremental_progress = job.current_step_progress - progress_values[index]
                    restrict(incremental_progress, min=0, max=100)
                    inner_bars[index].update(incremental_progress)
                    progress_values[index] = job.current_step_progress
                    outer_bars[index].update(0)

                if complete_count == len(jobs_to_monitor):
                    break
                time.sleep(interval)
                t_elapsed = time.time() - t_start
        except KeyboardInterrupt:
            logging.info("Job monitoring interrupted, stopping...")
            for ib, ob in zip(inner_bars, outer_bars):
                ib.close()
                ob.close()
            return jobs
        if t_elapsed < timeout:
            logging.info("All jobs completed, monitoring stopped.")
        else:
            logging.info(f"Monitoring stopped after {t_elapsed:.1f} seconds due to timeout.")
        for ib, ob in zip(inner_bars, outer_bars):
            ib.close()
            ob.close()
    return jobs


def monitor_job(session: GetiSession, job: Job, timeout: int = 10000, interval: int = 15) -> Job:
    """
    Monitor and print the progress of a single `job`. Execution is
    halted until the job has either finished, failed or was cancelled.

    Progress will be reported in 15s intervals

    :param session: GetiSession instance addressing the Intel® Geti™ platform
    :param job: Job to monitor
    :param timeout: Timeout (in seconds) after which to stop the monitoring
    :param interval: Time interval (in seconds) at which the TrainingClient polls
        the server to update the status of the job. Defaults to 15 seconds
    :return: The finished (or failed) job with it's status updated
    """
    monitoring = True
    completed_states = [
        JobState.FINISHED,
        JobState.CANCELLED,
        JobState.FAILED,
        JobState.ERROR,
    ]
    try:
        logging.info(f"Monitoring `{job.name}` job for project `{job.metadata.project.name}`: {job.metadata.task.name}")
    except AttributeError:
        logging.info(f"Monitoring `{job.name}` with id {job.id}")
    try:
        job.update(session)
    except GetiRequestException as error:
        if error.status_code == 404:
            logging.warning(
                f"Job with name `{job.name}` and id `{job.id}` was not "
                f"found on the Intel Geti instance. Monitoring is skipped "
                f"for this job."
            )
    if job.state in completed_states:
        logging.info(f"Job `{job.name}` has already finished with status {str(job.state)}, monitoring stopped")
        return job

    t_start = time.time()
    t_elapsed = 0

    with logging_redirect_tqdm(tqdm_class=tqdm), warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=TqdmWarning)
        try:
            previous_progress = 0
            previous_message = job.current_step_message
            current_step = job.current_step
            outer_description = (
                f"Project `{job.metadata.project.name}` - " if job.metadata.project else ""
            ) + f"{job.name}"
            total_steps = job.total_steps
            outer_bar = tqdm(
                total=total_steps,
                desc=outer_description,
                position=0,
                unit="step",
                initial=current_step,
                leave=True,
                bar_format="{desc}: Step {n_fmt}/{total_fmt} |{bar}| [Total time elapsed: {elapsed}]",
                miniters=0,
            )
            inner_bar_format_string = (
                "{desc:>"
                + str(len(outer_description) + 2)
                + "}"
                + "{percentage:7.0f}% |{bar}| [{elapsed}<{remaining}, {rate_fmt}]"
            )
            inner_bar = tqdm(
                total=100,
                unit="%",
                bar_format=inner_bar_format_string,
                position=1,
                leave=False,
            )
            inner_bar.set_description(previous_message)
            while monitoring and t_elapsed < timeout:
                job.update(session)
                if job.total_steps > total_steps:
                    total_steps = job.total_steps
                    outer_bar.reset(total=job.total_steps)
                    outer_bar.update(job.current_step)
                    current_step = job.current_step
                if job.state in completed_states:
                    outer_bar.update(total_steps - current_step)
                    inner_bar.update(100 - previous_progress)
                    monitoring = False
                    break
                no_step_message = job.current_step_message
                if no_step_message != previous_message:
                    # Next phase of the job, reset progress bar
                    inner_bar.set_description(f"{no_step_message}", refresh=True)
                    inner_bar.reset(total=100)
                    previous_message = no_step_message
                    previous_progress = 0
                    outer_bar.update(job.current_step - current_step)
                    current_step = job.current_step

                incremental_progress = job.current_step_progress - previous_progress
                restrict(incremental_progress, min=0, max=100)
                inner_bar.update(incremental_progress)
                outer_bar.update(0)
                previous_progress = job.current_step_progress
                time.sleep(interval)
                t_elapsed = time.time() - t_start
            inner_bar.close()
            outer_bar.close()
        except KeyboardInterrupt:
            logging.info("Job monitoring interrupted, stopping...")
            job.update(session)
            inner_bar.close()
            outer_bar.close()
            return job
        if t_elapsed < timeout:
            logging.info(f"Job `{job.name}` finished, monitoring stopped. Total time elapsed: {t_elapsed:.1f} seconds")
        else:
            logging.info(
                f"Monitoring stopped after {t_elapsed:.1f} seconds due to timeout. Current job state: {job.state}"
            )
    return job
