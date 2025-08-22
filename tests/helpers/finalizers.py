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
import time

from geti_sdk.data_models.project import Project
from geti_sdk.rest_clients import ProjectClient, TrainingClient


def force_delete_project(project: Project, project_client: ProjectClient) -> None:
    """
    Deletes the project named 'project_name'. If any jobs are running for the
    project, this finalizer cancels them.

    :param project_name: Name of project to delete
    :param project_client: ProjectClient to use for project deletion
    :param project_id: Optional ID of the project to delete. This can be useful in case
        there are multiple projects with the same name in the workspace
    """
    try:
        project_client.delete_project(project=project, requires_confirmation=False)
    except TypeError:
        logging.warning(f"Project {project.name} was not found on the server, it was most likely already deleted.")
    except ValueError:
        logging.error(
            f"Unable to delete project '{project.name}' from the server, it "
            f"is most likely locked for deletion due to an operation/training "
            f"session that is in progress. "
            f"\n\n Attempting to cancel the job and re-try project deletion."
        )
        training_client = TrainingClient(
            workspace_id=project_client.workspace_id,
            session=project_client.session,
            project=project,
        )
        jobs = training_client.get_jobs(project_only=True)
        for job in jobs:
            job.cancel(project_client.session)
        time.sleep(1)
        try:
            project_client.delete_project(project=project, requires_confirmation=False)
        except ValueError as error:
            raise ValueError(f"Unable to force delete project {project.name}, due to: {error} ")
