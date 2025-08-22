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


from geti_sdk.data_models.containers import AlgorithmList
from geti_sdk.data_models.enums import TaskType
from geti_sdk.data_models.project import Project
from geti_sdk.http_session import GetiSession


def get_supported_algorithms(
    rest_session: GetiSession,
    task_type: TaskType | None = None,
    project: Project | None = None,
    workspace_id: str | None = None,
) -> AlgorithmList:
    """
    Return the list of supported algorithms (including algorithm metadata) for the
    cluster.

    :param rest_session: HTTP session to the cluster
    :param task_type: Optional TaskType for which to get the supported algorithms.
    :param project: Project to get the supported algorithms for. NOTE: `project` is
        not required for Geti versions v1.8 and lower, but is mandatory for v1.9 and up.
    :param workspace_id: ID of the workspace in which the project to get the supported
        algorithms for lives. NOTE: This is not required for Geti versions v1.8 and
        lower, but is mandatory for v1.9 and up.
    :return: AlgorithmList holding the supported algorithms
    """
    if (workspace_id is None) or (project is None):
        raise ValueError(
            "For Geti v1.9 or higher, passing `workspace_id` and `project` is "
            "mandatory in order to retrieve the supported algorithms"
        )
    url = f"workspaces/{workspace_id}/projects/{project.id}/supported_algorithms"

    algorithm_rest_response = rest_session.get_rest_response(url=url, method="GET")

    if task_type:
        filtered_response = [
            algo
            for algo in algorithm_rest_response["supported_algorithms"]
            if algo["task_type"].upper() == task_type.name
        ]
        algorithm_rest_response["items"] = filtered_response
    return AlgorithmList.from_rest(algorithm_rest_response, geti_version=rest_session.version)


def get_default_algorithm_info(
    session: GetiSession,
    workspace_id: str,
    project: Project,
) -> dict[TaskType, str]:
    """
    Return the names of the default algorithms for the tasks in the `project`. The
    returned response is a map of TaskType to the default algorithm name for that task

    :param session: GetiSession to the Intel Geti server
    :param workspace_id: Workspace ID in which the project to retrieve the default
        algorithms for lives.
    :param project: Project to retrieve the default algorithms for
    :return: Dictionary mapping the default algorithm name to the TaskType, for each
        task in the project
    """
    if (workspace_id is None) or (project is None):
        raise ValueError(
            "For Geti v1.9 or higher, passing `workspace_id` and `project` is "
            "mandatory in order to retrieve the supported algorithms"
        )
    url = f"workspaces/{workspace_id}/projects/{project.id}/supported_algorithms"

    algorithm_rest_response = session.get_rest_response(url=url, method="GET")
    defaults = algorithm_rest_response.get("default_algorithms", None)
    if defaults is None:
        raise ValueError(
            "The `supported_algorithms` did not return a response to obtain the "
            "default algorithms. Most likely the Geti server you are using does not "
            "support this functionality yet."
        )
    task_type_names = [task.type.value.lower() for task in project.get_trainable_tasks()]
    result: dict[TaskType, str] = {}
    for entry in defaults:
        task_type = entry["task_type"]
        if task_type in task_type_names:
            result.update({TaskType(task_type): entry["model_template_id"]})
    return result
