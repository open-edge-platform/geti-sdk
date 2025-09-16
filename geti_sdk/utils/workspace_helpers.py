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

from geti_sdk.http_session import GetiSession


class MultipleWorkspacesException(Exception):
    """Exception raised when multiple workspaces are available thus it is not possible to automatically select one."""

    def __init__(self, workspaces_list: list) -> None:
        self.available_workspaces = workspaces_list
        ws_ids_and_names = [(ws["id"], ws["name"]) for ws in workspaces_list]
        error_message = (
            f"Multiple workspaces are available; please select one and provide its id through the 'workspace_id' "
            f"parameter when instantiating the client. Available workspaces (id, name): {ws_ids_and_names}."
        )
        super().__init__(error_message)


def get_workspace_id(rest_session: GetiSession) -> str:
    """
    Get the id of the workspace that is accessible with the provided session (cluster, organization, credentials, ...).

    In case of multiple workspaces, an error will be raised.

    :param rest_session: Session object which stores info about the connected cluster, organization and authentication
    :return: id of the workspace as a string
    :raises
    """
    workspaces = rest_session.get_rest_response(url="workspaces", method="GET")
    if isinstance(workspaces, list):
        workspace_list = workspaces
    elif isinstance(workspaces, dict):
        workspace_list = workspaces["workspaces"]
    else:
        raise ValueError(
            f"Unexpected response from cluster: {workspaces}. Expected to receive a "
            f"dictionary containing workspace data."
        )

    if len(workspace_list) > 1:
        raise MultipleWorkspacesException(workspace_list)

    return workspace_list[0]["id"]
