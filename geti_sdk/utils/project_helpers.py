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

from pathvalidate import sanitize_filename

from geti_sdk.data_models.enums import TaskType
from geti_sdk.data_models.project import Project


def get_task_types_by_project_type(project_type: str) -> list[TaskType]:
    """
    Return a list of task_type for each task in the project pipeline, for a certain
    'project_type'.

    :param project_type:
    :return:
    """
    return [TaskType(task) for task in project_type.split("_to_")]


def get_project_folder_name(project: Project) -> str:
    """
    Return a folder name for the project, that can be used to download the project to.

    :param project: Project to get a folder name for
    :return: string holding the folder name for the project
    """
    name_part = sanitize_filename(project.name)
    return f"{project.id}_{name_part}"
