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

import copy
import json
import logging
import os
import time
import warnings
from itertools import chain
from typing import Any

from geti_sdk.data_models import Project, Task, TaskType
from geti_sdk.data_models.utils import remove_null_fields
from geti_sdk.http_session import GetiRequestException, GetiSession
from geti_sdk.platform_versions import GETI_25_VERSION
from geti_sdk.rest_clients.dataset_client import DatasetClient
from geti_sdk.rest_converters import ProjectRESTConverter
from geti_sdk.utils.label_helpers import generate_unique_label_color
from geti_sdk.utils.project_helpers import get_task_types_by_project_type

from .task_templates import (
    ANOMALY_CLASSIFICATION_TASK,
    ANOMALY_DETECTION_TASK,
    ANOMALY_SEGMENTATION_TASK,
    ANOMALY_TASK,
    BASE_TEMPLATE,
    CLASSIFICATION_TASK,
    CROP_TASK,
    DETECTION_TASK,
    INSTANCE_SEGMENTATION_TASK,
    KEYPOINT_DETECTION_TASK,
    ROTATED_DETECTION_TASK,
    SEGMENTATION_TASK,
)

TASK_TYPE_MAPPING = {
    TaskType.CROP: CROP_TASK,
    TaskType.DETECTION: DETECTION_TASK,
    TaskType.SEGMENTATION: SEGMENTATION_TASK,
    TaskType.CLASSIFICATION: CLASSIFICATION_TASK,
    TaskType.ANOMALY_CLASSIFICATION: ANOMALY_CLASSIFICATION_TASK,
    TaskType.ANOMALY_DETECTION: ANOMALY_DETECTION_TASK,
    TaskType.ANOMALY_SEGMENTATION: ANOMALY_SEGMENTATION_TASK,
    TaskType.ANOMALY: ANOMALY_TASK,
    TaskType.INSTANCE_SEGMENTATION: INSTANCE_SEGMENTATION_TASK,
    TaskType.ROTATED_DETECTION: ROTATED_DETECTION_TASK,
    TaskType.KEYPOINT_DETECTION: KEYPOINT_DETECTION_TASK,
}


class ProjectClient:
    """
    Class to manipulate projects on the Intel® Geti™ server, within a certain workspace.
    """

    def __init__(self, session: GetiSession, workspace_id: str):
        self.session = session
        self.workspace_id = workspace_id
        self.base_url = f"workspaces/{workspace_id}/"

    def get_all_projects(self, request_page_size: int = 50, get_project_details: bool = True) -> list[Project]:
        """
        Return a list of projects found on the Intel® Geti™ server

        :param request_page_size: Max number of projects to fetch in a single HTTP
            request. Higher values may reduce the response time of this method when
            there are many projects, but increase the chance of timeout.
        :param get_project_details: True to get all details of the projects on the
            Intel® Geti™, False to fetch only a summary of each project. Set this to
            False if minimizing latency is a concern. Defaults to True
        :return: List of Project objects, containing the project information for each
            project on the Intel® Geti™ server
        """
        # The 'projects' endpoint uses pagination: multiple HTTP may be necessary to
        # fetch the full list of projects
        project_rest_list: list[dict] = []
        while response := self.session.get_rest_response(
            url=f"{self.base_url}projects?limit={request_page_size}&skip={len(project_rest_list)}",
            method="GET",
        ):
            project_rest_list.extend(response["projects"])
            if len(project_rest_list) >= response["project_counts"]:
                break

        project_list = [ProjectRESTConverter.from_dict(project_input=project) for project in project_rest_list]
        if get_project_details:
            project_detail_list: list[Project] = []
            for project in project_list:
                try:
                    project_detail_list.append(self.get_project_by_id(project.id))
                except GetiRequestException as e:
                    if e.status_code == 403:
                        logging.info(f"Unable to access project `{project.name}` details: Unauthorized.")
                        project_detail_list.append(project)
            return project_detail_list
        return project_list

    def get_project_by_name(
        self,
        project_name: str,
    ) -> Project | None:
        """
        Get a project from the Intel® Geti™ server by project_name.

        If multiple projects with the same name exist on the server, this method will
        raise a ValueError. In that case, please use the `ProjectClient.get_project()`
        method and provide a `project_id` to uniquely identify the project.

        :param project_name: Name of the project to get
        :raises: ValueError in case multiple projects with the specified name exist on
            the server, and no `project_id` is provided in order to allow unique
            identification of the project.
        :return: Project object containing the data of the project, if the project is
            found on the server. Returns None if the project doesn't exist.
        """
        project_list = self.get_all_projects(get_project_details=False)
        matches = [project for project in project_list if project.name == project_name]
        if len(matches) == 1:
            return self.get_project_by_id(matches[0].id)
        if len(matches) > 1:
            detailed_matches = [self.get_project_by_id(match.id) for match in matches]
            projects_info = [
                (f"Name: {p.name},\t Type: {p.project_type},\t ID: {p.id},\t creation_date: {p.creation_time}\n")
                for p in detailed_matches
            ]
            raise ValueError(
                f"A total of {len(matches)} projects named `{project_name}` were "
                f"found in the workspace. Unable to uniquely identify the "
                f"desired project. Please provide a `project_id` to ensure the "
                f"proper project is returned. The following projects were found:"
                f"{projects_info}"
            )
        warnings.warn(f"Project with name {project_name} was not found on the server.")
        return None

    def get_or_create_project(
        self,
        project_name: str,
        project_type: str,
        labels: list[list[str] | list[dict[str, Any]]],
    ) -> Project:
        """
        Create a new project with name `project_name` on the cluster, or retrieve
        the data for an existing project with `project_name` if it exists.

        :param project_name: Name of the project
        :param project_type: Type of the project
        :param labels: Nested list of labels
        :return:
        """
        project = self.get_project_by_name(project_name)
        if project is not None:
            logging.info(
                f"Project with name {project_name} exists, continuing with "
                f"exiting project. No new project has been created."
            )
        else:
            project = self.create_project(project_name=project_name, project_type=project_type, labels=labels)
        return project

    def create_project(
        self,
        project_name: str,
        project_type: str,
        labels: list[list[str] | list[dict[str, Any]]],
        keypoint_structure: dict[str, list] | None = None,
    ) -> Project:
        """
        Create a new project with name `project_name` on the cluster, containing
        tasks according to the `project_type` specified. Labels for each task are
        specified in the `labels` parameter, which should be a nested list (each entry
        in the outermost list corresponds to the labels for one of the tasks in the
        project pipeline)

        :param project_name: Name of the project
        :param project_type: Type of the project
        :param labels: Nested list of labels
        :param keypoint_structure: The structure of the keypoints to be used for the project,
            represented as a graph of nodes and edges.
        :raises ValueError: If a project with name `project_name` already exists in
            the workspace
        :return: Project object, as created on the cluster
        """
        project_template = self._create_project_template(
            project_name=project_name,
            project_type=project_type,
            labels=labels,
            keypoint_structure=keypoint_structure,
        )
        project_dict = self.session.get_rest_response(
            url=f"{self.base_url}projects", method="POST", data=project_template
        )
        logging.info("Project created successfully.")
        project = ProjectRESTConverter.from_dict(project_dict)
        self._await_project_ready(project=project)
        return project

    def download_project_info(self, project: Project, path_to_folder: str) -> None:
        """
        Get the project data that can be used for project creation on
        the Intel® Geti™ server. From the returned data, the
        method `ProjectClient.get_or_create_project` can create a project on the
        Intel® Geti™ server. The data is retrieved from the cluster and saved in the
        target folder `path_to_folder`.

        :param project: Project to download the data for
        :param path_to_folder: Target folder to save the project data to.
            Data will be saved as a .json file named "project.json"
        :raises ValueError: If the project with `project_name` is not found on the
            cluster
        """
        # Update the project state
        project = self.get_project_by_id(project.id)
        project_data = ProjectRESTConverter.to_dict(project)
        os.makedirs(path_to_folder, exist_ok=True, mode=0o770)
        project_config_path = os.path.join(path_to_folder, "project.json")
        with open(project_config_path, "w") as file:
            json.dump(project_data, file, indent=4)
        logging.info(f"Project parameters for project '{project.name}' were saved to file {project_config_path}.")

    @staticmethod
    def _add_task(
        project_template: dict,
        task_type: TaskType,
        labels: list[str] | list[dict[str, Any]],
        keypoint_structure: dict[str, list] | None = None,
    ) -> tuple[dict, dict]:
        """
        Add a task to the pipeline in a project template in dictionary form.

        :param project_template: Dictionary representing the project creation data
        :param task_type: Type of the task to be added
        :param labels: Labels to be used for the task
        :param keypoint_structure: The structure of the keypoints to be used for the project,
            represented as a graph of nodes and edges.
        :return:
        """
        new_template = copy.deepcopy(project_template)
        tasks = new_template["pipeline"]["tasks"]
        try:
            task_template = copy.deepcopy(TASK_TYPE_MAPPING[task_type])
        except KeyError as error:
            raise ValueError(f"Task of type {task_type} is currently not supported.") from error

        # Make sure the task title is unique in the pipeline
        task_titles_in_template = [task["title"] for task in tasks]
        unique_task_title = ProjectClient._ensure_unique_task_name(task_template["title"], task_titles_in_template)
        task_template["title"] = unique_task_title
        if task_type == task_type.KEYPOINT_DETECTION:
            if not keypoint_structure:
                raise ValueError("A keypoint detection project must have a keypoint structure.")
            task_template["keypoint_structure"] = keypoint_structure
        if not task_type.is_anomaly:
            label_group_name = f"{unique_task_title.lower()} label group"

            for label in labels:
                label_info = {"name": label, "group": label_group_name} if isinstance(label, str) else label
                task_template["labels"].append(label_info)
        tasks.append(task_template)
        return new_template, task_template

    @staticmethod
    def _add_crop_task(project_template: dict) -> tuple[dict, dict]:
        """
        Add a `crop` task to the pipeline in the project_template.

        :param project_template:
        :return: Tuple containing:
            - A dictionary representing the new project_template, with the crop task
                added to it
            - A dictionary representing the crop task that was added to the template
        """
        new_template = copy.deepcopy(project_template)
        tasks = new_template["pipeline"]["tasks"]
        crop_task = copy.deepcopy(CROP_TASK)
        tasks.append(crop_task)
        return new_template, crop_task

    @staticmethod
    def _add_connection(project_template: dict, to_task: str, from_task: str) -> dict:
        """
        Add a connection between `from_task` and `to_task` in the project_template
        dictionary.

        :param project_template:
        :param to_task:
        :param from_task:
        :return:
        """
        new_template = copy.deepcopy(project_template)
        connections = new_template["pipeline"]["connections"]
        connections.append({"from": from_task, "to": to_task})
        return new_template

    def create_project_from_folder(self, path_to_folder: str, project_name: str | None = None) -> Project:
        """
        Look for a `project.json` file in the folder at `path_to_folder`, and
        create a project using the parameters provided in this file.

        :param path_to_folder: Folder holding the project data
        :param project_name: Optional name of the project. If not specified, the
            project name found in the project configuration in the upload folder
            will be used.
        :return: Project as created on the cluster
        """
        path_to_project = os.path.join(path_to_folder, "project.json")
        if not os.path.isfile(path_to_project):
            raise ValueError(
                f"Unable to find project configuration file at {path_to_project}. "
                f"Please provide a valid path to the folder holding the project data."
            )
        with open(path_to_project) as file:
            project_data = json.load(file)
        project = ProjectRESTConverter.from_dict(project_data)
        if project_name is not None:
            project.name = project_name
        logging.info(f"Creating project '{project.name}' from parameters in configuration file at {path_to_project}.")
        project.prepare_for_post()
        datasets = project.datasets
        created_project = self.get_or_create_project(**project.get_parameters())
        if len(datasets) > 1:
            # Create the additional datasets if needed
            dataset_client = DatasetClient(
                session=self.session,
                workspace_id=self.workspace_id,
                project=created_project,
            )
            for dataset in datasets:
                if dataset.name not in [ds.name for ds in created_project.datasets]:
                    dataset_client.create_dataset(name=dataset.name)
        return created_project

    @staticmethod
    def _is_project_dir(path_to_folder: str) -> bool:
        """
        Check if the folder specified in `path_to_folder` is a directory
        containing valid Intel® Geti™ project data that can be used to upload to an
        Intel® Geti™ server.

        :param path_to_folder: Directory to check
        :return: True if the directory holds project data, False otherwise
        """
        if not os.path.isdir(path_to_folder):
            return False
        path_to_project = os.path.join(path_to_folder, "project.json")
        if not os.path.isfile(path_to_project):
            return False
        try:
            with open(path_to_project) as file:
                project_data = json.load(file)
        except json.decoder.JSONDecodeError:
            return False
        try:
            ProjectRESTConverter.from_dict(project_data)
        except (ValueError, TypeError, KeyError):
            return False
        return True

    def list_projects(self) -> list[Project]:
        """
        Print an overview of all projects that currently exists on the
        cluster, in the workspace managed by the ProjectClient

        NOTE: While this method also returns a list of all the projects, it is
        primarily meant to be used in an interactive environment, such as a
        Jupyter Notebook.

        :return: List of all Projects on the cluster. The returned list is the same as
            the list returned by the `get_all_projects` method
        """
        projects = self.get_all_projects()
        print(f"{len(projects)} projects were found on the platform:\n")
        for project in projects:
            print(" " + project.summary + "\n")
        return projects

    def _create_project_template(
        self,
        project_name: str,
        project_type: str,
        labels: list[list[str] | list[dict[str, Any]]],
        keypoint_structure: dict[str, list] | None = None,
    ) -> dict[str, Any]:
        """
        Create a template dictionary with data for project creation that is ready to
        be sent to  the cluster.

        :param project_name: Name of the project
        :param project_type: Type of the project
        :param labels: Nested list of labels
        :param keypoint_structure: The structure of the keypoints to be used for the project,
            represented as a graph of nodes and edges.
        :return: Dictionary containing the data to create a project named
            `project_name`, of type `project_type` and with labels `labels`
        """
        project_template = copy.deepcopy(BASE_TEMPLATE)
        previous_task_name = "Dataset"
        previous_task_type = TaskType.DATASET
        task_names_in_template: list[str] = [previous_task_name]
        is_first_task = True
        for task_type, task_labels in zip(get_task_types_by_project_type(project_type), labels):
            # Anomaly task reduction introduced in Intel Geti 2.5
            # The last on-premises version of Intel Geti to support legacy anomaly projects is 2.0
            if self.session.version >= GETI_25_VERSION and task_type.is_anomaly and task_type != TaskType.ANOMALY:
                logging.info(f"The {task_type} task is mapped to {TaskType.ANOMALY}.")
                task_type = TaskType.ANOMALY
            if not is_first_task and not previous_task_type.is_global:
                # Add crop task and connections, only for tasks that are not
                # first in the pipeline and are not preceded by a global task
                project_template, crop_task = self._add_crop_task(project_template)
                unique_task_name = self._ensure_unique_task_name(crop_task["title"], task_names_in_template)
                crop_task["title"] = unique_task_name

                project_template = self._add_connection(
                    project_template,
                    to_task=unique_task_name,
                    from_task=previous_task_name,
                )
                previous_task_name = unique_task_name
            project_template, added_task = self._add_task(
                project_template,
                task_type=task_type,
                labels=task_labels,
                keypoint_structure=keypoint_structure,
            )
            task_name = added_task["title"]

            project_template = self._add_connection(project_template, to_task=task_name, from_task=previous_task_name)
            previous_task_name = task_name
            previous_task_type = task_type
            is_first_task = False

        project_template["name"] = project_name
        return project_template

    @staticmethod
    def _ensure_unique_task_name(task_name: str, task_names_in_template: list[str]) -> str:
        """
        Check that the `task_name` passed is not already in the list of
        `task_names_in_template`. If the task_name is already in the list, this method
        will generate a new task_name that is unique

        NOTE: This method updates the list of task_names_in_template with the unique
        task_name that is ensured in this method. This update is done in-place

        :param task_name: Task name to check
        :param task_names_in_template: List of task names that are already taken in the
            project template
        :return: Name of the task that is guaranteed to be unique in the template
        """
        ii = 2
        if task_name in task_names_in_template:
            new_task_name = f"{task_name} {ii}"
            while new_task_name in task_names_in_template:
                ii += 1
                new_task_name = f"{task_name} {ii}"
            task_name = new_task_name
        task_names_in_template.append(task_name)
        return task_name

    def delete_project(self, project: Project, requires_confirmation: bool = True) -> None:
        """
        Delete a project.

        By default, this method will ask for user confirmation before deleting the
        project. This can be overridden by passing `requires_confirmation = False`.

        :param project: Project to delete
        :param requires_confirmation: True to ask for user confirmation before
            deleting the project, False to delete without confirmation. Defaults to
            True
        """
        if requires_confirmation:
            # Update the project details
            project = self.get_project_by_id(project.id)
            if project.datasets is None:
                project.datasets = []
            image_count = 0
            video_count = 0
            for dataset in project.datasets:
                dataset_statistics = self.session.get_rest_response(
                    url=f"{self.base_url}projects/{project.id}/datasets/{dataset.id}/statistics",
                    method="GET",
                )
                if isinstance(dataset_statistics, dict):
                    dataset_overview = dataset_statistics["overview"]
                    image_count += dataset_overview.get("images", 0)
                    video_count += dataset_overview.get("videos", 0)
                else:
                    logging.warning(f"Unable to retrieve statistics for dataset {dataset.name}.")

            user_confirmation = input(
                f"CAUTION: You are about to delete project '{project.name}', "
                f"containing {image_count} images and {video_count}"
                f" videos, from the platform. Are you sure you want to continue? Type "
                f"Y or YES to continue, any other key to cancel."
            )
            if not (user_confirmation.lower() == "yes" or user_confirmation.lower() == "y"):
                logging.info("Aborting project deletion.")
                return
        try:
            self.session.get_rest_response(url=f"{self.base_url}projects/{project.id}", method="DELETE")
        except GetiRequestException as error:
            if error.status_code == 409:
                raise ValueError(
                    f"Project {project.name} is locked for deletion/modification. "
                    f"Please wait until all jobs related to this project are finished "
                    f"or cancel them to allow deletion/modification."
                )
            raise error
        logging.info(f"Project '{project.name}' deleted successfully.")

    def update_labels(
        self,
        project: Project,
        colors_to_update: dict[str, str],
        hotkeys_to_update: dict[str, str],
    ) -> Project:
        """
        Update the color and/or hotkey of one or more labels in the project.

        :param project: Project to update
        :param colors_to_update: Dictionary mapping label names to new colors.
            The keys should be the label names, and the values should be the new colors
            in hex format (e.g., "#FF5733").
        :param hotkeys_to_update: Dictionary mapping label names to new hotkeys.

        :return: Updated Project instance with the new label colors applied
        """
        # Validate that the provided label names exist in the project
        label_names = {label.name for task in project.get_trainable_tasks() for label in task.labels}
        for label_name in chain(colors_to_update.keys(), hotkeys_to_update.keys()):
            if label_name not in label_names:
                raise ValueError(
                    f"Label '{label_name}' does not exist in the project '{project.name}'. "
                    "Please provide valid label names."
                )

        # Validate that the provided colors are valid hex colors
        for label_name, color in colors_to_update.items():
            if not isinstance(color, str) or not color.startswith("#"):
                raise ValueError(
                    f"Invalid color format for label '{label_name}': {color}. "
                    "Colors should be in hex format (e.g., '#FF5733')."
                )

        # Iterate over the labels in the project and update their colors/hotkeys as needed
        project.prepare_for_post()
        project_data = project.to_dict()
        for task in project.get_trainable_tasks():
            task_id = task.id
            task_data = next(
                (t for t in project_data["pipeline"]["tasks"] if t["id"] == task_id),
                None,
            )
            if task_data is None:
                continue
            for label in task_data["labels"]:
                if label["name"] in colors_to_update:
                    label["color"] = colors_to_update[label["name"]]
                if label["name"] in hotkeys_to_update:
                    label["hotkey"] = hotkeys_to_update[label["name"]]
        remove_null_fields(project_data)

        response = self.session.get_rest_response(
            url=f"{self.base_url}projects/{project.id}", method="PUT", data=project_data
        )
        logging.info(f"Updated labels of project '{project.name}'.")
        return ProjectRESTConverter.from_dict(response)

    def add_labels(
        self,
        labels: list[str] | list[dict[str, Any]],
        project: Project,
        task: Task | None = None,
        revisit_affected_annotations: bool = False,
    ) -> Project:
        """
        Add the `labels` to the project labels. For a project with multiple tasks,
        the `task` parameter can be used to specify the task for which the labels
        should be added.

        :param labels: List of labels to add. Can either be a list of strings
            representing label names, or a list of dictionaries representing label
            properties
        :param project: Project to which the labels should be added
        :param task: Optional Task to add the labels for. Can be left as None for a
            single task project, but is required for a task chain project
        :param revisit_affected_annotations: True to make sure that the server will
            assign a `to_revisit` status to all annotations linked to the label(s)
            that are added. False to not revisit any potentially linked annotations.
        :return: Updated Project instance with the new labels added to it
        """
        # Validate inputs
        if task is not None and task not in project.get_trainable_tasks():
            raise ValueError(f"The provided task {task} is not part of project {project}.")
        if len(project.get_trainable_tasks()) > 1 and task is None:
            raise ValueError(
                f"Project '{project}' is a task-chain project, but no target task was "
                f"specified. Please provide a valid task to perform label addition."
            )
        task_index = 0 if task is None else project.get_trainable_tasks().index(task)

        # Update the list of labels for the task
        label_list = project.get_labels_per_task()[task_index]
        existing_colors = [label["color"] for label in label_list]
        formatted_labels: list[dict[str, Any]] = []
        for label_data in labels:
            new_color = generate_unique_label_color(existing_colors)
            if isinstance(label_data, str):
                label_dict = {
                    "name": label_data,
                    "color": new_color,
                    "group": label_data,
                }
            elif isinstance(label_data, dict):
                label_name = label_data.get("name", None)
                if label_name is None:
                    raise ValueError(f"Unable to add label {label_data}: Label name not specified.")
                if "color" not in label_data:
                    label_data.update({"color": new_color})
                if "group" not in label_data:
                    label_data.update({"group": label_name})
                label_dict = label_data
            else:
                raise ValueError(
                    f"Invalid input label format found for label {label_data}. Please "
                    f"provide either the label name as a string or a dictionary of "
                    f"label properties."
                )
            label_dict.update({"revisit_affected_annotations": revisit_affected_annotations})
            formatted_labels.append(label_dict)
            existing_colors.append(new_color)
        label_list.extend(formatted_labels)

        # Prepare data for the update request
        project.prepare_for_post()
        project_data = project.to_dict()
        task_id = project.get_trainable_tasks()[task_index].id
        task_data = next(task for task in project_data["pipeline"]["tasks"] if task["id"] == task_id)
        task_data["labels"] = label_list
        remove_null_fields(project_data)
        logging.info(project_data)
        response = self.session.get_rest_response(
            url=f"{self.base_url}projects/{project.id}", method="PUT", data=project_data
        )
        return ProjectRESTConverter.from_dict(response)

    def _await_project_ready(self, project: Project, timeout: int = 5, interval: int = 1) -> None:
        """
        Await the completion of the project creation process on the Intel® Geti™ server

        :param project: Project object representing the project
        :param timeout: Time (in seconds) after which the method will time out and
            raise an error
        :param interval: Interval (in seconds) between status checks of the project
        :raises: TimeoutError if the project does not become ready after the specified
            timeout
        """
        t_start = time.time()
        error: BaseException | None = None
        while time.time() - t_start < timeout:
            try:
                self.session.get_rest_response(url=f"{self.base_url}projects/{project.id}/status", method="GET")
                return
            except GetiRequestException as latest_error:
                time.sleep(interval)
                error = latest_error
        raise TimeoutError(f"Project has not become ready within the specified timeout ({timeout} seconds).") from error

    def get_project_by_id(self, project_id: str) -> Project | None:
        """
        Get a project from the Intel® Geti™ server by project_id.

        :param project_id: ID of the project to get
        :return: Project object containing the data of the project, if the project is
            found on the server. Returns None if the project doesn't exist
        """
        response = self.session.get_rest_response(url=f"{self.base_url}projects/{project_id}", method="GET")
        return ProjectRESTConverter.from_dict(response)

    def get_project(
        self,
        project_name: str | None = None,
        project_id: str | None = None,
        project: Project | None = None,
    ) -> Project | None:
        """
        Get a project from the Intel® Geti™ server by project_name or project_id, or
        update a provided Project object with the latest data from the server.

        :param project_name: Name of the project to get
        :param project_id: ID of the project to get
        :param project: Project object to update with the latest data from the server
        :return: Project object containing the data of the project, if the project is
            found on the server. Returns None if the project doesn't exist
        """
        # The method prioritize the parameters in the following order:
        if project_id is not None:
            return self.get_project_by_id(project_id)
        if project is not None:
            if project.id is not None:
                return self.get_project_by_id(project.id)
            return self.get_project_by_name(project_name=project.name)
        if project_name is not None:
            return self.get_project_by_name(project_name=project_name)
        # No parameters provided
        # Warn the user and return None
        warnings.warn("At least one of the parameters `project_name`, `project_id`, or `project` must be provided.")
        return None
