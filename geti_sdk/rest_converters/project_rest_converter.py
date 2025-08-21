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
from typing import Any

from geti_sdk.data_models import Project
from geti_sdk.data_models.utils import remove_null_fields
from geti_sdk.utils import deserialize_dictionary


class ProjectRESTConverter:
    """
    Class that handles conversion of Intel® Geti™ REST output for project entities to
    objects and vice versa.
    """

    @classmethod
    def from_dict(cls, project_input: dict[str, Any]) -> Project:
        """
        Create a Project from a dictionary representing a project, as
        returned by the /projects endpoint on the Intel® Geti™ platform.

        :param project_input: Dictionary representing a project, as returned by the
            Intel® Geti™ server
        :return: Project object representing the project given in `project_input`
        """
        prepared_project = copy.deepcopy(project_input)
        for connection in prepared_project["pipeline"]["connections"]:
            from_ = connection.pop("from", None)
            if from_ is not None:
                connection.update({"from_": from_})
        return deserialize_dictionary(prepared_project, output_type=Project)

    @classmethod
    def to_dict(cls, project: Project, deidentify: bool = True) -> dict[str, Any]:
        """
        Convert the `project` to its dictionary representation.
        This functions removes database UID's and optional fields that are `None`
        from the output dictionary, to make the output more compact and improve
        readability.

        :param project: Project to convert to dictionary
        :param deidentify: True to remove all unique database ID's from the project
            and it's child entities, False to keep the ID's intact. Defaults to True,
            which is useful for project import/export
        :return:
        """
        if deidentify:
            project.deidentify()
        project_data = project.to_dict()
        remove_null_fields(project_data)
        return project_data
